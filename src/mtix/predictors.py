import re
from copy import deepcopy


QUERY_TEMPLATE = "2017-2021|{journal_title}|{title}|{abstract}"


class CnnModelTop100Predictor:
    def __init__(self, tensorflow_predictor):
        self.tensorflow_predictor = tensorflow_predictor

    @staticmethod
    def replace_brackets(citation_data_list):
        # Currently there is a bug with the SageMaker API that throws an error 
        # when the input data includes a pattern like "] [". If this pattern 
        # is found in the abstract, make the substitutions [ -> ( and ] -> ).
        for citation_data in citation_data_list:
            for key in ["title", "abstract"]:
                if key in citation_data:
                    if re.search(r'\]\s*\[', citation_data[key]):
                        citation_data[key] = re.sub(r'\[', '(', citation_data[key])
                        citation_data[key] = re.sub(r'\]', ')', citation_data[key])

        return citation_data_list

    def predict(self, citation_data_lookup):
        # Copy the citation data to avoid modifying the input data for other models.
        citation_data_copy = deepcopy(citation_data_lookup)
        citation_data_list = list(citation_data_copy.values())
        citation_data_list = CnnModelTop100Predictor.replace_brackets(citation_data_list)

        data = { "instances": [{ key: value for key, value in citation_data.items() if key not in ["pmid", "journal_title"] } for citation_data in citation_data_list] }
        response = self.tensorflow_predictor.predict(data)
        predictions = response["predictions"]
        pmids = [citation_data["pmid"] for citation_data in citation_data_list]
        top_results = { str(pmid): { str(int(desc_id)): float(score) for desc_id, score in citation_top_results} for pmid, citation_top_results in zip(pmids, predictions)}
        return top_results


class PointwiseModelTopNPredictor:

    def __init__(self, huggingface_predictor, desc_name_lookup, top_n, batch_size=None):
        self.huggingface_predictor = huggingface_predictor
        self.desc_name_lookup = desc_name_lookup
        self.top_n = top_n
        if batch_size is None:
            self.batch_size = top_n
        else:
            self.batch_size = batch_size

    def predict(self, citation_data_lookup, input_top_results):
        pmid_list = []
        label_id_list = []
        score_list = []
        queue = []
        for q_id in input_top_results:
            pmid = int(q_id)
            citation_data = citation_data_lookup[pmid]
            citation_top_results = input_top_results[q_id]
            citation_input_list, citation_label_id_list = self._create_citation_inputs(citation_data, citation_top_results)
            pmid_list.extend([pmid]*self.top_n)
            label_id_list.extend(citation_label_id_list)
            queue.extend(citation_input_list)
            while len(queue) >= self.batch_size:
                batch_inputs = queue[:self.batch_size]
                queue = queue[self.batch_size:]
                scores = self._batch_predict(batch_inputs)
                score_list.extend(scores)
        if len(queue) > 0:
            batch_inputs = queue[:self.batch_size]
            queue = queue[self.batch_size:]
            scores = self._batch_predict(batch_inputs)
            score_list.extend(scores)
        output_top_results = self._create_top_results(pmid_list, label_id_list, score_list)
        return output_top_results

    def _create_citation_inputs(self, citation_data, citation_top_results):
        input_list = []
        label_id_list = []
        for p_id, _ in sorted(citation_top_results.items(), key=lambda x: x[1], reverse=True)[:self.top_n]:
            label_id = int(p_id)
            query = QUERY_TEMPLATE.format(journal_title=citation_data["journal_title"], title=citation_data["title"], abstract=citation_data["abstract"])
            passage = self.desc_name_lookup[label_id]
            input_list.append([[query, passage]])
            label_id_list.append(label_id)
        return input_list, label_id_list

    def _create_top_results(self, pmid_list, label_id_list, score_list):
        top_results = {}
        result_count = len(score_list)
        for idx in range(result_count):
            pmid = pmid_list[idx]
            label_id = label_id_list[idx]
            score = score_list[idx]
            q_id = str(pmid)
            p_id = str(label_id)
            if q_id not in top_results:
                top_results[q_id] = {}
            top_results[q_id][p_id] = score
        return top_results

    def _batch_predict(self, batch_inputs):
        batch_input_data = { "inputs": batch_inputs, "parameters": {"max_length": 512, "padding": "max_length", "truncation": "longest_first", "return_all_scores": True, "batch_size": self.batch_size }, }
        batch_score_list = self.huggingface_predictor.predict(batch_input_data)
        batch_score_list = [float(label_score["score"]) for label_score_list in batch_score_list for label_score in label_score_list if label_score["label"] == "LABEL_1"]
        return batch_score_list


class ListwiseModelTopNPredictor:

    def __init__(self, huggingface_predictor, desc_name_lookup, top_n):
        self.huggingface_predictor = huggingface_predictor
        self.desc_name_lookup = desc_name_lookup
        self.top_n = top_n

    def predict(self, citation_data_lookup, input_top_results):
        input_data, pmid_list, top_label_ids = self._create_input_data(citation_data_lookup, input_top_results)
        score_lookup = self._predict_internal(input_data)
        output_top_results = self._create_top_results(pmid_list, top_label_ids, score_lookup)
        return output_top_results

    def _create_input_data(self, citation_data_lookup, input_top_results):
        input_data = { "inputs": [], "parameters": { "batch_size": len(citation_data_lookup)}, }
        pmid_list = []
        top_label_ids = []
        for q_id in input_top_results:
            pmid = int(q_id)
            pmid_list.append(pmid)

            citation_top_label_ids = [int(p_id) for p_id, _ in sorted(input_top_results[q_id].items(), key=lambda x: x[1], reverse=True)[:self.top_n]]
            top_label_ids.append(citation_top_label_ids)

            citation_data = citation_data_lookup[pmid]
            query = "|" + QUERY_TEMPLATE.format(journal_title=citation_data["journal_title"], title=citation_data["title"], abstract=citation_data["abstract"])
         
            passage = ""
            for label_id in citation_top_label_ids:
                desc_name = self.desc_name_lookup[label_id]
                passage += "|"
                passage += desc_name

            input_data["inputs"].append([[query, passage]])
        
        return input_data, pmid_list, top_label_ids

    def _create_top_results(self, pmid_list, top_label_ids, score_lookup):
        top_results = {}
        citation_count = len(pmid_list)
        for citation_idx in range(citation_count):
            pmid = pmid_list[citation_idx]
            q_id = str(pmid)
            top_results[q_id] = {}
            for label_idx in range(self.top_n):
                label_id = top_label_ids[citation_idx][label_idx]
                score = score_lookup[citation_idx][label_idx]
                p_id = str(label_id) 
                top_results[q_id][p_id] = score
        return top_results

    def _predict_internal(self, input_data):
        predictions = self.huggingface_predictor.predict(input_data)
        prediction_count = len(predictions)
        score_lookup = {}
        for citation_idx in range(prediction_count):
            score_lookup[citation_idx] = {}
            for citation_predictions in predictions[citation_idx]:
                label_idx = citation_predictions["index"]
                score = citation_predictions["score"]
                score_lookup[citation_idx][label_idx] = score
        return score_lookup