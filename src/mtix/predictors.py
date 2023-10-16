import copy
import re


QUERY_TEMPLATE = "Journal: {journal_title} | Title: {title} | Abstract: {abstract}"


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


class CnnModelTop100Predictor:
    def __init__(self, tensorflow_endpoint):
        self.tensorflow_endpoint = tensorflow_endpoint

    def predict(self, citation_data_lookup):
        citation_data_list = list(citation_data_lookup.values())
        instances = [{ key: value for key, value in citation_data.items() if key not in ["pmid", "journal_title"] } for citation_data in citation_data_list]
        instances = replace_brackets(instances)
        data = { "instances": instances }
        response = self.tensorflow_endpoint.predict(data)
        predictions = response["predictions"]
        pmids = [citation_data["pmid"] for citation_data in citation_data_list]
        top_results = { str(pmid): { str(int(label_id)): float(score) for label_id, score in citation_top_results} for pmid, citation_top_results in zip(pmids, predictions)}
        return top_results


class PointwiseModelTopNPredictor:

    def __init__(self, huggingface_endpoint, passage_lookup, top_n):
        self.huggingface_endpoint = huggingface_endpoint
        self.passage_lookup = passage_lookup
        self.top_n = top_n
 
    def predict(self, citation_data_lookup, input_top_results):
        pmid_list, label_id_list, input_list = self._create_inputs(citation_data_lookup, input_top_results)
        score_list = self._predict_internal(input_list)
        output_top_results = self._create_top_results(pmid_list, label_id_list, score_list)
        return output_top_results

    def _create_citation_inputs(self, citation_data, citation_top_results):
        input_list = []
        label_id_list = []
        for p_id, _ in sorted(citation_top_results.items(), key=lambda x: x[1], reverse=True)[:self.top_n]:
            label_id = int(p_id)
            query = QUERY_TEMPLATE.format(journal_title=citation_data["journal_title"], title=citation_data["title"], abstract=citation_data["abstract"])
            passage = self.passage_lookup[label_id]
            input_list.append([[query, passage]])
            label_id_list.append(label_id)
        return input_list, label_id_list

    def _create_inputs(self, citation_data_lookup, input_top_results):
        pmid_list = []
        label_id_list = []
        input_list = []
        for q_id in input_top_results:
            pmid = int(q_id)
            citation_data = citation_data_lookup[pmid]
            citation_top_results = input_top_results[q_id]
            citation_input_list, citation_label_id_list = self._create_citation_inputs(citation_data, citation_top_results)
            pmid_list.extend([pmid]*self.top_n)
            label_id_list.extend(citation_label_id_list)
            input_list.extend(citation_input_list)
        return pmid_list, label_id_list, input_list

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

    def _predict_internal(self, input_list):
        input_data = { "inputs": input_list, "parameters": { "max_length": 512, "padding": "max_length", "truncation": "longest_first", "return_all_scores": True }, }
        score_list = self.huggingface_endpoint.predict(input_data)
        score_list = [float(label_score["score"]) for label_score_list in score_list for label_score in label_score_list if label_score["label"] == "LABEL_1"]
        return score_list


class ListwiseModelTopNPredictor:

    def __init__(self, huggingface_endpoint, passage_lookup, top_n):
        self.huggingface_endpoint = huggingface_endpoint
        self.passage_lookup = passage_lookup
        self.top_n = top_n

    def predict(self, citation_data_lookup, input_top_results):
        input_data, pmid_list, top_label_ids = self._create_input_data(citation_data_lookup, input_top_results)
        score_lookup = self._predict_internal(input_data)
        output_top_results = self._create_top_results(pmid_list, top_label_ids, score_lookup)
        return output_top_results

    def _create_input_data(self, citation_data_lookup, input_top_results):
        input_data = { "inputs": [], "parameters": {} }
        pmid_list = []
        top_label_ids = []
        for q_id in input_top_results:
            pmid = int(q_id)
            pmid_list.append(pmid)

            citation_top_label_ids = [int(p_id) for p_id, _ in sorted(input_top_results[q_id].items(), key=lambda x: x[1], reverse=True)[:self.top_n]]
            top_label_ids.append(citation_top_label_ids)

            citation_data = citation_data_lookup[pmid]
            query = "|" + QUERY_TEMPLATE.format(journal_title=citation_data["journal_title"], title=citation_data["title"], abstract=citation_data["abstract"])
         
            combined_passages = ""
            for label_id in citation_top_label_ids:
                passage = self.passage_lookup[label_id]
                combined_passages += "|"
                combined_passages += passage

            input_data["inputs"].append([[query, combined_passages]])
        
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
        predictions = self.huggingface_endpoint.predict(input_data)
        prediction_count = len(predictions)
        score_lookup = {}
        for citation_idx in range(prediction_count):
            score_lookup[citation_idx] = {}
            for citation_predictions in predictions[citation_idx]:
                label_idx = citation_predictions["index"]
                score = citation_predictions["score"]
                score_lookup[citation_idx][label_idx] = score
        return score_lookup


class SubheadingPredictor:
    def __init__(self, input_parser, data_santizer, subheading_endpoint, subheading_name_lookup):
        self.parser = input_parser
        self.santizer = data_santizer
        self.subheading_endpoint = subheading_endpoint
        self.subheading_name_lookup = subheading_name_lookup

    def predict(self, mesh_heading_predictions):
        data = self._create_input_data(mesh_heading_predictions)
        response = self.subheading_endpoint.predict(data)
        result_lookup = self._create_result_lookup(response)
        predictions = self._attach_subheadings(result_lookup, mesh_heading_predictions)
        return predictions

    def _attach_subheadings(self, result_lookup, mesh_heading_predictions):
        mesh_heading_predictions = copy.deepcopy(mesh_heading_predictions)
        for citation_prediction in mesh_heading_predictions:
            pmid = citation_prediction["PMID"]
            for mesh_heading_prediction in citation_prediction["Indexing"]:
                if self._subheadings_allowed(mesh_heading_prediction):
                    dui = mesh_heading_prediction["ID"]
                    subheading_list = []
                    mesh_heading_prediction["Subheadings"] = subheading_list
                    for qui, score in sorted(result_lookup[pmid][dui].items(), key=lambda x: x[1], reverse=True):
                        subheading_list.append({
                            "ID": qui,
                            "IM": "NO",
                            "Name": self.subheading_name_lookup[qui],
                            "Reason": "ML Score",
                            "Score": round(score, 3)
                            })
        return mesh_heading_predictions

    def _create_input_data(self, mesh_heading_predictions):
        instances = []
        for citation_prediction in mesh_heading_predictions:
            encoded_citation_xml = citation_prediction["text-gz-64"]
            citation_data = self.parser.parse_data(encoded_citation_xml)
            self.santizer.sanitize(citation_data)
            citation_data = { key: value for key, value in citation_data.items() if key not in ["journal_title"] }
            citation_data["pmid"] = str(citation_data["pmid"])
            for mesh_heading_prediction in citation_prediction["Indexing"]:
                if self._subheadings_allowed(mesh_heading_prediction):
                    citation_data_copy = copy.deepcopy(citation_data)
                    citation_data_copy["main_heading_ui"] = mesh_heading_prediction["ID"]
                    instances.append(citation_data_copy)
        instances = replace_brackets(instances)
        data = { "instances": instances }
        return data

    def _create_result_lookup(self, response):
        result_lookup = {}
        for pmid, dui, qui, score in response["predictions"]:
            pmid = int(pmid)
            if pmid not in result_lookup:
                result_lookup[pmid] = {}
            if dui not in result_lookup[pmid]:
                result_lookup[pmid][dui] = {}
            if len(qui.strip()) > 0:
                result_lookup[pmid][dui][qui] = float(score)
        return result_lookup
    
    def _subheadings_allowed(self, mesh_heading_prediction):
        _type = mesh_heading_prediction["Type"].strip().lower()
        if (_type == "descriptor" or _type == "check tag"):
            allowed = True
        else:
            allowed = False
        return allowed