from .utils import average_top_results


# Note: top results are not always sorted
class MeshHeadingPredictionPipeline:

    def __init__(self, input_data_parser, citation_data_sanitizer, cnn_model_top_n_predictor, pointwise_model_top_n_predictor, listwise_model_top_n_predictor, results_formatter):
        self.input_data_parser = input_data_parser
        self.citation_data_sanitizer = citation_data_sanitizer
        self.cnn_model_top_n_predictor = cnn_model_top_n_predictor
        self.pointwise_model_top_n_predictor = pointwise_model_top_n_predictor
        self.listwise_model_top_n_predictor = listwise_model_top_n_predictor
        self.results_formatter = results_formatter

    def predict(self, input_data):
        citation_data_list = self.input_data_parser.parse(input_data)
        self.citation_data_sanitizer.sanitize_list(citation_data_list)
        citation_data_lookup = {citation_data["pmid"]: citation_data for citation_data in citation_data_list}
        cnn_results = self.cnn_model_top_n_predictor.predict(citation_data_lookup)
        pointwise_results = self.pointwise_model_top_n_predictor.predict(citation_data_lookup, cnn_results)
        pointwsie_avg_results = average_top_results(cnn_results, pointwise_results)
        listwise_results = self.listwise_model_top_n_predictor.predict(citation_data_lookup, pointwsie_avg_results)
        listwise_avg_results = average_top_results(pointwsie_avg_results, listwise_results)
        input_data_lookup = { item["uid"]: item["data"] for item in input_data}
        predictions = self.results_formatter.format(input_data_lookup, listwise_avg_results)
        return predictions

            
class IndexingPipeline:

    def __init__(self, mesh_heading_prediction_pipeline, subheading_predictor):
        self.mesh_heading_prediction_pipeline = mesh_heading_prediction_pipeline
        self.subheading_predictor = subheading_predictor

    def predict(self, input_data):
        mesh_heading_prediction_result = self.mesh_heading_prediction_pipeline.predict(input_data)
        predictions = self.subheading_predictor.predict(mesh_heading_prediction_result)
        return predictions


class MtiJsonResultsFormatter:
    def __init__(self, name_lookup, type_lookup, ui_lookup, threshold):
        self.name_lookup = name_lookup
        self.type_lookup = type_lookup
        self.ui_lookup = ui_lookup
        self.threshold = threshold

    def format(self, input_data_lookup, results):
        mti_json = []
        for q_id in results:
            pmid = int(q_id)
            citation_predictions = { "PMID": pmid, "text-gz-64": input_data_lookup[pmid], "Indexing": [] }
            mti_json.append(citation_predictions)
            for p_id, score in sorted(results[q_id].items(), key=lambda x: x[1], reverse=True):
                if score >= self.threshold:
                    label_id = int(p_id)
                    name = self.name_lookup[label_id]
                    _type = self.type_lookup[label_id]
                    ui = self.ui_lookup[label_id]
                    citation_predictions["Indexing"].append({
                        "Term": name, 
                        "Type": _type, 
                        "ID": ui, 
                        "IM": "NO", 
                        "Reasons": [{
                            "Kind": "MTIX",
                            "Score": round(score, 3) }]
                        })
        return mti_json