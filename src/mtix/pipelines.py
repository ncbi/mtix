from .utils import average_top_results, Base64Helper, MedlineDateParser, PubMedXmlParser


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

    def __init__(self, descriptor_prediction_pipeline, subheading_predictor):
        self.descriptor_prediction_pipeline = descriptor_prediction_pipeline
        self.subheading_predictor = subheading_predictor

    def predict(self, input_data):
        descriptor_prediction_result = self.descriptor_prediction_pipeline.predict(input_data)
        predictions = self.subheading_predictor.predict(descriptor_prediction_result)
        return predictions


class MtiJsonResultsFormatter:
    def __init__(self, desc_name_lookup, dui_lookup, threshold):
        self.desc_name_lookup = desc_name_lookup
        self.dui_lookup = dui_lookup
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
                    name = self.desc_name_lookup[label_id]
                    ui = self.dui_lookup[label_id]
                    citation_predictions["Indexing"].append({
                        "Term": name, 
                        "Type": "Descriptor", 
                        "ID": ui, 
                        "IM": "NO", 
                        "Reason": f"score: {score:.3f}"})
        return mti_json