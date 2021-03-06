import dateutil.parser
import re
from .utils import average_top_results, base64_decode
import xml.etree.ElementTree as ET


class CitationDataSanitizer:

    def __init__(self, max_year):
        self.max_year = max_year
        
    def sanitize(self, citation_data):
        for pmid in citation_data:
            pmid_data = citation_data[pmid]
            if pmid_data["journal_nlmid"] is None:
                pmid_data["journal_nlmid"] = "<unknown>"
            if pmid_data["pub_year"] is None:
                if pmid_data["year_completed"] is not None:
                    pmid_data["pub_year"] = pmid_data["year_completed"]
                else:
                    pmid_data["pub_year"] = self.max_year
            if pmid_data["year_completed"] is None:
                pmid_data["year_completed"] = self.max_year
        return citation_data


# Note: top results are not always sorted
class DescriptorPredictionPipeline:

    def __init__(self, input_data_parser, citation_data_sanitizer, cnn_model_top_n_predictor, pointwise_model_top_n_predictor, listwise_model_top_n_predictor, results_formatter):
        self.input_data_parser = input_data_parser
        self.citation_data_sanitizer = citation_data_sanitizer
        self.cnn_model_top_n_predictor = cnn_model_top_n_predictor
        self.pointwise_model_top_n_predictor = pointwise_model_top_n_predictor
        self.listwise_model_top_n_predictor = listwise_model_top_n_predictor
        self.results_formatter = results_formatter

    def predict(self, input_data):
        citation_data = self.input_data_parser.parse(input_data)
        citation_data = self.citation_data_sanitizer.sanitize(citation_data)
        cnn_results = self.cnn_model_top_n_predictor.predict(citation_data)
        pointwise_results = self.pointwise_model_top_n_predictor.predict(citation_data, cnn_results)
        pointwsie_avg_results = average_top_results(cnn_results, pointwise_results)
        listwise_results = self.listwise_model_top_n_predictor.predict(citation_data, pointwsie_avg_results)
        listwise_avg_results = average_top_results(pointwsie_avg_results, listwise_results)
        predictions = self.results_formatter.format(listwise_avg_results)
        return predictions


class SubheadingAttachmentPipeline:

    def __init__(self, input_data_parser, citation_data_sanitizer, subheading_predictor):
        self.input_data_parser = input_data_parser
        self.citation_data_sanitizer = citation_data_sanitizer
        self.subheading_predictor = subheading_predictor

    def predict(self, input_data, descriptor_predictions):
        citation_data = self.input_data_parser.parse(input_data)
        citation_data = self.citation_data_sanitizer.sanitize(citation_data)
        predictions = self.subheading_predictor.predict(citation_data, descriptor_predictions)
        return predictions


class IndexingPipeline:

    def __init__(self, input_data_parser, citation_data_sanitizer, cnn_model_top_n_predictor, pointwise_model_top_n_predictor, listwise_model_top_n_predictor, results_formatter, subheading_predictor):
        self.descriptor_prediction_pipeline = DescriptorPredictionPipeline(input_data_parser, citation_data_sanitizer, cnn_model_top_n_predictor, pointwise_model_top_n_predictor, listwise_model_top_n_predictor, results_formatter)
        self.subheading_attachment_pipeline = SubheadingAttachmentPipeline(input_data_parser, citation_data_sanitizer, subheading_predictor)

    def predict(self, input_data):
        descriptor_predictions = self.descriptor_prediction_pipeline.predict(input_data)
        predictions = self.subheading_attachment_pipeline.predict(input_data, descriptor_predictions)
        return predictions


class MedlineDateParser:
    def extract_pub_year(self, text):
        pub_year = text[:4]
        try:
            pub_year = int(pub_year)
        except ValueError:
            match = re.search(r"\d{4}", text)
            if match:
                pub_year = match.group(0)
                pub_year = int(pub_year)
            else:
                try:
                    pub_year = dateutil.parser.parse(text, fuzzy=True).date().year
                except:
                    pub_year = None
        return pub_year


class MtiJsonResultsFormatter:
    def __init__(self, desc_name_lookup, dui_lookup, threshold):
        self.desc_name_lookup = desc_name_lookup
        self.dui_lookup = dui_lookup
        self.threshold = threshold

    def format(self, results):
        mti_json = []
        for q_id in results:
            pmid = int(q_id)
            citation_predictions = { "PMID": pmid, "Indexing": [] }
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


class PubMedXmlInputDataParser:
    def __init__(self, medline_date_parser):
        self.medline_date_parser = medline_date_parser
    
    def parse(self, input_data):
        if input_data is None:
            raise ValueError("Input data cannot be None.")
        elif type(input_data) is not list:
            raise ValueError("Input data must be a list.")

        citation_data_lookup = {}
        for item in input_data:
            citation_xml = item["data"]
            citation_xml = base64_decode(citation_xml)
            citation_data = self._parse_xml(citation_xml)
            citation_data_lookup[citation_data["pmid"]] = citation_data
        return citation_data_lookup

    def _parse_xml(self, citation_xml):
        medline_citation_node = ET.fromstring(citation_xml)

        pmid_node = medline_citation_node.find("PMID")
        pmid = pmid_node.text.strip()
        pmid = int(pmid)

        title = ""
        title_node = medline_citation_node.find("Article/ArticleTitle") 
        title_text = ET.tostring(title_node, encoding="unicode", method="text")
        if title_text is not None:
            title = title_text.strip()
                
        abstract = ""
        abstract_node = medline_citation_node.find("Article/Abstract")
        if abstract_node is not None:
            abstract_text_node_list = abstract_node.findall("AbstractText")
            for abstract_text_node in abstract_text_node_list:
                if "Label" in abstract_text_node.attrib:
                    if len(abstract) > 0:
                        abstract += " "
                    abstract += abstract_text_node.attrib["Label"].strip() + ": "
                abstract_text = ET.tostring(abstract_text_node, encoding="unicode", method="text")
                if abstract_text is not None:
                    abstract += abstract_text.strip()

        journal_nlmid = None
        journal_nlmid_node = medline_citation_node.find("MedlineJournalInfo/NlmUniqueID")
        if journal_nlmid_node is not None:
            journal_nlmid = journal_nlmid_node.text.strip()

        journal_title = ""
        journal_title_node = medline_citation_node.find("Article/Journal/Title")
        if journal_title_node is not None:
            journal_title_text = ET.tostring(journal_title_node, encoding="unicode", method="text")
            if journal_title_text is not None:
                journal_title = journal_title_text.strip() 

        pub_year = None
        medline_date_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date_node is not None:
            medline_date_text = ET.tostring(medline_date_node, encoding="unicode", method="text")
            if medline_date_text is not None:
                medline_date_text = medline_date_text.strip()
                pub_year = self.medline_date_parser.extract_pub_year(medline_date_text)
        else:
            pub_year_node = medline_citation_node.find("Article/Journal/JournalIssue/PubDate/Year")
            pub_year = pub_year_node.text.strip()
            pub_year = int(pub_year)

        year_completed = None
        date_completed_node = medline_citation_node.find("DateCompleted")
        if date_completed_node is not None:
            year_completed_node = date_completed_node.find("Year")
            year_completed = year_completed_node.text.strip()
            year_completed = int(year_completed)
        
        citation_data = {
                    "pmid": pmid, # int, Not None
                    "title": title, # str Not None, possibly ""
                    "abstract": abstract, # str Not None, possibly ""
                    "journal_nlmid": journal_nlmid, # str, Possibly None
                    "journal_title": journal_title, # str, Not None, possibly ""
                    "pub_year": pub_year, # int, Possibly None
                    "year_completed": year_completed, # int, Possibly None
                    }

        return citation_data