from .data import *
from mtix.pipelines import MeshHeadingPredictionPipeline, IndexingPipeline, MtiJsonResultsFormatter
from mtix.predictors import CnnModelTop100Predictor, PointwiseModelTopNPredictor, ListwiseModelTopNPredictor, SubheadingPredictor
from mtix.utils import CitationDataSanitizer, PubMedXmlInputDataParser
import pytest
from unittest import skip, TestCase
from unittest.mock import MagicMock, Mock


THRESHOLD = 0.475


@pytest.mark.unit
class TestDescriptorPredictionPipeline(TestCase):

    def setUp(self):
        max_year = 2021
        input_data_parser = PubMedXmlInputDataParser()
        self.sanitizer = CitationDataSanitizer(max_year)
        self.sanitizer.sanitize_list = Mock(wraps=self.sanitizer.sanitize_list)
        self.cnn_predictor = CnnModelTop100Predictor(None)
        self.cnn_predictor.predict = MagicMock(return_value=CNN_RESULTS)
        self.pointwise_predictor = PointwiseModelTopNPredictor(None, {}, 100)
        self.pointwise_predictor.predict = MagicMock(return_value=POINTWISE_RESULTS)
        self.listwise_predictor = ListwiseModelTopNPredictor(None, {}, 50)
        self.listwise_predictor.predict = MagicMock(return_value=LISTWISE_RESULTS)
        self.results_formatter = MtiJsonResultsFormatter(DESC_NAME_LOOKUP, DUI_LOOKUP, THRESHOLD)
        self.results_formatter.format = Mock(wraps=self.results_formatter.format)
        self.pipeline = MeshHeadingPredictionPipeline(input_data_parser, self.sanitizer, self.cnn_predictor, self.pointwise_predictor, self.listwise_predictor, self.results_formatter)

    def test_predict(self):
        predictions = self.pipeline.predict(PUBMED_XML_INPUT_DATA)
        
        self.assertEqual(predictions, EXPECTED_DESCRIPTOR_PREDICTIONS, "Predictions do not match expected result.")

        self.sanitizer.sanitize_list.assert_called_once_with(list(EXPECTED_CITATION_DATA_LOOKUP.values()))
        self.cnn_predictor.predict.assert_called_once_with(EXPECTED_CITATION_DATA_LOOKUP)
        self.pointwise_predictor.predict.assert_called_once_with(EXPECTED_CITATION_DATA_LOOKUP, CNN_RESULTS)
        self.listwise_predictor.predict.assert_called_once_with(EXPECTED_CITATION_DATA_LOOKUP, POINTWISE_AVG_RESULTS)
        input_data_lookup = { item["uid"]: item["data"] for item in PUBMED_XML_INPUT_DATA}
        self.results_formatter.format.assert_called_once_with(input_data_lookup, UNORDERED_LISTWISE_AVG_RESULTS)


@pytest.mark.unit
class TestIndexingPipeline(TestCase):

    def setUp(self):
        self.descriptorPredictionPipeline = MeshHeadingPredictionPipeline(None, None, None, None, None, None)
        self.descriptorPredictionPipeline.predict = MagicMock(return_value=EXPECTED_DESCRIPTOR_PREDICTIONS)
        self.subheading_predictor = SubheadingPredictor(None, None, None, None)
        self.subheading_predictor.predict = MagicMock(return_value=EXPECTED_DESCRIPTOR_PREDICTIONS_WITH_SUBHEADINGS)
        self.pipeline = IndexingPipeline(self.descriptorPredictionPipeline, self.subheading_predictor)

    def test_predict(self):        
        predictions = self.pipeline.predict(PUBMED_XML_INPUT_DATA)
    
        self.assertEqual(predictions, EXPECTED_DESCRIPTOR_PREDICTIONS_WITH_SUBHEADINGS, "Predictions do not match expected result.")
        self.descriptorPredictionPipeline.predict.assert_called_once_with(PUBMED_XML_INPUT_DATA)
        self.subheading_predictor.predict.assert_called_once_with(EXPECTED_DESCRIPTOR_PREDICTIONS)


@pytest.mark.unit
class TestMtiJsonResultsFormatter(TestCase):

    def setUp(self):
        self.formatter = MtiJsonResultsFormatter(DESC_NAME_LOOKUP, DUI_LOOKUP, THRESHOLD)

    def test_format(self):
        input_data_lookup = { item["uid"]: item["data"] for item in PUBMED_XML_INPUT_DATA}
        predictions = self.formatter.format(input_data_lookup, UNORDERED_LISTWISE_AVG_RESULTS)
        self.assertEqual(predictions, EXPECTED_DESCRIPTOR_PREDICTIONS, "Predictions are not as expected.")