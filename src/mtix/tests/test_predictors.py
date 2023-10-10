from .data import *
from mtix.predictors import CnnModelTop100Predictor, ListwiseModelTopNPredictor, PointwiseModelTopNPredictor, replace_brackets, SubheadingPredictor
from mtix.utils import CitationDataSanitizer, PubMedXmlInputDataParser
import pytest
import random
from unittest import TestCase
from unittest.mock import call, MagicMock, Mock


def round_top_results(top_results, ndigits):
    top_results = {q_id: {p_id: round(score, ndigits) for p_id, score in sorted(top_results[q_id].items(), key=lambda x: x[1], reverse=True) } for q_id in top_results}
    return top_results


def shuffle_top_results(top_results):
    top_results = {q_id: {p_id: score for p_id, score in random.sample(list(top_results[q_id].items()), k=len(top_results[q_id])) } for q_id in top_results}
    return top_results


@pytest.mark.unit
class TestCnnModelTop100Predictor(TestCase):

    def test_predict(self):
        tensorflow_endpoint = Mock()
        tensorflow_endpoint.predict = MagicMock(return_value=TENSORFLOW_ENDPOINT_RESULTS)
        cnn_predictor = CnnModelTop100Predictor(tensorflow_endpoint)
        top_results = cnn_predictor.predict(EXPECTED_CITATION_DATA_LOOKUP)
        top_results = round_top_results(top_results, 4)
        cnn_results = round_top_results(CNN_RESULTS, 4)
        self.assertEqual(top_results, cnn_results, "top results not as expected.")
        tensorflow_endpoint.predict.assert_called_once_with(TENSORFLOW_ENDPOINT_EXPECTED_INPUT_DATA)

    def test_replace_brackets(self):
        results = replace_brackets(REPLACE_BRACKETS_INPUT)
        self.assertEqual(results, REPLACE_BRACKETS_OUTPUT)


@pytest.mark.unit
class TestPointwiseModelTopNPredictor(TestCase):

    def test_predict(self):
        huggingface_endpoint = Mock()
        huggingface_endpoint.predict = MagicMock(return_value=HUGGINGFACE_PREDICTOR_POINTWISE_RESULTS)
        
        top_n = 5
        pointwise_predictor = PointwiseModelTopNPredictor(huggingface_endpoint, NAME_W_TYPES_LOOKUP, top_n)
        top_results = pointwise_predictor.predict(EXPECTED_CITATION_DATA_LOOKUP, CNN_RESULTS_SHUFFLED)

        top_results = round_top_results(top_results, 6)
        expected_top_results = round_top_results(EXPECTED_POINTWISE_TOP_5_RESULTS, 6)
        
        self.assertEqual(top_results, expected_top_results, "top results not as expected.")
        self.assertEqual(len(top_results["32770536"]), top_n, f"Expected {top_n} top results for each pmid.")
        self.assertEqual(len(top_results["30455223"]), top_n, f"Expected {top_n} top results for each pmid.")
        
        huggingface_endpoint.predict.assert_called_once_with(HUGGINGFACE_ENDPOINT_EXPECTED_POINTWISE_INPUT_DATA)


@pytest.mark.unit
class TestListwiseModelTopNPredictor(TestCase):

    def test_predict(self):
        huggingface_endpoint = Mock()
        huggingface_endpoint.predict = MagicMock(return_value=HUGGINGFACE_ENDPOINT_LISTWISE_RESULTS)
        
        top_n = 50
        listwise_predictor = ListwiseModelTopNPredictor(huggingface_endpoint, NAME_W_TYPES_LOOKUP, top_n)
        pointwise_avg_results_shuffled = shuffle_top_results(POINTWISE_AVG_RESULTS)
        top_results = listwise_predictor.predict(EXPECTED_CITATION_DATA_LOOKUP, pointwise_avg_results_shuffled)

        top_results = round_top_results(top_results, 4)
        expected_top_results =  {q_id: {p_id: LISTWISE_RESULTS[q_id][p_id] for p_id in top_results[q_id]} for q_id in top_results}
        expected_top_results = round_top_results(expected_top_results, 4)
        
        self.assertEqual(top_results, expected_top_results, "top results not as expected.")
        self.assertEqual(len(top_results["32770536"]), top_n, f"Expected {top_n} top results for each pmid.")
        self.assertEqual(len(top_results["30455223"]), top_n, f"Expected {top_n} top results for each pmid.")
        
        huggingface_endpoint.predict.assert_called_once_with(HUGGINGFACE_ENDPOINT_EXPECTED_LISTWISE_INPUT_DATA)


@pytest.mark.unit
class TestSubheadingPredictor(TestCase):

    def test_predict(self):
        input_parser = PubMedXmlInputDataParser()
        max_year = 2021
        data_sanitizer = CitationDataSanitizer(max_year)
        data_sanitizer.sanitize = Mock(wraps=data_sanitizer.sanitize)
        subheading_endpoint = Mock()
        subheading_endpoint.predict = MagicMock(return_value=SUBHEADING_ENDPOINT_RESULTS)
        subheading_predictor = SubheadingPredictor(input_parser, data_sanitizer, subheading_endpoint, SUBHEADING_NAME_LOOKUP)
        predictions = subheading_predictor.predict(EXPECTED_DESCRIPTOR_PREDICTIONS)
     
        self.assertEqual(predictions, EXPECTED_DESCRIPTOR_PREDICTIONS_WITH_SUBHEADINGS, "subheading predictions not as expected.")
        santizer_call_list = [call(citation_data) for citation_data in EXPECTED_CITATION_DATA_LOOKUP.values()]
        data_sanitizer.sanitize.assert_has_calls(santizer_call_list, any_order=False)
        subheading_endpoint.predict.assert_called_once_with(SUBHEADING_ENDPOINT_EXPECTED_INPUT_DATA)