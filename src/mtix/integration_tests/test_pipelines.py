from .data import *
import gzip
import json
from mtix import create_indexing_pipeline
from mtix.sagemaker_factory import create_mesh_heading_prediction_pipeline
import os.path
import pytest
from .utils import compute_metrics, TestCaseBase
import xz


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
VPC_ENDPOINT = None
DESC_NAME_LOOKUP_PATH =                                 os.path.join(DATA_DIR, "main_heading_names_2023_mesh.tsv")
DUI_LOOKUP_PATH =                                       os.path.join(DATA_DIR, "main_headings_2023_mesh.tsv")
SUBHEADING_NAME_LOOKUP_PATH =                           os.path.join(DATA_DIR, "subheading_names_2023_mesh.tsv")
TEST_SET_DATA_PATH =                                    os.path.join(DATA_DIR, "test_set_2017-2023_data.json.gz")
TEST_SET_DESCRIPTOR_GROUND_TRUTH_PATH =                 os.path.join(DATA_DIR, "test_set_2017-2023_Descriptor_Ground_Truth.json.gz")
TEST_SET_EXPECTED_CHAINED_SUBHEADING_PREDICTIONS_PATH = os.path.join(DATA_DIR, "test_set_2017-2023_Chained_Subheading_Predictions.json.xz")
TEST_SET_EXPECTED_DESCRIPTOR_PREDICTIONS_PATH =         os.path.join(DATA_DIR, "test_set_2017-2023_Descriptor_Predictions.json.gz")
TEST_SET_SUBHEADING_GROUND_TRUTH_PATH =                 os.path.join(DATA_DIR, "test_set_2017-2023_Subheading_Ground_Truth.json.xz")


@pytest.mark.integration
class TestDescriptorPredictionPipeline(TestCaseBase):

    def setUp(self):
        self.pipeline = create_mesh_heading_prediction_pipeline(DESC_NAME_LOOKUP_PATH, 
                                            DUI_LOOKUP_PATH, 
                                            "raear-cnn-endpoint-2023-v1-async", 
                                            "raear-pointwise-endpoint-2023-v1-async", 
                                            "raear-listwise-endpoint-2023-v1-async",
                                            "ncbi-aws-pmdm-ingest",
                                            "async_inference",
                                            cnn_batch_size=128,
                                            pointwise_batch_size=128,
                                            listwise_batch_size=128,
                                            vpc_endpoint=VPC_ENDPOINT)
        self.test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))

    def test_output_for_n_articles(self):
        n = 1
        expected_predictions = json.load(gzip.open(TEST_SET_EXPECTED_DESCRIPTOR_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:n]
        predictions = self._predict(n)
        self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def test_performance(self):
        delta = 0.001
        limit = 40000

        ground_truth = json.load(gzip.open(TEST_SET_DESCRIPTOR_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))
        predictions = self._predict(limit)
        precision, recall, f1score = compute_metrics(ground_truth, predictions)
        
        self.assertAlmostEqual(f1score,   0.7029, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected.")
        self.assertAlmostEqual(precision, 0.7330, delta=delta, msg=f"Precision of {precision:.4f} not as expected.")
        self.assertAlmostEqual(recall,    0.6753, delta=delta, msg=f"Recall of {recall:.4f} not as expected.")
    
    def test_replace_brackets(self):
        # PMID 33998125 contains an abstract with the pattern "] [".
        self.pipeline.predict(ARTICLE_33998125)


@pytest.mark.integration
class TestIndexingPipeline(TestCaseBase):

    def setUp(self):
        self.pipeline = create_indexing_pipeline(DESC_NAME_LOOKUP_PATH, 
                                            DUI_LOOKUP_PATH, 
                                            SUBHEADING_NAME_LOOKUP_PATH,
                                            "raear-cnn-endpoint-2023-v1-async", 
                                            "raear-pointwise-endpoint-2023-v1-async", 
                                            "raear-listwise-endpoint-2023-v1-async",
                                            "raear-all-subheading-cnn-endpoint-2023-v1-async",
                                            "ncbi-aws-pmdm-ingest",
                                            "async_inference",
                                            cnn_batch_size=128,
                                            pointwise_batch_size=128,
                                            listwise_batch_size=128,
                                            subheading_batch_size=128,
                                            vpc_endpoint=VPC_ENDPOINT)
        self.test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))

    def test_output_for_n_articles(self):
        n = 1
        expected_predictions = json.load(xz.open(TEST_SET_EXPECTED_CHAINED_SUBHEADING_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:n]
        predictions = self._predict(n)
        self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def test_performance(self):
        delta = 0.001
        limit = 40000

        ground_truth = json.load(xz.open(TEST_SET_SUBHEADING_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))
        predictions = self._predict(limit)

        precision, recall, f1score = compute_metrics(ground_truth, predictions)
        self.assertAlmostEqual(f1score,   0.4814, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(precision, 0.4768, delta=delta, msg=f"Precision of {precision:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(recall,    0.4862, delta=delta, msg=f"Recall of {recall:.4f} not as expected for all subheadings.")

        precision, recall, f1score = compute_metrics(ground_truth, predictions, CRITICAL_SUBHEADINGS)
        self.assertAlmostEqual(f1score,   0.5022, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(precision, 0.4955, delta=delta, msg=f"Precision of {precision:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(recall,    0.5090, delta=delta, msg=f"Recall of {recall:.4f} not as expected for critical subheadings.")