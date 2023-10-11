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
NAME_LOOKUP_PATH =                                        os.path.join(DATA_DIR, "mesh_heading_names_2023.tsv")
UI_LOOKUP_PATH =                                          os.path.join(DATA_DIR, "mesh_heading_uis_2023.tsv")
TYPE_LOOKUP_PATH =                                        os.path.join(DATA_DIR, "mesh_heading_types_2023.tsv")
POINTWISE_PASSAGE_LOOKUP_PATH =                           os.path.join(DATA_DIR, "mesh_heading_names_2023_w_types.tsv")
LISTWISE_PASSAGE_LOOKUP_PATH =                            os.path.join(DATA_DIR, "mesh_heading_names_2023_w_types_max_len_32.tsv")
SUBHEADING_NAME_LOOKUP_PATH =                             os.path.join(DATA_DIR, "subheading_names_2023_mesh.tsv")
TEST_SET_DATA_PATH =                                      os.path.join(DATA_DIR, "val_set_2017-2023_data.json.gz")
TEST_SET_MESH_HEADING_GROUND_TRUTH_PATH =                 os.path.join(DATA_DIR, "val_set_2017-2023_MeSH_Heading_Ground_Truth.json.gz")
TEST_SET_EXPECTED_CHAINED_SUBHEADING_PREDICTIONS_PATH =   os.path.join(DATA_DIR, "val_set_2017-2023_Chained_Subheading_Predictions.json.xz")
TEST_SET_EXPECTED_MESH_HEADING_PREDICTIONS_PATH =         os.path.join(DATA_DIR, "val_set_2017-2023_MeSH_Heading_Predictions.json.gz")
TEST_SET_SUBHEADING_GROUND_TRUTH_PATH =                   os.path.join(DATA_DIR, "val_set_2017-2023_Subheading_Ground_Truth.json.xz")


@pytest.mark.integration
class TestMeSHHeadingPredictionPipeline(TestCaseBase):

    def setUp(self):
        self.pipeline = create_mesh_heading_prediction_pipeline(NAME_LOOKUP_PATH, 
                                            UI_LOOKUP_PATH, 
                                            TYPE_LOOKUP_PATH,
                                            POINTWISE_PASSAGE_LOOKUP_PATH,
                                            LISTWISE_PASSAGE_LOOKUP_PATH,
                                            "raear-cnn-endpoint-2023-v2-async", 
                                            "raear-pointwise-endpoint-2023-v2-async", 
                                            "raear-listwise-endpoint-2023-v2-async",
                                            "ncbi-aws-pmdm-ingest",
                                            "async_inference",
                                            cnn_batch_size=128,
                                            pointwise_batch_size=128,
                                            listwise_batch_size=128,
                                            vpc_endpoint=VPC_ENDPOINT)
        self.test_set_data = json.load(gzip.open(TEST_SET_DATA_PATH, "rt", encoding="utf-8"))

    def test_output_for_n_articles(self):
        n = 1
        expected_predictions = json.load(gzip.open(TEST_SET_EXPECTED_MESH_HEADING_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:n]
        predictions = self._predict(n)
        self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def test_performance(self):
        delta = 0.001
        limit = 40000

        ground_truth = json.load(gzip.open(TEST_SET_MESH_HEADING_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))
        predictions = self._predict(limit)
        precision, recall, f1score = compute_metrics(ground_truth, predictions)
        
        self.assertAlmostEqual(f1score,   0.7151, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected.")
        self.assertAlmostEqual(precision, 0.7552, delta=delta, msg=f"Precision of {precision:.4f} not as expected.")
        self.assertAlmostEqual(recall,    0.6790, delta=delta, msg=f"Recall of {recall:.4f} not as expected.")
    
    def test_replace_brackets(self):
        # PMID 33998125 contains an abstract with the pattern "] [".
        self.pipeline.predict(ARTICLE_33998125)


@pytest.mark.integration
class TestIndexingPipeline(TestCaseBase):

    def setUp(self):
        self.pipeline = create_indexing_pipeline(NAME_LOOKUP_PATH, 
                                            UI_LOOKUP_PATH, 
                                            TYPE_LOOKUP_PATH,
                                            POINTWISE_PASSAGE_LOOKUP_PATH,
                                            LISTWISE_PASSAGE_LOOKUP_PATH,
                                            SUBHEADING_NAME_LOOKUP_PATH,
                                            "raear-cnn-endpoint-2023-v2-async", 
                                            "raear-pointwise-endpoint-2023-v2-async", 
                                            "raear-listwise-endpoint-2023-v2-async",
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

        precision, recall, f1score = compute_metrics(ground_truth, predictions, subheadings=True)
        self.assertAlmostEqual(f1score,   0.4767, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(precision, 0.4791, delta=delta, msg=f"Precision of {precision:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(recall,    0.4742, delta=delta, msg=f"Recall of {recall:.4f} not as expected for all subheadings.")

        precision, recall, f1score = compute_metrics(ground_truth, predictions, subheadings=True, subheading_filter=CRITICAL_SUBHEADINGS)
        self.assertAlmostEqual(f1score,   0.4993, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(precision, 0.5003, delta=delta, msg=f"Precision of {precision:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(recall,    0.4982, delta=delta, msg=f"Recall of {recall:.4f} not as expected for critical subheadings.")