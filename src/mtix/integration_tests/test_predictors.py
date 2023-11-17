from .data import *
import gzip
import json
from mtix.sagemaker_factory import create_subheading_predictor
import os.path
import pytest
from .utils import compute_metrics, TestCaseBase
import xz


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(THIS_DIR, "data")
SUBHEADING_NAME_LOOKUP_PATH =                    os.path.join(DATA_DIR, "subheading_names_2023_mesh.tsv")
TEST_SET_MESH_HEADING_GROUND_TRUTH_PATH =        os.path.join(DATA_DIR, "val_set_2017-2023_MeSH_Heading_Ground_Truth.json.gz")
TEST_SET_EXPECTED_SUBHEADING_PREDICTIONS_PATH =  os.path.join(DATA_DIR, "val_set_2017-2023_Subheading_Predictions.json.xz")
TEST_SET_SUBHEADING_GROUND_TRUTH_PATH =          os.path.join(DATA_DIR, "val_set_2017-2023_Subheading_Ground_Truth.json.xz")


@pytest.mark.integration
class TestSubheadingPredictor(TestCaseBase):

    def setUp(self):
        self.pipeline = create_subheading_predictor(SUBHEADING_NAME_LOOKUP_PATH, 
                                                    "raear-all-subheading-cnn-endpoint-2023-v1-async", 
                                                    "ncbi-aws-pmdm-ingest",
                                                    "async_inference",
                                                     batch_size=128)
        self.test_set_data = json.load(gzip.open(TEST_SET_MESH_HEADING_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))

    # Expected predictions format needs to be updated (add reasons)
    # def test_output_for_n_articles(self):
    #     n = 100
    #     expected_predictions = json.load(xz.open(TEST_SET_EXPECTED_SUBHEADING_PREDICTIONS_PATH, "rt", encoding="utf-8"))[:n]
    #     predictions = self._predict(n)
    #     self.assertEqual(predictions, expected_predictions, "MTI JSON output not as expected.")

    def test_performance(self):
        delta = 0.001
        limit = 40000

        ground_truth = json.load(xz.open(TEST_SET_SUBHEADING_GROUND_TRUTH_PATH, "rt", encoding="utf-8"))
        predictions = self._predict(limit)
        
        precision, recall, f1score = compute_metrics(ground_truth, predictions, subheadings=True)
        self.assertAlmostEqual(f1score,   0.6561, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(precision, 0.6255, delta=delta, msg=f"Precision of {precision:.4f} not as expected for all subheadings.")
        self.assertAlmostEqual(recall,    0.6897, delta=delta, msg=f"Recall of {recall:.4f} not as expected for all subheadings.")

        precision, recall, f1score = compute_metrics(ground_truth, predictions, subheadings=True, subheading_filter=CRITICAL_SUBHEADINGS)
        self.assertAlmostEqual(f1score,   0.6697, delta=delta, msg=f"F1 score of {f1score:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(precision, 0.6372, delta=delta, msg=f"Precision of {precision:.4f} not as expected for critical subheadings.")
        self.assertAlmostEqual(recall,    0.7056, delta=delta, msg=f"Recall of {recall:.4f} not as expected for critical subheadings.")