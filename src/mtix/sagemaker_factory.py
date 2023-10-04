import boto3
import sagemaker.session
from .endpoints import HuggingFaceAsyncEndpoint, HuggingFaceRealTimeEndpoint, TensorflowAsyncEndpoint, TensorflowRealTimeEndpoint
from .pipelines import MeshHeadingPredictionPipeline, IndexingPipeline, MtiJsonResultsFormatter
from .predictors import CnnModelTop100Predictor, ListwiseModelTopNPredictor, PointwiseModelTopNPredictor, SubheadingPredictor
from sagemaker.huggingface import HuggingFacePredictor
from sagemaker.predictor_async import AsyncPredictor
from sagemaker.tensorflow import TensorFlowPredictor
from .utils import CitationDataSanitizer, create_lookup, PubMedXmlInputDataParser


CONCURRENT_BATCHES = 100
MAX_YEAR = 2023
WAIT_DELAY = 1
WAIT_MAX_ATTEMPTS = 900


def create_async_pipeline(name_lookup_path, dui_lookup_path, subheading_name_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, subheading_endpoint_name, async_bucket_name, async_prefix, cnn_batch_size=128, pointwise_batch_size=128, listwise_batch_size=128, subheading_batch_size=128, vpc_endpoint = None):
    return create_indexing_pipeline(name_lookup_path, dui_lookup_path, subheading_name_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, subheading_endpoint_name, async_bucket_name, async_prefix, cnn_batch_size, pointwise_batch_size, listwise_batch_size, subheading_batch_size, vpc_endpoint=vpc_endpoint)


def create_real_time_pipeline(name_lookup_path, dui_lookup_path, subheading_name_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, subheading_endpoint_name, cnn_batch_size=128, pointwise_batch_size=128, listwise_batch_size=128, subheading_batch_size=128, vpc_endpoint = None):
    async_bucket_name = None
    async_prefix = None
    return create_indexing_pipeline(name_lookup_path, dui_lookup_path, subheading_name_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, subheading_endpoint_name, async_bucket_name, async_prefix, cnn_batch_size, pointwise_batch_size, listwise_batch_size, subheading_batch_size, vpc_endpoint=vpc_endpoint)


def create_mesh_heading_prediction_pipeline(name_lookup_path, dui_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, async_bucket_name, async_prefix, cnn_batch_size=128, pointwise_batch_size=128, listwise_batch_size=128, vpc_endpoint = None):
    is_async = (async_bucket_name is not None) and (async_prefix is not None)
    
    concurrent_batches = CONCURRENT_BATCHES
    max_year = MAX_YEAR
    wait_delay = WAIT_DELAY
    wait_max_attempts = WAIT_MAX_ATTEMPTS

    listwise_top_n = 40
    pointwise_top_n = 100
    threshold = 0.49

    input_data_parser = PubMedXmlInputDataParser()
    sanitizer = CitationDataSanitizer(max_year)

    sagemaker_session = create_sagemaker_session(vpc_endpoint)
    sagemaker_cnn_endpoint = TensorFlowPredictor(cnn_endpoint_name, sagemaker_session=sagemaker_session)
    if is_async:
        sagemaker_async_cnn_endpoint = AsyncPredictor(sagemaker_cnn_endpoint)
        async_cnn_endpoint = TensorflowAsyncEndpoint(sagemaker_async_cnn_endpoint, "cnn_endpoint", async_bucket_name, async_prefix, cnn_batch_size, wait_delay=wait_delay, wait_max_attempts=wait_max_attempts)
        cnn_endpoint = TensorflowRealTimeEndpoint(async_cnn_endpoint, batch_size=concurrent_batches*cnn_batch_size)
    else:
        cnn_endpoint = TensorflowRealTimeEndpoint(sagemaker_cnn_endpoint, batch_size=cnn_batch_size)
    cnn_model_top_100_predictor = CnnModelTop100Predictor(cnn_endpoint)

    name_lookup = create_lookup(name_lookup_path)
    sagemaker_pointwise_endpoint = HuggingFacePredictor(pointwise_endpoint_name, sagemaker_session=sagemaker_session)
    if is_async:
        sagemaker_async_pointwise_endpoint = AsyncPredictor(sagemaker_pointwise_endpoint)
        async_pointwise_endpoint = HuggingFaceAsyncEndpoint(sagemaker_async_pointwise_endpoint, "pointwise_endpoint", async_bucket_name, async_prefix, pointwise_batch_size, wait_delay=wait_delay, wait_max_attempts=wait_max_attempts)
        pointwise_endpoint = HuggingFaceRealTimeEndpoint(async_pointwise_endpoint, batch_size=concurrent_batches*pointwise_batch_size)
    else:
        pointwise_endpoint = HuggingFaceRealTimeEndpoint(sagemaker_pointwise_endpoint, batch_size=pointwise_batch_size)
    pointwise_model_top100_predictor = PointwiseModelTopNPredictor(pointwise_endpoint, name_lookup, pointwise_top_n)

    sagemaker_listwise_endpoint = HuggingFacePredictor(listwise_endpoint_name, sagemaker_session=sagemaker_session)
    if is_async:
        sagemaker_async_listwise_endpoint = AsyncPredictor(sagemaker_listwise_endpoint)
        async_listwise_endpoint = HuggingFaceAsyncEndpoint(sagemaker_async_listwise_endpoint, "listwise_endpoint", async_bucket_name, async_prefix, listwise_batch_size, wait_delay=wait_delay, wait_max_attempts=wait_max_attempts)
        listwise_endpoint = HuggingFaceRealTimeEndpoint(async_listwise_endpoint, batch_size=concurrent_batches*listwise_batch_size)
    else:
        listwise_endpoint = HuggingFaceRealTimeEndpoint(sagemaker_listwise_endpoint, batch_size=listwise_batch_size)
    listwise_model_topN_predictor = ListwiseModelTopNPredictor(listwise_endpoint, name_lookup, listwise_top_n)

    dui_lookup = create_lookup(dui_lookup_path)
    results_formatter = MtiJsonResultsFormatter(name_lookup, dui_lookup, threshold)

    pipeline = MeshHeadingPredictionPipeline(input_data_parser, sanitizer, cnn_model_top_100_predictor, pointwise_model_top100_predictor, listwise_model_topN_predictor, results_formatter)
    return pipeline


def create_indexing_pipeline(name_lookup_path, dui_lookup_path, subheading_name_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, subheading_endpoint_name, async_bucket_name, async_prefix, cnn_batch_size=128, pointwise_batch_size=128, listwise_batch_size=128, subheading_batch_size=128, vpc_endpoint = None):
    mesh_heading_prediction_pipeline = create_mesh_heading_prediction_pipeline(name_lookup_path, dui_lookup_path, cnn_endpoint_name, pointwise_endpoint_name, listwise_endpoint_name, async_bucket_name, async_prefix, cnn_batch_size, pointwise_batch_size, listwise_batch_size, vpc_endpoint=vpc_endpoint)
    subheading_predictor = create_subheading_predictor(subheading_name_lookup_path, subheading_endpoint_name, async_bucket_name, async_prefix, subheading_batch_size, vpc_endpoint=vpc_endpoint)
    indexing_pipeline = IndexingPipeline(mesh_heading_prediction_pipeline, subheading_predictor)
    return indexing_pipeline


def create_subheading_predictor(subheading_name_lookup_path, subheading_endpoint_name, async_bucket_name, async_prefix, batch_size=128, vpc_endpoint = None):
    is_async = (async_bucket_name is not None) and (async_prefix is not None)

    concurrent_batches = CONCURRENT_BATCHES
    max_year = MAX_YEAR
    wait_delay = WAIT_DELAY
    wait_max_attempts = WAIT_MAX_ATTEMPTS

    input_data_parser = PubMedXmlInputDataParser()
    sanitizer = CitationDataSanitizer(max_year)

    sagemaker_session = create_sagemaker_session(vpc_endpoint)
    sagemaker_subheading_endpoint = TensorFlowPredictor(subheading_endpoint_name, sagemaker_session=sagemaker_session)
    if is_async:
        sagemaker_async_subheading_endpoint = AsyncPredictor(sagemaker_subheading_endpoint)
        async_subheading_endpoint = TensorflowAsyncEndpoint(sagemaker_async_subheading_endpoint, "subheading_endpoint", async_bucket_name, async_prefix, batch_size, wait_delay=wait_delay, wait_max_attempts=wait_max_attempts)
        subheading_endpoint = TensorflowRealTimeEndpoint(async_subheading_endpoint, batch_size=concurrent_batches*batch_size)
    else:
        subheading_endpoint = TensorflowRealTimeEndpoint(sagemaker_subheading_endpoint, batch_size=batch_size)
    
    subheading_name_lookup = create_lookup(subheading_name_lookup_path)
    subheading_predictor = SubheadingPredictor(input_data_parser, sanitizer, subheading_endpoint, subheading_name_lookup)

    return subheading_predictor

def create_sagemaker_session(vpc_endpoint):
    boto_session = boto3.Session()
    sagemaker_runtime_client = boto_session.client("sagemaker-runtime", endpoint_url=vpc_endpoint)
    return sagemaker.session.Session(sagemaker_runtime_client=sagemaker_runtime_client)
