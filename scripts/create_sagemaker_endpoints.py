from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.tensorflow import TensorFlowModel


INSTANCE_TYPE = "ml.g4dn.2xlarge"
S3_BUCKET = "s3://ncbi-aws-pmdm-ingest"
CNN_MODEL_FILENAME = "cnn_model_2023_v2.tar.gz"
POINTWISE_MODEL_FILENAME = "pointwise_model_2023_v2.tar.gz"
LISTWISE_MODEL_FILENAME = "listwise_model_2023_v2.tar.gz"
SUBHEADING_MODEL_FILENAME = "all_subheading_cnn_model_2023_v1.tar.gz"


def create_cnn_model_endpoint():
    model = TensorFlowModel(
        model_data=f"{S3_BUCKET}/mtix-models/{CNN_MODEL_FILENAME}",
        role="AmazonSageMaker-ExecutionRole-PMDM",
        framework_version="2.8.0",
        name="raear-cnn-model-2023-v2-async"
    )
    output_path = f"{S3_BUCKET}/async_inference/cnn_endpoint/outputs/"
    async_config = AsyncInferenceConfig(output_path=output_path)
    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name="raear-cnn-endpoint-2023-v2-async",
        async_inference_config=async_config,
    )


def create_pointwise_model_endpoint():
    huggingface_model = HuggingFaceModel(
        model_data=f"{S3_BUCKET}/mtix-models/{POINTWISE_MODEL_FILENAME}",
        role="AmazonSageMaker-ExecutionRole-PMDM",
        transformers_version="4.12.3",
        pytorch_version="1.9.1",
        py_version="py38",
        env={ "HF_TASK": "text-classification" },
        name="raear-pointwise-model-2023-v2-async"
    )
    output_path = f"{S3_BUCKET}/async_inference/pointwise_endpoint/outputs/"
    async_config = AsyncInferenceConfig(output_path=output_path)
    huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name="raear-pointwise-endpoint-2023-v2-async",
        async_inference_config=async_config
    )

def create_listwise_model_endpoint():
    huggingface_model = HuggingFaceModel(
        model_data=f"{S3_BUCKET}/mtix-models/{LISTWISE_MODEL_FILENAME}",
        role="AmazonSageMaker-ExecutionRole-PMDM",
        transformers_version="4.12.3",
        pytorch_version="1.9.1",
        py_version="py38",
        env={ "HF_TASK": "token-classification" },
        name="raear-listwise-model-2023-v2-async"
    )
    output_path = f"{S3_BUCKET}/async_inference/listwise_endpoint/outputs/"
    async_config = AsyncInferenceConfig(output_path=output_path)
    huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name="raear-listwise-endpoint-2023-v2-async",
        async_inference_config=async_config
    )

def create_subheading_endpoint():
    model = TensorFlowModel(
        model_data=f"{S3_BUCKET}/mtix-models/{SUBHEADING_MODEL_FILENAME}",
        role="AmazonSageMaker-ExecutionRole-PMDM",
        framework_version="2.8.0",
        name="raear-all-subheading-cnn-model-2023-v1-async"
    )
    output_path = f"{S3_BUCKET}/async_inference/subheading_endpoint/outputs/"
    async_config = AsyncInferenceConfig(output_path=output_path)
    model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name="raear-all-subheading-cnn-endpoint-2023-v1-async",
        async_inference_config=async_config
    )


def main():
    create_cnn_model_endpoint()
    create_pointwise_model_endpoint()
    create_listwise_model_endpoint()
    create_subheading_endpoint()


if __name__ == "__main__":
    main()