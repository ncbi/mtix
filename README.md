# MTIX

## Installation

1. Create a Python 3.9 virtual environment. 
2. Install the package:

```
pip install .
```
3. Upload the CNN, Pointwise, Listwise, and Subheading model files to an AWS S3 bucket.
4. Deploy the SageMaker endpoints using the script provided in the git repository (./scripts/create_sagemaker_endpoints.py). Deployment settings (e.g. S3 bucket name) can be modified at the top of the script.
5. Create the following S3 directories for async prediction temporary files:<br>
s3://ncbi-aws-pmdm-ingest/async_inference/cnn_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/cnn_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/pointwise_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/pointwise_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/listwise_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/listwise_endpoint/outputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/subheading_endpoint/inputs/<br>
s3://ncbi-aws-pmdm-ingest/async_inference/subheading_endpoint/outputs/<br>

## Test
Automated unit and integration tests can be run using pytest. To run the integration test you may need to update the SageMaker endpoint names in the integration test code. The integration tests check predictions for 40k citations, and they may therefore take a long time to run.
```
pytest -m unit
pytest -m integration
```

## Usage

The pipeline is constructed with the following input parameters:

1. Path to the term name lookup file. This file maps internal term ids to term names.
2. Path to term unique identifier file. This file maps internal term ids to NLM UIs.
3. Path to subheading name lookup file. This file maps NLM QUIs to subheading names.
4. Sagemaker endpoint name for CNN model.
5. Sagemaker endpoint name for Pointwise model.
6. Sagemaker endpoint name for Listwise model.
7. Sagemaker endpoint name for Subheading model.
8. The s3 bucket name (for async prediction temporary data).
9. The s3 prefix (for async prediction temporary data).
10. The cnn model batch size.
11. The pointwise model batch size.
12. The listwise model batch size.
13. The subheading model batch size.

Example usage for async endpoints:

```
from mtix import create_async_pipeline

pipeline = create_async_pipeline("path/to/mesh_heading_names_2023.tsv", 
                                "path/to/mesh_headings_2023.tsv", 
                                "path/to/subheading_names_2023_mesh.tsv",
                                "raear-cnn-endpoint-2023-v1-async", 
                                "raear-pointwise-endpoint-2023-v1-async", 
                                "raear-listwise-endpoint-2023-v1-async",
                                "raear-all-subheading-cnn-endpoint-2023-v1-async",
                                "ncbi-aws-pmdm-ingest",
                                "async_inference",
                                cnn_batch_size=128,
                                pointwise_batch_size=128,
                                listwise_batch_size=128,
                                subheading_batch_size=128)

predictions = pipeline.predict(input_data)
```

Example usage for real-time endpoints:


```
from mtix import create_real_time_pipeline

pipeline = create_real_time_pipeline("path/to/mesh_heading_names_2023.tsv", 
                                    "path/to/mesh_headings_2023.tsv", 
                                    "path/to/subheading_names_2023_mesh.tsv",
                                    "raear-cnn-endpoint-2023-v1", 
                                    "raear-pointwise-endpoint-2023-v1", 
                                    "raear-listwise-endpoint-2023-v1",
                                    "raear-all-subheading-cnn-endpoint-2023-v1",
                                    cnn_batch_size=128,
                                    pointwise_batch_size=128,
                                    listwise_batch_size=128,
                                    subheading_batch_size=128)

predictions = pipeline.predict(input_data)
```