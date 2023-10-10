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
3. Path to term types file. This file maps internal term ids to NLM term types (Descriptor, Check Tag, Publication Type, SCR).
4. Path to pointwise model passage lookup file. This file maps internal term ids to passages for the pointwise model.
5. Path to listwise model passage lookup file. This file maps internal term ids to passages for the listwise model.
6. Path to subheading name lookup file. This file maps NLM QUIs to subheading names.
7. Sagemaker endpoint name for CNN model.
8. Sagemaker endpoint name for Pointwise model.
9. Sagemaker endpoint name for Listwise model.
10. Sagemaker endpoint name for Subheading model.
11. The s3 bucket name (for async prediction temporary data).
12. The s3 prefix (for async prediction temporary data).
13. The cnn model batch size.
14. The pointwise model batch size.
15. The listwise model batch size.
16. The subheading model batch size.

Example usage for async endpoints. Set s3 bucket name (11.) or s3 prefix (12.) to None to use real-time endpoints.

```
from mtix import create_indexing_pipeline

pipeline = create_indexing_pipeline("path/to/mesh_heading_names_2023.tsv", 
                                    "path/to/mesh_headings_uis_2023.tsv",
                                    "path/to/mesh_headings_types_2023.tsv", 
                                    "path/to/mesh_heading_names_2023_w_types.tsv",
                                    "path/to/mesh_heading_names_2023_w_types_max_len_32.tsv",
                                    "path/to/subheading_names_2023_mesh.tsv",
                                    "raear-cnn-endpoint-2023-v2-async", 
                                    "raear-pointwise-endpoint-2023-v2-async", 
                                    "raear-listwise-endpoint-2023-v2-async",
                                    "raear-all-subheading-cnn-endpoint-2023-v1-async",
                                    "ncbi-aws-pmdm-ingest",
                                    "async_inference",
                                    cnn_batch_size=128,
                                    pointwise_batch_size=128,
                                    listwise_batch_size=128,
                                    subheading_batch_size=128)

predictions = pipeline.predict(input_data)
```