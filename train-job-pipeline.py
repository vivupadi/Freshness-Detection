# job.py
"""
Submit training job to Azure ML
"""

from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential

import os
from dotenv import load_dotenv
load_dotenv()

# ============================================
# 1. Connect to workspace
# ============================================
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.getenv("SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("RG_NAME"),
    workspace_name=os.getenv("WS_NAME")
)

# ============================================
# 2. Define job
# ============================================
job = command(
    # Code
    code="./src/training",
    command="python train.py --data_dir ${{inputs.dataset}} --batch_size 32 --stage1_epochs 6 --stage2_epochs 6 --output_dir ${{outputs.model_output}}",
    
    # Input data (your registered dataset)
    inputs={
        "dataset": Input(
            type="uri_folder",
            path="azureml:freshnesscase:1"
        )
    },
    
    # Outputs (auto-created)
    outputs={
        "model_output": {
            "type": "uri_folder"
        }
    },
    
    # Environment (created in GUI)
    environment="freshness_env:1",
    
    # Compute (created in GUI)
    compute="cpu-cluster",  # or cpu-cluster
    
    # Experiment info
    experiment_name="fruit-freshness-training",
    display_name="ResNet50-Training"
)

# ============================================
# 3. Submit job
# ============================================
returned_job = ml_client.jobs.create_or_update(job)

print(f"✓ Job submitted: {returned_job.name}")
print(f"Studio URL: {returned_job.studio_url}")