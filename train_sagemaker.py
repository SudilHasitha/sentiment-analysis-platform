from sagemaker.train import ModelTrainer
from sagemaker.core.inputs import TrainingInput # This line works for inputs

def start_training():
    # 1. Define Inputs (TrainingInput import is confirmed to be working)
    inputs = {
        "training": TrainingInput(
            s3_data="s3://your-bucket-name/dataset/train", 
            distribution="FullyReplicated", 
            s3_data_type="S3Prefix"
        ),
        "validation": TrainingInput(
            s3_data="s3://your-bucket-name/dataset/dev", 
            distribution="FullyReplicated", 
            s3_data_type="S3Prefix"
        ),
        "test": TrainingInput(
            s3_data="s3://your-bucket-name/dataset/test", 
            distribution="FullyReplicated", 
            s3_data_type="S3Prefix"
        ),
    }

    # 2. Define the Low-Level Boto3 Resource Configuration
    # This replaces the need for ComputeConfig, StoppingCondition, and OutputDataConfig classes.
    resource_config = {
        "InstanceType": "ml.g5.xlarge",
        "InstanceCount": 1,
        "VolumeSizeInGB": 30
    }

    stopping_condition = {
        "MaxRuntimeInSeconds": 86400
    }

    output_config = {
        "S3OutputPath": "s3://your-bucket-name/model-output/"
    }

    # 3. Create the ModelTrainer
    # Pass the dictionaries directly to the ModelTrainer constructor.
    trainer = ModelTrainer(
        training_image="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker",
        role="your-execution-role",
        base_job_name="video-sentiment-pytorch-job",

        # FIX: Pass the resource dictionaries directly
        resource_config=resource_config,
        stopping_condition=stopping_condition,
        output_data_config=output_config, # Note: This parameter name is required
        
        hyperparameters={"batch-size": 32, "epochs": 25},
        environment={
            "SAGEMAKER_PROGRAM": "train.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "s3://your-bucket-name/code/video-sentiment-training.tar.gz",
        }
    )

    # 4. Submit the training job
    trainer.train(
        inputs=inputs, 
        wait=True
    )

if __name__ == "__main__":
    start_training()