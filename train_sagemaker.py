from sagemaker.estimator import Estimator 
from sagemaker.inputs import TrainingInput
from sagemaker.debugger import TensorBoardOutputConfig
import boto3
from sagemaker.session import Session 

def start_training():
    aws_region = 'us-east-1' 
    boto_session = boto3.Session(region_name=aws_region)
    sagemaker_session = Session(boto_session=boto_session)
    
    inputs = {
        "training": TrainingInput(
            s3_data="s3://sentiment-analysis-mybucket/dataset/train", 
            distribution="FullyReplicated", 
            s3_data_type="S3Prefix"
        ),
        "validation": TrainingInput(
            s3_data="s3://sentiment-analysis-mybucket/dataset/val", 
            distribution="FullyReplicated", 
            s3_data_type="S3Prefix"
        ),
        "test": TrainingInput(
            s3_data="s3://sentiment-analysis-mybucket/dataset/test", 
            distribution="FullyReplicated", 
            s3_data_type="S3Prefix"
        ),
    }

    tb_config = TensorBoardOutputConfig(
        s3_output_path="s3://sentiment-analysis-mybucket/tensorboard",
        container_local_output_path="/opt/ml/output/tensorboard" # Optional, but good practice
    )

    estimator = Estimator(
        image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker",
        role="arn:aws:iam::067518243038:role/sentiment-analysis-execution-role",
        base_job_name="video-sentiment-pytorch-job",
        
        instance_type="ml.g5.xlarge",
        instance_count=1,
        volume_size=30,
        
        output_path="s3://sentiment-analysis-mybucket/model-output/",
        max_run=86400,
        
        tensorboard_output_config=tb_config,
        
        hyperparameters={"batch-size": 32, "epochs": 25},
        
        entry_point="train.py",
        source_dir="s3://sentiment-analysis-mybucket/code/video-sentiment-training.tar.gz",
        
        sagemaker_session=sagemaker_session
    )

    estimator.fit(
        inputs=inputs, 
        wait=True
    )

if __name__ == "__main__":
    start_training()