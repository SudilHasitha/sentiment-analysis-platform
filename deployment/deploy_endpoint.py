from sagemaker.pytorch import PyTorch
import sagemaker
from sagemaker.pytorch.model import PyTorchModel

def deploy_endpoint():
    print("--test--")
    sagemaker.Session()
    role = "arn:aws:iam::067518243038:role/sentiment-analysis-deloy-endpoint-role"
    # s3://sentiment-analysis-mybucket/inference/
    model_uri = "s3://sentiment-analysis-mybucket/inference/model.tar.gz"
    model = PyTorchModel(
        model_data=model_uri,
        role=role,
        framework_version="2.5.1",
        py_version="py311",
        entry_point="inference.py",
        source_dir=".",
        name="sentiment-analysis-model"
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type="",
        endpoint_name="sentiment-analysis-endpoint",
    )

if __name__ == "__main__":
    deploy_endpoint()