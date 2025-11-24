import os

# AWS SageMaker Training Job

SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
SM_CHANNEL_TRAIN = os.environ.get('SM_CHANNEL_TRAIN', 
'/opt/ml/input/data/train')