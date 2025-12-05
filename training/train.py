# AWS SageMaker Training Job
import os
import argparse
import subprocess
import torchaudio
from meld_dataset import MELDDataset
from models import MultimodalSentimentModel, MultiModelTrainer
import torch
import tqdm
from install_ffmpeg import install_ffmpeg
import json

SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
SM_CHANNEL_TRAIN = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train')
SM_CHANNEL_VAL = os.environ.get('SM_CHANNEL_VAL', '/opt/ml/input/data/val')
SM_CHANNEL_TEST = os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test')

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=SM_MODEL_DIR)
    parser.add_argument('--train_dir', type=str, default=SM_CHANNEL_TRAIN)
    parser.add_argument('--val_dir', type=str, default=SM_CHANNEL_VAL)
    parser.add_argument('--test_dir', type=str, default=SM_CHANNEL_TEST)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    return parser.parse_args()

def main():
    # install ffmpeg in sagemaker if failed to install, exit with error
    try:
        install_ffmpeg()
    except Exception as e:
        print(f"Failed to install ffmpeg: {e}")
        raise e

    # print available backends
    print(f"Available backends: {torchaudio.get_audio_backend()}")
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(args)

    # track gpu memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_cached() / 1024**2:.2f} MB")
    else:
        print("No GPU available")
    
    train_loader, test_loader, val_loader = MELDDataset.prepare_dataloader(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'), 
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'), 
        val_csv=os.path.join(args.val_dir, 'val_sent_emo.csv'), 
        train_video_dir=os.path.join(args.train_dir, 'train_splits'), 
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'), 
        val_video_dir=os.path.join(args.val_dir, 'val_splits_complete'), 
        batch_size=args.batch_size)

    # print csv paths
    print(f"Training csv path: {os.path.join(args.train_dir, 'train_sent_emo.csv')}")
    print(f"Test csv path: {os.path.join(args.test_dir, 'test_sent_emo.csv')}")
    print(f"val csv path: {os.path.join(args.val_dir, 'val_sent_emo.csv')}")
    print(f"Training video path: {os.path.join(args.train_dir, 'train_splits')}")
    print(f"Test video path: {os.path.join(args.test_dir, 'output_repeated_splits_test')}")
    print(f"val video path: {os.path.join(args.val_dir, 'val_splits_complete')}")
    print(f"Batch size: {args.batch_size}")

    model = MultimodalSentimentModel().to(device)
    trainer = MultiModelTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')

    metrics_data = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
    }

    for epoch in tqdm.tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        metrics_data['train_loss'].append(train_loss)
        metrics_data['epochs'].append(epoch)

        val_loss, val_metrics = trainer.evaluate(val_loader, phase='val')
        metrics_data['val_loss'].append(val_loss)
        metrics_data['epochs'].append(epoch)

        # log metrics in sagemaker format
        print(json.dumps({
            "metrics":{
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name":"validation:emotion_precision", "Value": val_metrics["emotion_precision"]},
                {"Name":"validation:emotion_accuracy", "Value": val_metrics["emotion_accuracy"]},
                {"Name":"validation:sentiment_precision", "Value": val_metrics["sentiment_precision"]},
                {"Name":"validation:sentiment_accuracy", "Value": val_metrics["sentiment_accuracy"]},
            }
        }, indent=4))
        
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"GPU memory cached: {torch.cuda.memory_cached() / 1024**2:.2f} MB")

        # save best model
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
        
    # after training, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase='test')
    print(f"Test loss: {test_loss['total']}")
    print(f"Test metrics: {test_metrics}")

    metrics_data['test_loss'].append(test_loss['total'])
    print(json.dumps({
        "metrics":{
            {"Name": "test:loss", "Value": test_loss['total']},
            {"Name": "test:emotion_precision", "Value": test_metrics["emotion_precision"]},
            {"Name": "test:emotion_accuracy", "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_precision", "Value": test_metrics["sentiment_precision"]},
            {"Name": "test:sentiment_accuracy", "Value": test_metrics["sentiment_accuracy"]},
        }
    }, indent=4))
        

if __name__ == "__main__":
    main()