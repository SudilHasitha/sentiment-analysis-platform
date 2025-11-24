from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from models import MultimodalSentimentModel, MultiModelTrainer

def test_logging():
    Batch = namedtuple('Batch', 
    ['text_input', 'video_frames', 'audio_features'])

    mock_batch = Batch(
        text_input=torch.randint(0, 100, (1, 128)),
        video_frames=torch.randn(1, 3, 16, 224, 224),
        audio_features=torch.randn(1, 1, 64, 300),
    )

    mock_dataloader = DataLoader([mock_batch])
    model = MultimodalSentimentModel()
    trainer = MultiModelTrainer(model, mock_dataloader, mock_dataloader)

    train_losses = {
        'total': 2.4,
        'emotion': 1.2,
        'sentiment': 1.2,
    }

    trainer.log_metrics(train_losses, None, phase='train')

    val_losses = {
        'total': 2.8,
        'emotion': 1.4,
        'sentiment': 1.4,
    }
    val_metrics = {
        'emotion_precision': 0.85,
        'emotion_accuracy': 0.85,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.85,
    }
    trainer.log_metrics(val_losses, val_metrics, phase='val')

if __name__ == "__main__":
    test_logging()