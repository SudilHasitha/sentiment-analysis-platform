from torch._tensor import Tensor
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import os
try:
    from .meld_dataset import MELDDataset  # when imported as package
except ImportError:
    from meld_dataset import MELDDataset  # fallback when running as script
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # freeze bert model
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, input_ids, attention_mask):
        # extract BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # use CLS token represntation
        pooler_output = outputs.pooler_output

        # project to 128 dimensions
        return self.projection(pooler_output)

# add video encoder
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(weights='DEFAULT')

        # freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # swith the tensor to align with backborn model
        # [batch_size, frames, channels, height, width] -> [batch_size, channels, frames, height, width]
        x = x.transpose(1, 2)
        return self.backbone(x)

# add audio encoder
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # pattern detection layers
        self.con_layers = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64,128,kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # freeze pattern detection layers
        for param in self.con_layers.parameters():
            param.requires_grad = False

        # trainable projection layer
        self.projection = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        # remove channel dimension from mel spectrogram
        # example: [batch_size,1, 64, 300] -> [batch_size,64,300]
        # [batch_size, channels, frames, time_steps] -> [batch_size, frames, time_steps]
        x = x.squeeze(1)
        # features output : [batch_size, 128, 1] # 1 is from -> AdaptiveAvgPool1D(1)
        features = self.con_layers(x)
        return self.projection(features.squeeze(-1))

# Sentiment analysis model
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(128*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
        ) 

        # emotion classification layers
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),
        )

        # sentiment classification layers
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, text_input, video_frames, audio_features):
        # encode text, video, and audio
        text_features = self.text_encoder(
            text_input['input_ids'],
            text_input['attention_mask']
        )
        video_features = self.video_encoder(video_frames)
        audio_features = self.audio_encoder(audio_features)
        
        # concat features
        combined_features = torch.cat([
            text_features, 
            video_features, 
            audio_features], 
        dim=1)

        # fuse features
        fusion_features = self.fusion_layers(combined_features)

        # classify emotion and sentiment
        # emotion output: [batch_size, 7]
        # sentiment output: [batch_size, 3]
        emotion_logits = self.emotion_classifier(fusion_features)
        sentiment_logits = self.sentiment_classifier(fusion_features)
        
        return {'emotion': emotion_logits, 'sentiment': sentiment_logits}

class MultiModelTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # print dataset sizes
        print(f"Training dataset size: {len(train_loader):,}")
        print(f"Validation dataset size: {len(val_loader):,}")
        print(f"Batch per epoch: {len(train_loader):,}")

        # tensorboard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = '/opt/ml/output/tensorboard' if 'SM_MODEL_DIR' in os.environ else 'runs'
        log_dir = os.path.join(base_dir, f'multimodal_sentiment_model_run_{timestamp}')
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        # optimizer and loss function
        self.optimizer = torch.optim.Adam([
            {'params': self.model.text_encoder.parameters(), 'lr': 8e-6},
            {'params': self.model.video_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.audio_encoder.parameters(), 'lr': 8e-5},
            {'params': self.model.fusion_layers.parameters(), 'lr': 5e-4},
            {'params': self.model.emotion_classifier.parameters(), 'lr': 5e-4},
            {'params': self.model.sentiment_classifier.parameters(), 'lr': 5e-4},
        ], weight_decay=1e-5)

        self.current_train_losses = None

        # calculate class weights
        emotion_weights, sentiment_weights = self.compute_class_weights(self.train_loader)

        # move weights to device
        device = next(self.model.parameters()).device

        # emotion loss
        self.emotion_criterion = nn.CrossEntropyLoss(
            weight=emotion_weights.to(device),
            label_smoothing=0.05,
        )

        # sentiment loss
        self.sentiment_criterion = nn.CrossEntropyLoss(
            weight=sentiment_weights.to(device),
            label_smoothing=0.05,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1, 
            patience=2
        )

        # to avoid overfitting
        self.emotion_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
        )

        # to avoid overfitting
        self.sentiment_criterion = nn.CrossEntropyLoss(
            label_smoothing=0.05,
        )

    # there is an over representation of neutral emotion, so we need to balance the dataset
    #  additon of weights for each class
    def compute_class_weights(self, dataset):

        emotion_counts = torch.tensor.zeros(7)
        sentiment_counts = torch.tensor.zeros(3)
        skipped = 0
        total = len(dataset)

        print("Counting class frequencies...")
        for i in range(total):
            sample = dataset[i]

            if sample is None:
                skipped += 1
                continue

            emotion_label = sample['emotion_label']
            sentiment_label = sample['sentiment_label']
            emotion_counts[emotion_label] += 1
            sentiment_counts[sentiment_label] += 1
        
        print(f"Skipped {skipped} samples due to None values")
        print(f"Total samples: {total}")
        print(f"Emotion counts: {emotion_counts}")

        valid_counts = total - skipped
        print("Skipped samples: ",(skipped/total)*100, "%")
        print("Valid samples: ",(valid_counts/total)*100, "%")

        # class distribution
        emotions_map = {
            0: 'anger',
            1: 'disgust',
            2: 'fear',
            3: 'joy',
            4: 'neutral',
            5: 'sadness',
            6: 'surprise',
        }
        for i, count in enumerate[Tensor](emotion_counts):
            print(f"Emotion {emotions_map[i]}: {count.item():.2f}")
        print("-"*50)
        sentiment_map = {
            0: 'negative',
            1: 'neutral',
            2: 'positive',
        }
        for i, count in enumerate[Tensor](sentiment_counts):
            print(f"Sentiment {sentiment_map[i]}: {count.item():.2f}")
        print("-"*50)

        #  calculate class weights
        emotion_weights = 1.0 / (emotion_counts + 1e-5)
        sentiment_weights = 1.0 / (sentiment_counts + 1e-5)

        # normalize weights
        emotion_weights = emotion_weights / emotion_weights.sum()
        sentiment_weights = sentiment_weights / sentiment_weights.sum()

        return emotion_weights, sentiment_weights

    def log_metrics(self,loss, metrics, phase="train"):
        if phase == "train":
            self.current_train_losses = loss
        else:
            self.writer.add_scalar('loss/total/train', 
                self.current_train_losses['total'],
                self.global_step
            )
            self.writer.add_scalar('loss/total/val', 
                loss['total'],
                self.global_step
            )
            self.writer.add_scalar('loss/emotion/train', 
                self.current_train_losses['emotion'],
                self.global_step
            )
            self.writer.add_scalar('loss/emotion/val', 
                loss['emotion'],
                self.global_step
            )
            self.writer.add_scalar('loss/sentiment/train', 
                self.current_train_losses['sentiment'],
                self.global_step
            )
            self.writer.add_scalar('loss/sentiment/val', 
                loss['sentiment'],
                self.global_step
            )

        if metrics:
            self.writer.add_scalar(
                f'{phase}/emotion_precision',
                metrics['emotion_precision'],
                self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/emotion_accuracy',
                metrics['emotion_accuracy'],
                self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_precision',
                metrics['sentiment_precision'],
                self.global_step
            )
            self.writer.add_scalar(
                f'{phase}/sentiment_accuracy',
                metrics['sentiment_accuracy'],
                self.global_step
            )
    
    def train_epoch(self):
        self.model.train()
        
        running_loss = {
            'total': 0.0,
            'emotion': 0.0,
            'sentiment': 0.0,
        }

        for batch in self.train_loader:
            # getting tensors to same device
            device = next(self.model.parameters()).device
            text_input = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device),
            }
            video_frames = batch['video_frames'].to(device)
            audio_features = batch['audio_features'].to(device)
            emotion_labels = batch['emotion_label'].to(device)
            sentiment_labels = batch['sentiment_label'].to(device)
            
            # zero gradients
            self.optimizer.zero_grad()
            
            # forward pass
            outputs = self.model(text_input, video_frames, audio_features)

            # calculate loss
            emotion_logits = outputs['emotion']
            sentiment_logits = outputs['sentiment']
            emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)
            sentiment_loss = self.sentiment_criterion(sentiment_logits, sentiment_labels)
            loss = emotion_loss + sentiment_loss

            # backward pass
            loss.backward()

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # update optimizer
            self.optimizer.step()

            # update running loss
            running_loss['total'] += loss.item()
            running_loss['emotion'] += emotion_loss.item()
            running_loss['sentiment'] += sentiment_loss.item()

            # log metrics
            self.log_metrics(
                {
                    'total': loss.item(),
                    'emotion': emotion_loss.item(),
                    'sentiment': sentiment_loss.item(),
                }
            )

        return {
            k: v / len(self.train_loader) for k, v in running_loss.items()
        }

    def evaluate(self, data_loader, phase='val'):
        self.model.eval()

        # running loss
        running_loss = {
            'total': 0.0,
            'emotion': 0.0,
            'sentiment': 0.0,
        }

        # all predictions and labels
        all_emotion_preds = []
        all_sentiment_preds = []
        all_emotion_labels = []
        all_sentiment_labels = []
        
        with torch.inference_mode():
            for batch in data_loader:
                device = next(self.model.parameters()).device
                text_input = {
                    'input_ids': batch['text_inputs']['input_ids'].to(device),
                    'attention_mask': batch['text_inputs']['attention_mask'].to(device),
                }
                video_frames = batch['video_frames'].to(device)
                audio_features = batch['audio_features'].to(device)
                emotion_labels = batch['emotion_label'].to(device)
                sentiment_labels = batch['sentiment_label'].to(device)
                
                # forward pass
                outputs = self.model(text_input, video_frames, audio_features)

                # calculate loss
                emotion_logits = outputs['emotion']
                sentiment_logits = outputs['sentiment']
                emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)
                sentiment_loss = self.sentiment_criterion(sentiment_logits, sentiment_labels)
                loss = emotion_loss + sentiment_loss

                # update running loss
                running_loss['total'] += loss.item()
                running_loss['emotion'] += emotion_loss.item()
                running_loss['sentiment'] += sentiment_loss.item()

                # collect predictions
                all_emotion_preds.extend(
                    torch.argmax(emotion_logits, dim=1).cpu().numpy()
                )
                all_sentiment_preds.extend(
                    torch.argmax(sentiment_logits, dim=1).cpu().numpy()
                )
                all_emotion_labels.extend(emotion_labels.cpu().numpy())
                all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

                # update the global step
                self.global_step += 1

        # calculate average loss
        average_loss = {
            k: v / len(data_loader) for k, v in running_loss.items()
        }

        # calculate accuracy, precision
        emotion_precision = precision_score(
            all_emotion_labels, all_emotion_preds, average='weighted')
        sentiment_precision = precision_score(
            all_sentiment_labels, all_sentiment_preds, average='weighted')
        emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
        sentiment_accuracy = accuracy_score(all_sentiment_labels, all_sentiment_preds)

        # log metrics
        self.log_metrics(
            average_loss,
            {
                'total': average_loss['total'],
                'emotion': average_loss['emotion'],
                'sentiment': average_loss['sentiment'],
            },
            phase=phase
        )

        # update scheduler
        if phase == 'val':
            self.scheduler.step(average_loss['total'])
        
        # return average loss and metrics
        metrics = {
            'emotion_precision': emotion_precision,
            'emotion_accuracy': emotion_accuracy,
            'sentiment_precision': sentiment_precision,
            'sentiment_accuracy': sentiment_accuracy,
        }
        return average_loss, metrics

    
if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root (parent of training/)
    project_root = os.path.dirname(script_dir)
    # Construct absolute paths
    train_csv_path = os.path.join(project_root, 'dataset', 'train', 'train_sent_emo.csv')
    train_video_dir = os.path.join(project_root, 'dataset', 'train', 'train_splits')
    
    dataset = MELDDataset(csv_path=train_csv_path, video_dir=train_video_dir)
    sample = dataset[0]

    model = MultimodalSentimentModel()
    model.eval()
    
    text_input = {
        'input_ids': sample['text_input']['input_ids'].unsqueeze(0),
        'attention_mask': sample['text_input']['attention_mask'].unsqueeze(0),
    }
    video_frames = sample['video_frames'].unsqueeze(0)
    audio_features = sample['audio_features'].unsqueeze(0)

    with torch.inference_mode():
        outputs = model(text_input, video_frames, audio_features)

        emotion_probs = torch.softmax(outputs['emotion'], dim=1)[0]
        sentiment_probs = torch.softmax(outputs['sentiment'], dim=1)[0]

    emotion_map = {
    0 : 'anger', 1 : 'disgust', 2 : 'fear', 3 : 'joy', 4 : 'neutral', 5 : 'sadness', 6 : 'surprise'
    }
    sentiment_map = {
    0 : 'negative', 1 : 'neutral', 2 : 'positive'
    }

    # map probabilities to emotions and sentiments
    for i, prob in enumerate[Tensor](emotion_probs):
        print(f"Emotion {emotion_map[i]}: {prob.item():.4f}")
    for i, prob in enumerate[Tensor](sentiment_probs):
        print(f"Sentiment {sentiment_map[i]}: {prob.item():.4f}")

    