import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models
import os
from meld_dataset import MELDDataset

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # freeze bert model
        for param in self.bert.parameters():
            param.requires_grad = False

        self.projection = nn.Linear(768, 128)

    def forward(self, text_input):
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
        self.backbone = vision_models.video.r3d_18(pretrained=True)

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
    for i, prob in enumerate(emotion_probs):
        print(f"Emotion {emotion_map[i]}: {prob.item():.4f}")
    for i, prob in enumerate(sentiment_probs):
        print(f"Sentiment {sentiment_map[i]}: {prob.item():.4f}")

    