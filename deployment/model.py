# inferencing with encoder and feed forwarding for predictions

import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models as vision_models

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

# --- AudioEncoder ---
class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
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
        # Using self.conv_layers here
        for param in self.conv_layers.parameters():
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
        features = self.conv_layers(x) # Using self.conv_layers here
        return self.projection(features.squeeze(-1))

# --- MultimodalSentimentModel ---
class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()

        # encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()

        # fusion layers
        self.fusion_layer = nn.Sequential(
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
        fusion_features = self.fusion_layer(combined_features) # Using self.fusion_layer here

        # classify emotion and sentiment
        # emotion output: [batch_size, 7]
        # sentiment output: [batch_size, 3]
        emotion_logits = self.emotion_classifier(fusion_features)
        sentiment_logits = self.sentiment_classifier(fusion_features)
        
        return {'emotion': emotion_logits, 'sentiment': sentiment_logits}