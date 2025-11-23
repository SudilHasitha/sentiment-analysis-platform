from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio

class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
        self.data = pd.read_csv(csv_path)
        self.video_dir = video_dir
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        # emotons mapping
        self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
        }
        self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
        }

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            # get idx from tensor to int
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            row = self.data.iloc[idx]

            filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
            video_path = os.path.join(self.video_dir, filename)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found at: {video_path}")
            # load text input
            text_input = self.tokenizer(row['Utterance'], 
                                        padding='max_length', 
                                        truncation=True, 
                                        max_length=128, 
                                        return_tensors='pt')
            
            # loading video frames
            video_frames = self._load_video_frames(video_path)
            audio_features = self._extract_audio_features(video_path)
            # print(audio_features)

            # map sentiment and emotion labrld
            emotion_label = self.emotion_map[row['Emotion'].lower()]
            sentiment_label = self.sentiment_map[row['Sentiment'].lower()]

            return {
                'text_input': {
                    'input_ids': text_input['input_ids'].squeeze(),
                    'attention_mask': text_input['attention_mask'].squeeze(),
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label),
            }
        # print(video_frames)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            return None
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # reading first frame
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Could not read first frame from video: {video_path}")
            
            # reset video to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # 30 fps video so get 1 second of frames
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                resized_frame = cv2.resize(frame, (224, 224))
                normalized_frame = resized_frame / 255.0
                frames.append(normalized_frame)
        except Exception as e:
            raise ValueError(f"Video error: {e}")
        finally:
            cap.release()

        # no frames found
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {video_path}")
        
        # pad (black) or truncate to 30 frames
        if len(frames) < 30:
            padding = [np.zeros_like(frames[0])] * (30 - len(frames))
            frames = frames + padding
        elif len(frames) > 30:
            frames = frames[:30]

        # Before permute [frames,height,width,channels]
        # After permute [frames,channels,height,width]
        return torch.FloatTensor(frames).permute(0, 3, 1, 2)

    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace('.mp4', '.wav')
        # use ffmpeg to extract audio features
        try:
            # get audio using ffmpeg
            subprocess.run([
                'ffmpeg', 
                '-i', video_path, 
                '-vn',
                '-acodec', 'pcm_s16le', 
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL)
            
            # load audio using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )  

            mel_spec = mel_spectrogram(waveform)

            # Normalize
            # Use 1e-5 for improved numerical stability when dividing by std, to reduce risk of division by nearly-zero.
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)   

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec
        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFmpeg error: {e}")
        except Exception as e:
            raise ValueError(f"Audio error: {e}")
        finally:
            # remove audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # collate function for dataloader to handle variable length 
    @staticmethod
    def collate_fn(batch):
        # filter out None items
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            raise ValueError("Batch is empty after filtering None items")
        return torch.utils.data.dataloader.default_collate(batch)

    @classmethod
    def prepare_dataloader(cls, train_csv, train_video_dir,
                           test_csv,test_video_dir,
                           dev_csv, dev_video_dir, batch_size=32):
        
        train_dataset = MELDDataset(csv_path=train_csv, video_dir=train_video_dir)
        test_dataset = MELDDataset(csv_path=test_csv, video_dir=test_video_dir)
        dev_dataset = MELDDataset(csv_path=dev_csv, video_dir=dev_video_dir)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=cls.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=cls.collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=cls.collate_fn)
        
        return train_loader, test_loader, dev_loader
        

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root (parent of training/)
    project_root = os.path.dirname(script_dir)
    # Construct absolute paths for dev, test and train datasets
    dev_csv_path = os.path.join(project_root, 'dataset', 'dev', 'dev_sent_emo.csv')
    dev_video_dir = os.path.join(project_root, 'dataset', 'dev', 'dev_splits_complete')
    test_csv_path = os.path.join(project_root, 'dataset', 'test', 'test_sent_emo.csv')
    test_video_dir = os.path.join(project_root, 'dataset', 'test', 'output_repeated_splits_test')
    train_csv_path = os.path.join(project_root, 'dataset', 'train', 'train_sent_emo.csv')
    train_video_dir = os.path.join(project_root, 'dataset', 'train', 'train_splits')
    
    # dataset = MELDDataset(csv_path=csv_path, video_dir=video_dir)
    # prepare dataloaders
    train_loader, test_loader, dev_loader = MELDDataset.prepare_dataloader(train_csv_path, train_video_dir,
                                                                       test_csv_path, test_video_dir,
                                                                       dev_csv_path, dev_video_dir, batch_size=32)

    print("Dataloaders prepared successfully")
    print("Training loader: ", len(train_loader))
    print("Test loader: ", len(test_loader))
    print("Dev loader: ", len(dev_loader))

    # print first batch one record
    for batch in train_loader:
        print(batch['text_input'])
        print(batch['video_frames'].shape)
        print(batch['audio_features'].shape)
        print(batch['emotion_label'])
        print(batch['sentiment_label'])
        break