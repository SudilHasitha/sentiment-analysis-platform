# load the model
import os
import subprocess
import cv2
import torch
import numpy as np
import torchaudio
import whisper

from model import MultimodalSentimentModel
from transformers import AutoTokenizer 


class VideoProcessor:
     def load_video_frames(self, video_path):
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
        # Convert list to numpy array first to avoid slow tensor creation warning
        frames_array = np.array(frames)
        return torch.FloatTensor(frames_array).permute(0, 3, 1, 2)

class AudioProcessor:
    def extract_audio_features(self, video_path):
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


class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segments(self, video_path, start_time, end_time, temp_dir="/tmp"):
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(temp_dir, f"segment_{start_time}_{end_time}.mp4")

        subprocess.run([
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-y',
            str(segment_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
            raise ValueError(f"Failed to extract segment: {segment_path}")

        return segment_path

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)
    mode_path = os.path.join(model_path, "model.pth")

    if not os.path.exists(mode_path):
        raise FileNotFoundError(f"Model file not found at: {mode_path}")

    model.load_state_dict(torch.load(mode_path, map_location=device))
    model.eval()
    return {
        'model': model,
        'tokenizer': AutoTokenizer.from_pretrained('bert-base-uncased'),
        'transcriber': whisper.load_model(
            'base',
            device='cpu' if not torch.cuda.is_available() else 'cuda',
        ),
        'device': device,
    }

def predict_fn(input_data, model_dict):
    model = model_dict['model']
    tokenizer = model_dict['tokenizer']
    device = model_dict['device']
    video_path = input_data['video_path']

    # get the transcription of the video
    result = model_dict['transcriber'].transcribe(
        video_path,
        word_timestamps=True
    )

    utterance_processor = VideoUtteranceProcessor()
    predictions = []

    for segment in result['segments']:
        try:
            segment_path = utterance_processor.extract_segments(
                video_path,
                segment['start'],
                segment['end']
            )

            # video_frames = utterance_processor.load_video_frames
        except Exception as e:
            print(e)



def process_local_video(video_path, model_dir="model"):
    model_dict = load_model(model_dir)
    input_data = {'video_path':video_path}

    predictions = predict_fn(input_data, model_dict)

if __name__ == "__main__":
    model = load_model("deployment/model")
    print(model)
    process_local_video("./joy.mp4")
