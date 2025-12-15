# load the model
import os
import subprocess
import tempfile
import boto3
from torch._tensor import Tensor
from torch._tensor import Tensor
import cv2
import torch
import numpy as np
import torchaudio
import whisper
import json

from model import MultimodalSentimentModel
from transformers import AutoTokenizer 

emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
            }

sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2
            }

def install_ffmpeg():
    print("Installing ffmpeg...")
    subprocess.run(['python3', '-m', 'pip', 'install', '--upgrade','pip'], check=True)
    subprocess.run(['python3', '-m', 'pip', 'install', '--upgrade','setuptools'], check=True)
    try:
        subprocess.run(['python3', '-m', 'pip', 'install', 'ffmpeg'], check=True)
        print("FFmpeg installed successfully")
    except Exception as e:
        print(f"Error installing ffmpeg: {e} via pip")

    
    # install ffmpg via static build
    try:
        subprocess.run(['wget', 'https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz',
         '-O', '/tmp/ffmpeg-release-amd64-static.tar.xz'], check=True)
        subprocess.run(['tar', '-xvf', '/tmp/ffmpeg-release-amd64-static.tar.xz', '-C', '/tmp'], check=True)
        subprocess.run(['cp', '/tmp/ffmpeg-release-amd64-static/ffmpeg', '/usr/local/bin/ffmpeg'], check=True)
        # make ffmpeg executable
        subprocess.run(['chmod', '+x', '/usr/local/bin/ffmpeg'], check=True)
        # remove temporary files
        subprocess.run(['rm', '-rf', '/tmp/ffmpeg-release-amd64-static'], check=True)
        subprocess.run(['rm', '-rf', '/tmp/ffmpeg-release-amd64-static.tar.xz'], check=True)
        print("FFmpeg installed successfully via static build")
    except Exception as e:
        print(f"Error installing ffmpeg: {e} via static build")
        raise e 

    # verify ffmpeg is installed
    try:
        result = subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True, text=True)
        print(f"FFmpeg version: {result.stdout}")
    except Exception as e:
        print(f"Error verifying ffmpeg: {e}")
        raise e
    
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
    
def download_from_s3(s3_uri, local_dir="/tmp"):
    s3_client = boto3.client('s3')
    bucket = s3_uri.split('/')[2]
    key = '/'.join(s3_uri.split('/')[3:])

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        s3_client.download_file(bucket, key, temp_file.name)
        return temp_file.name
        
# function to handle user requests
def input_fn(request_body, request_content_type):

    if request_content_type != 'application/json':
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    input_data = json.loads(request_body)
    s3_uri = input_data.get('s3_uri', None)
    local_path = download_from_s3(s3_uri)
    return {'video_path': local_path}

def load_model(model_path):

    if not install_ffmpeg():
        raise RuntimeError(" FFmpeg failed to install ")

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

            video_frames = utterance_processor.video_processor.load_video_frames(segment_path)
            audio_features = utterance_processor.audio_processor.extract_audio_features(segment_path)
            text_inputs = tokenizer(
                segment['text'],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # move to device
            text_inputs = {k:v.to(device) for k,v in text_inputs.items()}
            # data struct to expected by the model -> in traing we have the bathc size so do the static model
            #  [batch_size, channels, height, width]
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            # get predictions 
            with torch.inference_mode():
                outputs = model.forward(text_inputs, video_frames, audio_features)
                emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
                sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

                emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
                sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

            # passing predictions
            predictions.append({
                "start_time":segment["start"],
                "end_time": segment["end"],
                "text":segment["text"],
                "emotions":[
                    {"label": emotion_map[idx.item()],
                    "confidence": conf.item()} \
                        for idx, conf in zip[tuple[Tensor, Tensor]](emotion_indices,emotion_values)
                ],
                "sentiments":[
                    {"label": sentiment_map[idx.item()],
                    "confidence": conf.item()} \
                        for idx, conf in zip[tuple[Tensor, Tensor]](sentiment_indices,sentiment_values)
                ]
            })

        except Exception as e:
            print(e)

        finally:
            # Cleanup
            if os.path.exists(segment_path):
                os.remove(segment_path)
    
    return {"utterances":predictions}



def process_local_video(video_path, model_dir="deployment/model/"):
    model_dict = load_model(model_dir)
    input_data = {'video_path':video_path}

    predictions = predict_fn(input_data, model_dict)

    for utterance in predictions["utterances"]:
        print("\n Utterance \n")
        print(f"start: {utterance['start_time']} s End: {utterance['end_time']} s \n")
        print(f"Text: {utterance["text"]} \n")
        print(f" Top Emotions and sentiments ")
        
        for emotion in utterance['emotions']:
            print(f"{emotion['label']}: {emotion['confidence']:.2f}")


        for sentiment in utterance['sentiments']:
            print(f"{sentiment['label']}: {sentiment['confidence']:.2f}")

        print("-"*50)

if __name__ == "__main__":
    model = load_model("deployment/model")
    # print(model)
    process_local_video("deployment/joy.mp4")
