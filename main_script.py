import boto3
import whisper
import os
import pandas as pd
from pydub import AudioSegment
import time
import torch
import numpy as np
import re
import soundfile as sf
from dotenv import load_dotenv
load_dotenv()

# Global constants
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY')
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_ENDPOINT = os.getenv('S3_ENDPOINT')
BUCKET_NAME = os.getenv('BUCKET_NAME')

PROCESSED_FILES = set()

# Set up S3 client
s3_client = boto3.client('s3',
                         aws_access_key_id=S3_ACCESS_KEY,
                         aws_secret_access_key=S3_SECRET_KEY,
                         endpoint_url=S3_ENDPOINT,
                         region_name='auto')


def transcribe_audio(audiofile) -> list:
    """Transcribe an audio file and return metadata."""
    data = []
    audio = whisper.load_audio(audiofile)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect spoken language and transcribe
    result = model.transcribe(audiofile)
    data.extend([result['text'], result['segments'][0]['tokens'], result['language']])
    return data


def tokenise(audio_np_array: np.ndarray) -> torch.Tensor:
    """Tokenize a given audio file represented as a NumPy array."""
    if not isinstance(audio_np_array, np.ndarray):
        raise ValueError("Input should be a NumPy array")

    time.sleep(0.15)
    tensor_length = np.random.randint(20, 1001)
    return torch.randint(low=-32768, high=32767, size=(tensor_length,), dtype=torch.int16)


def process_s3_object(audio) -> list:
    """Process a single S3 object."""
    if audio not in PROCESSED_FILES and 'flac' in audio:
        local_path = audio.split("/")[-1]
        s3_client.download_file(BUCKET_NAME, audio, local_path)

        transcription_data = transcribe_audio(local_path)
        audio_np_array, _ = sf.read(local_path)
        token_tensor = tokenise(audio_np_array)

        directory = "/".join(audio.split("/")[:-1])
        match = re.search(r'_(\d+)_', audio)
        id_val = directory + '/' + (match.group(1) if match else '')
        os.remove(local_path)
        print(f'finished for file {audio}')

        return [id_val, *transcription_data, token_tensor]
    return None


# Initialize model
model = whisper.load_model("base")

# Get a list of all objects in the bucket
paginator = s3_client.get_paginator('list_objects')
page_iterator = paginator.paginate(Bucket=BUCKET_NAME)

all_objects = [obj['Key'] for page in page_iterator if 'Contents' in page for obj in page['Contents']]
print(f"Found {len(all_objects)} objects in S3 bucket.")

# Process each S3 object
data = []
# all_objects[:13] <- this sets it to a sample set and quick run
for audio in all_objects:
    try:
        result = process_s3_object(audio)
        if result:
            data.append(result)
            PROCESSED_FILES.add(audio)
    except Exception as e:
        print(f"Error processing {audio}. Error: {e}")

# Convert to DataFrame and save to parquet
df = pd.DataFrame(data, columns=['id', 'transcription', 'openai_token', 'language', 'token_sample'])
df['token_sample'] = df['token_sample'].apply(lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
# df.to_parquet('data_science_example.parquet', index=False, partition_cols=['id'])

df.to_parquet('/app/data/data_science_example.parquet', index=False, partition_cols=['id'])


