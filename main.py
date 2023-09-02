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

model = whisper.load_model("base")
# basics in transcribing the audo
# I'd want to do more testing on load_model and what type to use
# Some of the transcriptions where off and I'd like to add more testing in place
# I'm returning tokens because that may be helpful to include in a vector database
# including the language as well to expand eventually
def transcribe_audio(audiofile) -> list:
    data = []
    audio = whisper.load_audio(audiofile)
    audio = whisper.pad_or_trim(audio)
    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    # behind the scenes uses a sliding 30 second window to transcribe
    result = model.transcribe(audiofile)
    data.append(result['text'])
    data.append(result['segments'][0]['tokens'])
    data.append(result['language'])
    return data

# this is the tokenise function that was told to leverage and use for the token step
def tokenise(audio_np_array: np.ndarray) -> torch.Tensor:
    """
    Function to tokenise an audio file represented as a NumPy array.

    Args:
    - audio_np_array (np.ndarray): The audio file as a NumPy array.

    Returns:
    - torch.Tensor: A random 1D tensor with dtype int16 and variable length in range (20, 1000).
    """

    # Check if the input is a NumPy array
    if not isinstance(audio_np_array, np.ndarray):
        raise ValueError("Input should be a NumPy array")

    # Time delay to simulate model inference
    time.sleep(0.15)

    tensor_length = np.random.randint(20, 1001)  # 1001 is exclusive
    return torch.randint(low=-32768, high=32767, size=(tensor_length,), dtype=torch.int16)

# sets up credentials to connect to the bucket
s3_client = boto3.client('s3',
                         aws_access_key_id='***',
                         aws_secret_access_key='***',
                         endpoint_url='***',
                         region_name='auto')
bucket_name = 'data-engineer-test'

# set up to page through all the files not just 1000
paginator = s3_client.get_paginator('list_objects')
page_iterator = paginator.paginate(Bucket=bucket_name)

all_objects = []

# Loop through each page of the results
for page in page_iterator:
    if 'Contents' in page:  # Check if the 'Contents' key exists
        all_objects.extend(page['Contents'])

# print the full length
list_of_audio_files = []
for obj in all_objects:
    print(type(obj))
    if 'Key' in obj:
        list_of_audio_files.append(obj['Key'])

print(len(list_of_audio_files))
print(list_of_audio_files[0])




# loop through all values here
# see if its an audio file - if not go to the next file
# 1 - download to local
# 2 - send to transcribe
# 3 - create audio_np_array, samplerate = sf.read(audio_file)
# 4 - tokenise with the sample code function and return the token
# 5 - create the ID from the path of the s3 location
# 6 - create an array with the results returned from both function and the id
# 7 - delete the local file for memory constraits

data = []
for audio in list_of_audio_files:
    print(audio)

    if 'flac' in audio:
        saved_file = audio.split("/")[-1]
        print(f'saving local file {saved_file}')
        s3_client.download_file(bucket_name, audio, saved_file)
        written = transcribe_audio(saved_file)
        print(f'transcribe: {written}')
        audio_np_array, samplerate = sf.read(saved_file)
        tokenised_tensor = tokenise(audio_np_array)
        # <id - relative path of audio file and adding the transcription number to include in the partition>, <transcription>, <token>

        # get the directory
        directory = "/".join(audio.split("/")[:-1])
        match = re.search(r'_(\d+)_', audio)
        # get the transcription number of the conversation to include in the partition
        if match:
            value = match.group(1)
            print(value)
        else:
            value = ''
        id = directory = "/".join(audio.split("/")[:-1]) + '/' + value

        data.append([id, written[0], written[1], written[2], tokenised_tensor])
        try:
            os.remove(saved_file)
            print(f"The file {saved_file} was deleted")
        except FileNotFoundError:
            print(f"The file {saved_file} was not found.")
        except PermissionError:
            print(f"Permission denied for deleting the file {saved_file}.")
        except Exception as e:
            print(f"An error occurred while deleting the file {saved_file}: {e}")

print(len(data))

df = pd.DataFrame(data, columns=['id', 'transcription','openai_token','language', 'token_sample'])
df['token_sample'] = df['token_sample'].apply(lambda x: x.tolist() if isinstance(x, torch.Tensor) else x)
df.to_parquet('data_science_example.parquet', index=False, partition_cols=['id'])

