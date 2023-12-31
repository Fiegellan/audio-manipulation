Of course! Here's a comprehensive README for your GitHub repository:

---

# Audio Processing with Docker

This repository contains a script for processing audio files stored in an S3 bucket. It transcribes the audio and saves the results as a parquet file.

## Prerequisites

- Docker installed on your machine.
- An S3 bucket containing your audio files.

## Quick Start

1. Clone this repository:
    ```bash
    git clone https://github.com/Fiegellan/audio-manipulation.git
    cd audio-manipulation
    ```

2. Build the Docker image:
    ```bash
    docker build -t audio-processing .
    ```

3. Run the Docker container, mounting the directory where you want to save the generated parquet files:
    ```bash
    docker run -v /path/on/your/machine:/app/data audio-processing
    ```

    **Note**: Replace `/path/on/your/machine` with the directory path on your host where you'd like to save the parquet files. The container will save the parquet files to `/app/data`, which will be mirrored to your specified host directory.

## Environment Configuration

You'll need to set some environment variables to access your S3 bucket. Place them in a `.env` file at the root of this project:

```
S3_ACCESS_KEY=your_access_key
S3_SECRET_KEY=your_secret_key
S3_ENDPOINT=your_endpoint_url
BUCKET_NAME=your_bucket_name
```

These will be loaded and used in the Python script.

## Under the Hood

### Dockerfile

We use a lightweight Python 3.9 base image. The required packages, such as `ffmpeg` and `libsndfile1`, are installed to support audio processing. The Python dependencies are listed in `requirements.txt` and are installed during the image build.

### Python Script

The script does the following:

1. Loads environment variables from `.env`.
2. Connects to the S3 bucket and retrieves a list of audio files.
3. Transcribes each audio file.
4. Processes and tokenizes the audio data.
5. Saves the results to a parquet file.

### Accessing Environment Variables in the Script

We use the `dotenv` library to manage environment variables. This allows us to easily load and use variables from the `.env` file:

```python
from dotenv import load_dotenv
load_dotenv()

S3_ACCESS_KEY = config('S3_ACCESS_KEY')
S3_SECRET_KEY = config('S3_SECRET_KEY')
S3_ENDPOINT = config('S3_ENDPOINT')
BUCKET_NAME = config('BUCKET_NAME')
```

### Saving Data

The processed data is saved as a parquet file inside the container at `/app/data`. Thanks to Docker's volume mounting, this data is also saved to the specified directory on your host machine.

# Thoughts and word dump


---

This is a 2.5 hour code challenge that includes the notebook on how I did ADA and started the analysis and then built it out.  
Main.py was my first go to
main_script.py is a bit cleaner and cleaned up

If I had more time I would work on paralizing the runs, batch download from s3, caching, and add logging vs print statements
I added more parquet partitioning information - i saw some audio may not be in english
The removal of the audio file also helps with memory control
Being in a container, means we can throw it at any compute engine we want and change cloud services if we need to

Really fun project, first time really working with audio files and being able to transcribe them!  Thank you for this opportunity and sorry it took longer to get to than planned originally.