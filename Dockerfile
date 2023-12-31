# Use an official Python runtime as a base image
FROM python:3.9-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# Run your script when the container launches
CMD ["python", "main_script.py"]