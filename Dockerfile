# Use an official Python 3.7 image compatible with older Magenta
FROM python:3.7-slim

# Set working directory
WORKDIR /music_generator

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    ffmpeg \
    sox \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your local requirements for Magenta
# We'll install the versions compatible with Magenta 2.1.4
RUN pip install --upgrade pip

# Install Magenta and its dependencies
RUN pip install \
    tensorflow==2.9.1 \
    magenta==2.1.4 \
    note-seq==0.0.3 \
    librosa==0.7.2 \
    dm-sonnet==2.0.0 \
    matplotlib==3.5.2 \
    imageio==2.20.0 \
    mir-eval==0.7 \
    pygtrie==2.5.0 \
    python-rtmidi==1.1.2 \
    scikit-image==0.19.3 \
    sk-video==1.1.10 \
    sox==1.4.1 \
    mido==1.2.6 \
    absl-py==1.2.0 \
    numba==0.49.1 \
    numpy==1.21.6 \
    Pillow==9.2.0 \
    pretty-midi==0.2.9 \
    scipy==1.7.3 \
    six==1.16.0 \
    wheel==0.37.1 \
    tf_slim==1.1.0

# Set environment variable to avoid AVX errors if needed
ENV TF_CPP_MIN_LOG_LEVEL=2

# Mount the model and output directories when running the container
VOLUME ["/magenta_models", "/magenta_output"]

# Set default working directory inside container
WORKDIR /music_generator

# Default command: open a bash shell
CMD ["/bin/bash"]
