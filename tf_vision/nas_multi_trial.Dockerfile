# This Dockerfile runs the TF-Vision cloud-search-main file.

FROM tensorflow/tensorflow:2.7.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

# This is added to fix docker build error related to Nvidia key update.
RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install basic libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        curl \
        wget \
        sudo \
        gnupg \
        libsm6 \
        libxext6 \
        libxrender-dev \
        lsb-release \
        ca-certificates \
        build-essential \
        git


# Install google cloud SDK
RUN wget -q https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-359.0.0-linux-x86_64.tar.gz
RUN tar xzf google-cloud-sdk-359.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh -q
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


# Install required libs
RUN pip install --upgrade pip
RUN pip install cloud-tpu-client==0.10
RUN pip install pyyaml==5.4.1
RUN pip install fsspec==2021.10.1
RUN pip install gcsfs==2021.10.1
RUN pip install tensorflow-text==2.7.0
RUN pip install tf-models-official==2.7.1
RUN pip install pyglove==0.1.0
RUN pip install google-cloud-storage==1.42.3

# Prepare workspace
RUN mkdir /root/training
COPY . /root/training/
ENV PATH="/root/training/:$PATH"

WORKDIR /root/training
RUN mv tf_vision/cloud_search_main.py tf_vision_cloud_search_main.py
ENTRYPOINT ["python", "tf_vision_cloud_search_main.py"]
