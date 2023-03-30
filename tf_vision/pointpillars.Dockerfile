# This Dockerfile is for pointpillars model only.

# Pointpillars training pipeline may use WOD-2.6, and WOD-2.6 requires
# the tensorflow version to be 2.6 (b/237606169). But the Python version of
# tf-2.6 docker image is too low. So we install Python-3.9 and tf-2.6 manually.

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

# Install Python with higher version
ARG py_version=3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python${py_version} \
        python${py_version}-dev \
        python${py_version}-distutils \
        python${py_version}-venv
RUN python${py_version} -m pip install six

# Switch the default Python to the new one
ENV VENV_PATH=/venv
RUN python${py_version} -m venv --system-site-packages $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

# Install basic libs
RUN python${py_version} -m pip install --upgrade setuptools
RUN python${py_version} -m pip install --upgrade pip
RUN python${py_version} -m pip install --upgrade distlib

# Install tensorflow manually in upgraded Python
RUN python${py_version} -m pip install tf-models-official==2.7.2
RUN python${py_version} -m pip install --upgrade --force-reinstall tensorflow==2.6.0
RUN python${py_version} -m pip install --upgrade --force-reinstall keras==2.6.0
RUN python${py_version} -m pip install waymo-open-dataset-tf-2-6-0

# Install tools
RUN python${py_version} -m pip install cloud-tpu-client==0.10
RUN python${py_version} -m pip install pyyaml==5.4.1
RUN python${py_version} -m pip install fsspec==2021.10.1
RUN python${py_version} -m pip install gcsfs==2021.10.1
RUN python${py_version} -m pip install pyglove==0.1.0
RUN python${py_version} -m pip install google-cloud-storage==1.42.3

# Prepare workspace
RUN mkdir /root/training
COPY . /root/training/
ENV PATH="/root/training/:$PATH"

WORKDIR /root/training
RUN mv tf_vision/cloud_search_main.py tf_vision_cloud_search_main.py
ENTRYPOINT ["python", "tf_vision_cloud_search_main.py"]
