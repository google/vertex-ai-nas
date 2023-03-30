FROM tensorflow/tensorflow:1.15.0-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive

# This is added to fix docker build error related to Nvidia key update.
RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

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
        git \
        python \
        python-dev \
        python-pip \
        python-setuptools \
        python-tk && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk && \
    apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip setuptools wheel
RUN pip install google-cloud-storage==1.42.3
RUN pip install google-api-python-client==2.29.0
RUN pip install oauth2client==4.1.3
RUN pip install Cython==0.29.24
RUN pip install matplotlib==3.3.4
RUN pip install pycocotools==2.0.2
RUN pip install pyyaml==5.4.1
RUN pip install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
RUN pip install opencv-python-headless==4.1.2.30
RUN pip install opencv-python==4.5.3.56
RUN pip install lvis==0.5.3
RUN pip install cloudml-hypertune==0.1.0.dev6
RUN pip install typing==3.7.4.3
RUN pip install pydot==1.4.2
RUN pip install fsspec==2021.10.1
RUN pip install gcsfs==2021.10.1
RUN pip install pyglove==0.1.0


# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg


RUN mkdir /root/training/
COPY . /root/training/
ENV PATH="/root/training/:${PATH}"

WORKDIR /root/training/
RUN mv tf1/cloud_search_main.py tf1_cloud_search_main.py
ENTRYPOINT ["python", "tf1_cloud_search_main.py"]
