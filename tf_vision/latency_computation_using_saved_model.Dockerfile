# NOTE: The source for this starting-point is:
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#unique_1446644071
FROM nvcr.io/nvidia/tensorflow:22.05-tf2-py3

ENV DEBIAN_FRONTEND=noninteractive

# This is added to fix docker build error related to Nvidia key update.
RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

RUN pip install --upgrade pip==21.3.1 && \
    pip install Pillow==8.4.0 && \
    pip install nvgpu==0.9.0 && \
    pip install fsspec==2021.10.1 && \
    pip install gcsfs==2021.10.1 && \
    pip install pyglove==0.1.0

# NOTE: Installing protobuf in the very end else it gives error when importing
# tensorflow.
RUN pip install protobuf==3.20.*

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN mkdir /root/training
COPY . /root/training/
ENV PATH="/root/training/:$PATH"

WORKDIR /root/training
RUN mv tf_vision/latency_computation_using_saved_model.py tf_vision_latency_computation_using_saved_model.py
ENTRYPOINT ["python", "tf_vision_latency_computation_using_saved_model.py"]
