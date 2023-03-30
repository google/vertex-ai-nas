FROM tensorflow/tensorflow:2.7.0-gpu

ENV DEBIAN_FRONTEND=noninteractive

# This is added to fix docker build error related to Nvidia key update.
RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install google cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
    | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

RUN pip install --upgrade pip
RUN pip install Pillow==8.4.0 && \
    pip install pyglove==0.1.0 && \
    pip install google-cloud-storage==2.6.0 && \
    pip install protobuf==3.20.*

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN mkdir /root/training
COPY . /root/training/
ENV PATH="/root/training/:${PATH}"

WORKDIR /root/training
RUN mv third_party/tutorial/tutorial1_mnist_search.py tutorial1_mnist_search.py
ENTRYPOINT ["python", "tutorial1_mnist_search.py"]
