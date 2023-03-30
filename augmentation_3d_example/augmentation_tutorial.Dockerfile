# NOTE: This docker runs python-3.6.
FROM tensorflow/tensorflow:1.15.4-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN pip install --upgrade pip==21.3.1
RUN pip install fsspec==2021.10.1 && \
    pip install gcsfs==2021.10.1 && \
    pip install pyglove==0.1.0

# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN mkdir /root/training
COPY . /root/training/
ENV PATH="/root/training/:${PATH}"

WORKDIR /root/training
RUN pip install augmentation_3d_example/lingvo-0.8.2-cp36-cp36m-manylinux2010_x86_64.whl
RUN mv augmentation_3d_example/augmentation_tutorial_train.py augmentation_tutorial_train.py
ENTRYPOINT ["python", "augmentation_tutorial_train.py"]
