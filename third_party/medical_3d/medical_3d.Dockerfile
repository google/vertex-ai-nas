# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Install the latest version of pytorch
FROM gcr.io/cloud-ml-public/training/pytorch-gpu.1-6
WORKDIR /root

# Installs pandas, google-cloud-storage and tensorflow.
RUN pip install --upgrade pip==21.3.1

# Install tensorflow to use its tf.gfile lib to access objects in GCS.
# TODO: remove the dependency on tensorflow and use fsspec instead.
RUN pip install pandas==1.3.4 google-cloud-storage==1.42.3 tensorflow==1.15
RUN pip install pyglove==0.1.0

# Install Monai.
RUN pip install "monai[all]==0.5.2"
RUN pip install -q matplotlib==3.4.3

RUN pip install protobuf==3.20.*

RUN mkdir /root/training
COPY . /root/training/
ENV PATH="/root/training/:${PATH}"

WORKDIR /root/training
RUN mv third_party/medical_3d/medical_3d_main.py medical_3d.py
ENTRYPOINT ["python", "medical_3d.py"]
