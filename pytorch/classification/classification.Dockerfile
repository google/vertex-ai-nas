# Copyright 2022 Google LLC
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

# Install pytorch-1.12.1 with cuda-11.3.
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

# Install additional libraries.
RUN pip install pyglove==0.1.0
RUN pip install torchmetrics==0.10.0
RUN pip install webdataset==0.2.26
RUN pip install google-cloud-storage==1.42.3

RUN mkdir /root/training
COPY . /root/training/
ENV PATH="/root/training/:${PATH}"

WORKDIR /root/training
RUN mv pytorch/classification/cloud_search_main.py pytorch_classification.py
ENTRYPOINT ["python", "pytorch_classification.py"]
