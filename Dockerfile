FROM nvcr.io/nvidia/pytorch:19.09-py3

# install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt