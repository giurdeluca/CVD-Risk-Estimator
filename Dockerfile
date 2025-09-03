FROM continuumio/miniconda3:4.10.3
RUN conda update conda
RUN conda install python=3.8 -y

## Create the environment:
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Set the working directory
WORKDIR /cvd-risk-estimator

COPY . .
RUN mkdir -p input output

ENTRYPOINT ["python3", "pred_batch_updated_map.py"]
