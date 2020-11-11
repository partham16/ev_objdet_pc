FROM python:3.8-slim

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

COPY requirements_cpu.txt ./requirements_cpu.txt

RUN apt-get update && apt-get install -y gcc \
    && pip install -r requirements_cpu.txt \
    && pip install streamlit==0.70.0 \
    && apt-get remove -y gcc \
    && apt-get -y clean && apt-get -y autoremove && rm -rf /var/lib/apt/lists/*

RUN mkdir -p models \
    && gdown "https://drive.google.com/uc?id=1vwHgzExqIyt9Ln-l9CWmMTwecULnfnUd" -O ./models/fasterrcnn_r50.pth

COPY streamlit_docker.py ./streamlit_docker.py
COPY src/* ./src/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_docker.py"]
