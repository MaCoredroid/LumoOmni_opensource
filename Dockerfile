FROM nvcr.io/nvidia/pytorch:25.09-py3

WORKDIR /opt/lumo

# Archive tools for dataset extraction + audio decode deps
RUN apt-get update && apt-get install -y p7zip-full zip unzip libsndfile1 && rm -rf /var/lib/apt/lists/*

# Preinstall project dependencies so runtime containers only need to mount the repo.
COPY qwen3-vlm/requirements.txt /tmp/requirements.txt

ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu130"
ARG TORCH_VERSION="2.11.0.dev20260129+cu130"
ARG TORCHVISION_VERSION="0.25.0.dev20260129+cu130"
ARG TORCHAUDIO_VERSION="2.11.0.dev20260129+cu130"
ARG DIFFUSERS_VERSION="0.36.0"
ARG TIMM_VERSION="1.0.24"
ARG ENCODEC_VERSION="0.1.1"
ARG SOUNDFILE_VERSION="0.13.1"
ARG SCIPY_VERSION="1.16.1"

RUN python3 -m venv --copies /opt/lumo/venv && \
    /opt/lumo/venv/bin/pip install --upgrade pip && \
    /opt/lumo/venv/bin/pip install --pre --index-url ${TORCH_INDEX_URL} \
      torch==${TORCH_VERSION} \
      torchvision==${TORCHVISION_VERSION} \
      torchaudio==${TORCHAUDIO_VERSION} && \
    /opt/lumo/venv/bin/pip install -r /tmp/requirements.txt && \
    /opt/lumo/venv/bin/pip install \
      soundfile==${SOUNDFILE_VERSION} \
      scipy==${SCIPY_VERSION} && \
    /opt/lumo/venv/bin/pip install --no-deps \
      diffusers==${DIFFUSERS_VERSION} \
      timm==${TIMM_VERSION} \
      encodec==${ENCODEC_VERSION}

ENV PATH="/opt/lumo/venv/bin:${PATH}"
ENV HF_HOME="/root/.cache/huggingface"
ENV HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"
ENV HF_HUB_CACHE="/root/.cache/huggingface/hub"

WORKDIR /workspace/lumoOmni
