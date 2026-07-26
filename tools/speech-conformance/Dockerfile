FROM python@sha256:da5aee29682d12a6649f51c8d6f15b87deb3e6c524b923c41d0cb3304d07c913

ARG COQUI_TTS_REV=0cf3265a4686d7e856bd472cdaf1572d61cab2b8
ARG DESCRIPT_MELGAN_REV=6488045bfba1975602288de07a58570c7b4d66ea

RUN apt-get update \
    && apt-get install --yes --no-install-recommends espeak-ng \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --disable-pip-version-check \
        torch==1.13.1 \
        torchaudio==0.13.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY tools/speech-conformance/requirements.txt /tmp/requirements.txt
RUN python -m pip install --disable-pip-version-check -r /tmp/requirements.txt
RUN git clone https://github.com/coqui-ai/TTS.git /opt/coqui-tts \
    && git -C /opt/coqui-tts checkout "$COQUI_TTS_REV" \
    && python -m pip install --disable-pip-version-check --no-deps /opt/coqui-tts
RUN git clone https://github.com/descriptinc/melgan-neurips.git /opt/descript-melgan \
    && git -C /opt/descript-melgan checkout "$DESCRIPT_MELGAN_REV"

ENV PYTHONPATH=/opt/coqui-tts:/opt/descript-melgan
WORKDIR /workspace
ENTRYPOINT ["python", "scripts/coqui-reference.py"]
