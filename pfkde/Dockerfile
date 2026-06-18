FROM ghcr.io/timeeval/python3-base:0.3.0

LABEL maintainer="filipejinacio@tecnico.ulisboa.pt"
# LABEL org.opencontainers.image.licenses=MIT

ENV ALGORITHM_MAIN="/app/algorithm.py"

# install algorithm dependencies
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY manifest.json /app/
COPY model.py /app/
COPY algorithm.py /app/
