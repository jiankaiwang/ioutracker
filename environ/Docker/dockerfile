FROM ubuntu:latest

LABEL maintainer="JianKai Wang <GLjankai@gmail.com>"

RUN apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get update

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y \
    wget \
    git \
    unzip \
    build-essential \
    tzdata \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    python3-pip && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN mkdir -p /tracker && \
    git clone https://github.com/jiankaiwang/ioutracker /tracker/ioutracker

WORKDIR /tracker/ioutracker
RUN pip install -q --no-cache-dir -e .

WORKDIR /tracker/ioutracker
CMD ["jupyter", "notebook", "--notebook-dir", "/tracker/ioutracker", "--ip", "0.0.0.0", "--allow-root"]