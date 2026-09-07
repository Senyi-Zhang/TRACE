FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

## User environment
#
ENV DEV_USER=devusr
ENV DEV_GROUP=devgrp
ENV DEV_UID=1110
ENV DEV_GID=1110
ENV APP_HOME=/app
ENV PYTHON_VER_MAJ=3.8
ENV PYTHON_VER_MIN=${PYTHON_VER_MAJ}.15

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


## Build tools
#
# install basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        file \
        ssh \
        git \
        git-lfs \
        unzip \
        wget \
        curl \
        vim \
        build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev llvm \
        libncurses5-dev libncursesw5-dev xz-utils tk-dev \
        libffi-dev


# install Python
#
WORKDIR /tmp
RUN wget --no-verbose https://www.python.org/ftp/python/${PYTHON_VER_MIN}/Python-${PYTHON_VER_MIN}.tgz
RUN tar xf Python-${PYTHON_VER_MIN}.tgz
WORKDIR Python-${PYTHON_VER_MIN}
RUN ./configure --enable-optimizations && \
	make -j8 && \
	make altinstall
RUN cd /bin && \
    ln -s /usr/local/bin/python3.8 python && \
    ln -s /usr/local/bin/pip3.8 pip


# Rust and build the Python tokenizers package with it
#
WORKDIR /tmp
RUN pip install -U setuptools-rust wheel
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly-2020-02-01 -y
RUN git clone https://github.com/huggingface/tokenizers
WORKDIR /tmp/tokenizers
RUN git checkout python-v0.7.0
WORKDIR /tmp/tokenizers/bindings/python
RUN /bin/bash -c "PATH=$(echo /root/.rustup/toolchains/*/bin):$PATH ; python setup.py bdist_wheel"
RUN mkdir /inst && mv dist/* /inst && pip install /inst/*


## Spring AMR parser
#
WORKDIR ${APP_HOME}
RUN cd / && \
    git clone https://github.com/SapienzaNLP/spring app
# service requirements
COPY src/requirements.txt .
RUN pip install -r requirements.txt
# reduce logging noise
RUN pip uninstall -y wandb
# next two deps needed for (optional) parse caching
RUN pip install pandas~=1.5.3
RUN pip install --no-deps zensols.db~=1.3.0


## Add service
#
COPY src/inference_server.py bin
COPY src/run-server.sh .
COPY src/entrypoint.sh .

## Setup and install user account
#
# create development user/group
RUN addgroup --gid $DEV_GID $DEV_GROUP && \
    adduser --gecos 'Development User' --disabled-password \
      --uid $DEV_UID --gid $DEV_GID --home ${APP_HOME} $DEV_USER

# volume mapped
RUN mkdir -p /opt/data && mkdir -p /opt/models

# make application user ownership
RUN chown -R $DEV_USER:$DEV_GROUP .
RUN chown -R $DEV_USER:$DEV_GROUP /opt/data
RUN chmod 0755 /opt/data


## Cleanup
#
RUN apt-get -y autoremove && \
    apt-get -y clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* && \
    rm -rf /var/tmp/* \
    rm -rf ~/.cache


## Config
#
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
