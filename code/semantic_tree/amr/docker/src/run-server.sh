#!/bin/bash
#@meta {author: "Paul Landes"}
#@meta {desc: "server start script", date: "2025-07-28"}


PROG=$(basename $0)
MODEL_DIR=/opt/models


# log message with separator
function prhead() {
    echo "--------------------${1}:"
}

# fail out of the program with an error message
function bail() {
    msg=$1 ; shift
    usage=$1 ; shift
    echo "$PROG: error: $msg" > /dev/stderr
    if [ "$usage" == 1 ] ; then
	printf "${USAGE}\n" > /dev/stderr
    fi
    exit 1
}

# make sure the last command ran was successful and fail otherwise
function assert_success() {
    ret=$1 ; shift
    if [ $ret -ne 0 ] ; then
	bail "last command failed"
    fi
}

# add base model
function download_model() {
    if [ ! -d ${MODEL_DIR}/bart-large ] ; then
	prhead "downloading model"
	git lfs install
	assert_success $?

	( cd ${MODEL_DIR} && git clone https://huggingface.co/facebook/bart-large )
	assert_success $?
    fi
}

function start() {
    prhead "starting server"
    export PYTHONPATH=$PYTHONPATH:.
    export TRANSFORMERS_OFFLINE=1
    BASE_DIR=${BASE_DIR:-/opt}
    mkdir -p ${BASE_DIR:-/opt}/data

    python bin/inference_server.py \
	   --model ${MODEL_DIR}/bart-large \
	   --checkpoint ${BASE_DIR:-/opt}/models/model.pt \
	   --datadir ${BASE_DIR:-/opt}/data \
	   --beam-size 5 \
	   --max-tokens 200 \
	   --batch-size 500 \
	   --penman-linearization \
	   --use-pointer-tokens \
	   > ${BASE_DIR:-/opt}/data/server.log 2>&1
}

function main() {
    download_model
    start
}

main $@
