#!/bin/bash

# @meta {desc: 'amrspring server build', date: '2024-02-23'}
#
# This script builds out a server to run locally for those that don't want to
# run the Docker container in .../docker.  This script must be invoked this
# from the repository root.

PROG=$(basename $0)
USAGE="usage: $PROG <python home>"
TARG=./server
PY_HOME=$1 ; shift
PY_BIN=$PY_HOME/bin/python3
PYV_HOME=${TARG}/python
PYV_BIN=${PYV_HOME}/bin/python3
REPO_DIR=${TARG}/spring

if [ -z "$PY_HOME" ] ; then
    echo $USAGE
    exit 1
fi

function log() {
    echo "$PROG: $1"
}

function assert_python() {
    if [ ! -x "$PY_BIN" ] ; then
	echo "python interpreter not found: $PY_BIN"
	exit 1
    fi
    log "using Python biniary: $PY_BIN"

    pyver=$($PY_BIN --version | sed -ne 's/[^0-9]*\([0-9].*\)\..*/\1/p')
    if [ "$pyver" != "3.8" ] ; then
	echo "python version must be 3.8: $pyver"
	exit 1
    fi
    log "using Python version: $pyver"
}

function mk_targ_dir() {
    if [ -d "$TARG" ] ; then
	log "already exists: $TARG"
    else
	log "making server directory $TARG"
	mkdir -p $TARG
    fi
}

function mk_virenv() {
    if [ -d "$PYV_HOME" ] ; then
	log "python virtual home already exists in $PYV_HOME"
    else
	log "creating python virtual env in $PYV_HOME."
	$PY_BIN -m venv --copies $PYV_HOME
    fi
}

function clone_repo() {
    if [ -d $REPO_DIR ] ; then
	log "repo already cloned: $REPO_DIR"
    else
	log "cloning repo into $REPO_DIR"
	git clone https://github.com/SapienzaNLP/spring $REPO_DIR
    fi
}

function install_deps() {
    $PYV_BIN -m pip freeze | grep tokenizers
    if [ $? -eq 1 ] ; then
	log "installing python packages..."
	$PYV_BIN -m pip install -r $REPO_DIR/requirements.txt
	$PYV_BIN -m pip uninstall -y wandb
	# server
	$PYV_BIN -m pip install "Flask~=2.2.5"
	# next two deps needed for (optional) parse caching
	$PYV_BIN -m pip install "pandas~=1.5.3"
	$PYV_BIN -m pip install "zensols.util~=1.14.1"
	$PYV_BIN -m pip install --no-deps "zensols.db~=1.3.0"
	log "if installing tokenizers failed, look at docker/Dockerfile to install from source"
    else
	log "python packages already installed"
    fi
}

function install_script() {
    if [ -x "${REPO_DIR}/run-server.sh" ] ; then
       log "scripts already installed in $REPO_DIR"
    else
	log "installing scripts to $REPO_DIR..."
	cp ./docker/src/inference_server.py $REPO_DIR/bin
	cp ./docker/src/run-server.sh $REPO_DIR
	cp ./src/bin/template/* ${REPO_DIR}/..
	mkdir -p ${REPO_DIR}/models
    fi
    log "Completed.  Now add the model to use in ${REPO_DIR}/models/model.pt"
    log "Start by typing: ( cd server ; ./serverctl start )"
}

function main() {
    assert_python
    mk_targ_dir
    mk_virenv
    clone_repo
    install_deps
    install_script
}

main
