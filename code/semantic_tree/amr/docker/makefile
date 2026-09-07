##@meta {desc: 'build and deployment for docker projects', date: '2024-02-17'}


## Build system
#
#
# type of project
PROJ_TYPE=		docker


## Project
#
DOCKER_IMG_NAME=	springserv
DOCKER_USER=		plandes
DOCKER_UP_DEPS=		mkdata

DOCKER_BUILD_ARGS =	--progress plain


## Includes
#
include ../zenbuild/main.mk


## Targets
#
.PHONY:			mkdata
mkdata:
			mkdir -p data


.PHONY:			rerun
rerun:			down build rmzombie rmi up

.PHONY:			testmodel
testmodel:
			wget -q -O - --post-data='{"sents": ["Obama was the 44th president."]}' \
			  --header='Content-Type:application/json' \
			  'http://localhost:8080/parse' | \
			  jq -r '.amrs."0"."graph"'
