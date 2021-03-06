.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = MAIN_TEMPLATE
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements/requirements_dev.txt

## Make Dataset
data:
	$(PYTHON_INTERPRETER) src/data/make_dataset.py


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src
	black src/*.py
	pylint --disable=R,C ./src


## docker build image
build_docker_image:
	docker build -t model .


## docker run image
run_docker_entrypoint: build_docker_image
	docker run -it --entrypoint /bin/bash model


## docker expose port to run notebooks
run_predict: build_docker_image
	# docker run model
	docker run -p 8000:8000 -t -i model


## Download Data from remote directory


## Test python environment is setup correctly
test:
	pytest -vv --cov-report term-missing --cov=app tests/*.py
	# pytest -vv --cov-report term-missing --cov=app tests/test_data.py


preprocess:
	python src/data/make_dataset.py
	python src/features/build_features.py

train_model:
	# dvc repro
	python src/models/pipeline.py
	python src/models/train_model.py
predict:
	# dvc repro
	python src/models/predict_model.py


hello:
	docker run --rm -d -v $(PWD):/app -p 8000:8000 model
