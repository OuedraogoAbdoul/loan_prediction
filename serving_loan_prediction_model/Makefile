.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

PYTHON_INTERPRETER = python3

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -r requirements/requirements_dev.txt

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 app
	black app/*.py
	pylint --disable=R,C ./src


## docker build image
build:
	docker build -t model .


## docker run image
run_docker_image: build_docker_image
	docker run -it --entrypoint /bin/bash model


## docker expose port to run notebooks
run_notebooks:
	docker run -it --rm -p 8822:8822 loan_default




## Download Data from remote directory


## Test python environment is setup correctly
test:
	pytest -vv --cov-report term-missing --cov=app tests/*.py
	# pytest -vv --cov-report term-missing --cov=app tests/test_data.py

predict:
	# dvc repro
	python src/models/predict_model.py

uvicorn:
	uvicorn app.app:app --reload


