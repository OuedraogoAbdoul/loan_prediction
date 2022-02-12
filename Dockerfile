FROM python:3.8

WORKDIR /app/

RUN apt-get update && apt-get install -y \
    automake \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


COPY requirements/*.txt setup.py /app/

RUN pip install --requirement /app/requirements_dev.txt

COPY . /app/

EXPOSE 8000:8000

# CMD ["python3", "/app/serving_loan_prediction_model/app/app.py"]

CMD ["uvicorn", "serving_loan_prediction_model.app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
 