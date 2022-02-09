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