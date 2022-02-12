FROM python:3.8

WORKDIR /appai/

RUN apt-get update && apt-get install -y \
    automake \
    curl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


COPY requirements/*.txt setup.py /appai/

RUN pip install --requirement /appai/requirements_dev.txt

COPY . /appai/