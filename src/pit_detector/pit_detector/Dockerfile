# 
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /code

RUN apt-get update
RUN apt-get install git -y
RUN apt-get install build-essential python3-dev python3-setuptools python3-pip libomp-dev -y
RUN pip install Cython

COPY ./requirements.txt /code/requirements.txt
RUN pip install --upgrade -r /code/requirements.txt

COPY ./src /code/src
COPY ./model_data /code/model_data
COPY ./test_data /code/test_data
WORKDIR /code/src

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "3000"]