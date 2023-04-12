FROM nvidia/cuda:11.4.0-base-ubuntu20.04

# Set geographic area environment variable
ENV DEBIAN_FRONTEND=noninteractive

# installing packages required for installation
RUN echo "downloading basic packages for installation"
RUN apt-get update
RUN apt-get install -y tmux wget curl nano
RUN apt-get install -y gcc
RUN apt-get install -y python3-pip python3-dev

COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt