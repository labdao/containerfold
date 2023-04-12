FROM nvidia/cuda:11.4.0-base-ubuntu20.04

# Set geographic area environment variable
ENV DEBIAN_FRONTEND=noninteractive

# installing packages required for installation
RUN echo "downloading basic packages for installation"
RUN apt-get update
RUN apt-get install -y tmux wget curl nano
RUN apt-get install -y gcc
RUN apt-get install -y python3-pip python3-dev

# installing application specific packages
RUN apt-get install -y git aria2
#RUN pip install numpy jax==0.4.4
RUN pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.0
RUN mkdir /params && aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar && \
    tar -xf alphafold_params_2022-03-02.tar -C /params