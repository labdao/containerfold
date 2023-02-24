FROM ghcr.io/labdao/moler:main

# installing packages required for installation
RUN echo "downloading basic packages for installation"
RUN apt-get update
RUN apt-get install -y tmux wget curl nano

# checking installation of tools
RUN gcc --version
RUN nvcc --version

# creating param dir
# TODO harmonize with public Moler
RUN mkdir model_dir