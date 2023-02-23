# FROM ubuntu:focal
FROM athbaltzis/colabfold_proteinfold:v0.11

# installing packages required for installation
RUN echo "downloading basic packages for installation"
RUN apt-get update
RUN apt-get install -y tmux wget curl nano

# checking installation of tools
RUN gcc --version
RUN nvcc --version

# download weights
RUN cd colabfold_batch
RUN colabfold-conda/bin/python3.7 -m colabfold.download
RUN mv /root/.cache/colabfold /colabfold_batch/

# create test
RUN mkdir /test
RUN echo ">sp|P61823"$'\n'"MALKSLVLLSLLVLVLLLVRVQPSLGKETAAAKFERQHMDSSTSAASSSNYCNQMMKSRN" > /test/test.fasta

# run test
RUN cd /
RUN colabfold_batch --amber --templates --num-recycle 3 --use-gpu-relax /test/test.fasta outputdir/ --cpu