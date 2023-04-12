FROM huggingface/transformers-pytorch-gpu

RUN apt update

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN python3 app.py test.fasta output.fasta