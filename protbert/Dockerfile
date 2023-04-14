FROM huggingface/transformers-pytorch-gpu

RUN apt update

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# test battery
RUN python3 app.py test.fasta /test --mode embedding
RUN python3 app.py test.fasta /test --mode fill-mask
RUN python3 app.py test.fasta /test --mode conditional-probability
RUN python3 app.py test.fasta /test --mode joint-probability
RUN python3 app.py test.fasta /test --mode generate