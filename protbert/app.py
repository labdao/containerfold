import typer
import os
import pandas as pd
import numpy as np
from transformers import BertForMaskedLM, BertTokenizer, pipeline, BertModel
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import torch
import re
import json

app = typer.Typer()

def softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr), axis=0)

def generate_embedding(record, tokenizer, bert_model, output):
    sequence = re.sub(r"[UZOB]", "X", str(record.seq))
    spaced_sequence = " ".join(sequence)
    print("embedding sequence: ", spaced_sequence)
    encoded_input = tokenizer(spaced_sequence, return_tensors='pt')
    bert_output = bert_model(**encoded_input)
    embeddings = bert_output.last_hidden_state.detach().numpy()
    np.savetxt(os.path.join(output, f"{record.id}_encoded.csv"), embeddings[0], delimiter=",")

def fill_mask(record, unmasker, output):
    sequence = str(record.seq)
    spaced_sequence = " ".join(sequence)
    spaced_masked_sequence = spaced_sequence.replace("X", "[MASK]")
    print("filling masked sequence: ", spaced_masked_sequence)
    predictions = unmasker(spaced_masked_sequence)
    
    with open(os.path.join(output, f"{record.id}_filled_mask.json"), "w") as f:
        json.dump(predictions, f)

def generate_scoring_matrix(record, unmasker):
    sequence = str(record.seq).replace("X", "[MASK]")
    spaced_sequence = " ".join(sequence)
    predictions = unmasker(spaced_sequence)

    # Initialize an empty DataFrame to store scores
    scored = pd.DataFrame()

    # Process predictions for each [MASK] token
    for idx, pred_list in enumerate(predictions):
        scores = [prediction["score"] for prediction in pred_list]
        temp_df = pd.DataFrame(scores).T
        temp_df.columns = [prediction["token_str"] for prediction in pred_list]

        # Merge the DataFrame with the scored DataFrame
        if idx == 0:
            scored = temp_df
        else:
            scored = scored.add(temp_df, fill_value=0)
    
    return scored


def top_k(sequence, unmasker, k):
    # Your code for top_k_mode here
    print("stay tuned")

def sample_n(sequence, unmasker, n):
    # Your code for sample_n_mode here
    print("stay tuned")

@app.command()
def main(input: str, output: str, mode: str, k: int = 10, n: int = 10):
    # Create output directory if it doesn't exist
    if not os.path.exists(output):
        os.makedirs(output)
    
    # Load models and tokenizer
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=True)
    masked_model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
    bert_model = BertModel.from_pretrained("Rostlab/prot_bert")
    unmasker = pipeline('fill-mask', model=masked_model, tokenizer=tokenizer)

    # Load input sequence
    record = SeqIO.read(input, "fasta")

    if mode == "embedding":
        generate_embedding(record, tokenizer, bert_model, output)
    elif mode == "fill-mask":
        fill_mask(record, unmasker, output)
        # Save the result to a file or print it as needed
    elif mode == "scoring-matrix":
        generate_scoring_matrix(record, unmasker, output)
    elif mode == "top-k":
        top_k(record, unmasker, k, output)
    elif mode == "sample-n":
        sample_n(record, unmasker, n, output)
    else:
        typer.echo("Invalid mode. Please choose from 'embedding', 'fill-mask', 'scoring-matrix', 'top-k', or 'sample-n'.")

if __name__ == "__main__":
    typer.run(main)
