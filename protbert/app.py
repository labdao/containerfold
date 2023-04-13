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
import csv

app = typer.Typer()

def softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr), axis=0)

def generate_embedding(record, tokenizer, model, output_path):
    sequence = re.sub(r"[UZOB]", "X", str(record.seq))
    spaced_sequence = " ".join(sequence)
    print("embedding sequence: ", spaced_sequence)
    encoded_input = tokenizer(spaced_sequence, return_tensors='pt')
    model_output = model(**encoded_input)
    embeddings = model_output.last_hidden_state.detach().numpy()
    np.savetxt(os.path.join(output, f"{record.id}_encoded.csv"), embeddings[0], delimiter=",")

def fill_mask(record, unmasker, output_path):
    sequence = str(record.seq)
    spaced_sequence = " ".join(sequence)
    spaced_masked_sequence = spaced_sequence.replace("X", "[MASK]")
    print("filling masked sequence: ", spaced_masked_sequence)
    predictions = unmasker(spaced_masked_sequence)
    
    with open(os.path.join(output, f"{record.id}_filled_mask.json"), "w") as f:
        json.dump(predictions, f)

def generate_scoring_matrix(record, tokenizer, masked_model, output_path):
    sequence = str(record.seq)
    print("generating scoring matrix for sequence: ", sequence)

    all_token_scores = []
    for idx in range(len(sequence)):
        x_sequence = sequence[:idx] + "X" + sequence[idx+1:]
        spaced_sequence = " ".join(x_sequence)
        spaced_masked_sequence = spaced_sequence.replace("X", "[MASK]")
        print("tokenising masked sequence: ", spaced_masked_sequence)
        encoded_input = tokenizer(spaced_masked_sequence, return_tensors='pt')

        with torch.no_grad():
            model_output = masked_model(**encoded_input)

        scores = model_output.logits
        print("scored sequence: ", scores)
        mask_position = torch.tensor([idx], dtype=torch.long)
        mask_scores = scores[0, mask_position, :]

        token_scores = torch.softmax(mask_scores, dim=-1).squeeze()

        token_score_dict = {}
        for token, token_score in zip(tokenizer.vocab.keys(), token_scores):
            token_score_dict[token] = token_score.item()

        all_token_scores.append(token_score_dict)

    # Write the scoring matrix to a CSV file
    output_file = os.path.join(output_path, f"{record.id}_scoring_matrix.csv")
    with open(output_file, 'w', newline='') as csvfile:
        print("writing scoring matrix to " + output_file)
        fieldnames = ['position', 'identity'] + list(tokenizer.vocab.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, token_scores in enumerate(all_token_scores):
            row = {'position': i+1, 'identity': sequence[i]}
            row.update(token_scores)
            writer.writerow(row)

    print(f"scoring matrix saved to {output_file}")
    return(output_file)

def top_k(scoring_matrix_path, k, output_path):
    scoring_matrix = pd.read_csv(scoring_matrix_path)

    

    print("generating top k sequences")
    top_k_sequences = []
    for i, row in scoring_matrix.iterrows():
    # Excluding the first two columns (position and identity) and finding the k largest probabilities
        numeric_series = row.iloc[2:].astype(float)
        top_k_tokens = numeric_series.nlargest(k).index.values
        if i == 0:
            for token in top_k_tokens:
                top_k_sequences.append(token)
                print(top_k_sequences)
        else:
            new_sequences = []
            for seq in top_k_sequences:
                for token in top_k_tokens:
                    new_sequences.append(seq + token)
                    print(new_sequences)
            top_k_sequences = new_sequences
    print(top_k_sequences)

    # Convert the resulting sequences to SeqRecord objects
    seq_records = []
    for i, seq in enumerate(top_k_sequences):
        seq_record = SeqRecord(Seq.Seq(seq), id=f"sequence_{i+1}", description="")
        seq_records.append(seq_record)
    print(f"Generated {len(seq_records)} sequences.")

    # Write the SeqRecords to a FASTA file
    output_file = os.path.join(output_path, "top_k.fasta")
    with open(output_file, "w") as output_handle:
        SeqIO.write(seq_records, output_handle, "fasta")

def sample_n(sequence, unmasker, n):
    # Your code for sample_n_mode here
    print("stay tuned")

@app.command()
def main(input: str, output_path: str, mode: str, k: int = 1, n: int = 10):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load models and tokenizer
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    masked_model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert")
    model = BertModel.from_pretrained("Rostlab/prot_bert")
    unmasker = pipeline('fill-mask', model=masked_model, tokenizer=tokenizer)

    # Load input sequence
    record = SeqIO.read(input, "fasta")

    if mode == "embedding":
        generate_embedding(record, tokenizer, model, output_path)
    elif mode == "fill-mask":
        fill_mask(record, unmasker, output_path)
        # Save the result to a file or print it as needed
    elif mode == "scoring-matrix":
        generate_scoring_matrix(record, tokenizer, masked_model, output_path)
    elif mode == "top-k":
        matrix_path = generate_scoring_matrix(record, tokenizer, masked_model, output_path)
        top_k(matrix_path, k, output_path)
    elif mode == "sample-n":
        sample_n(record, unmasker, n, output_path)
    else:
        typer.echo("Invalid mode. Please choose from 'embedding', 'fill-mask', 'scoring-matrix', 'top-k', or 'sample-n'.")

if __name__ == "__main__":
    typer.run(main)
