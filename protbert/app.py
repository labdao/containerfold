# app.py
import typer
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer

app = typer.Typer()

def read_fasta_file(fasta_file: str):
    with open(fasta_file, "r") as f:
        fasta_content = f.read().strip()
    
    lines = fasta_content.split("\n")
    sequence = "".join(lines[1:])
    
    return sequence

def write_fasta_file(fasta_file: str, header: str, sequence: str):
    with open(fasta_file, "w") as f:
        f.write(f"{header}\n{sequence}\n")

@app.command()
def process_protein_sequence(
    input_sequence: Path = typer.Argument(..., help="Path to the protein sequence in FASTA format"),
    output_sequence: Path = typer.Argument(..., help="Path to the output FASTA file with the predicted sequence")
):
    if input_sequence.is_file():
        original_sequence = read_fasta_file(input_sequence)
        
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        model = AutoModelForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
        
        tokens = tokenizer(original_sequence, return_tensors="pt", padding=True, truncation=True)
        prediction = model(**tokens)
        
        # Process the prediction result as needed
        # For demonstration purposes, let's assume the predicted sequence is the original sequence reversed
        predicted_sequence = original_sequence[::-1]
        
        # Write the predicted sequence to the output FASTA file
        output_header = ">predicted_sequence"
        write_fasta_file(output_sequence, output_header, predicted_sequence)

        print(f"Processed protein sequence and saved to {output_sequence}")
    else:
        typer.echo(f"File not found: {input_sequence}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
