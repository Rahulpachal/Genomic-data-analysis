!pip install biopython

from google.colab import drive
drive.mount('/content/drive')


# %pwd

directory_path = "/content/drive/MyDrive/NCBI_Dataset/data"

import os # Import the 'os' module to use its functions

directory_path = "/content/drive/MyDrive/NCBI_Dataset/data"

# List files in the directory
files = os.listdir(directory_path)
print("Files in directory:", files)

from Bio import SeqIO
import os

# Corrected path to the FASTA file, ensure this path is correct and the file exists
fasta_file_path = "/content/drive/MyDrive/NCBI_Dataset/data/GCA_000001405.29/GCA_000001405.29_GRCh38.p14_genomic.fna"  # Added .fna extension

# Check if the file exists
if os.path.exists(fasta_file_path):
    # Read and access the FASTA file
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        print(f"ID: {record.id}")
        print(f"Description: {record.description}")
        print(f"Sequence: {record.seq[:100]}...")  # Print the first 100 bases for brevity
else:
    print(f"File not found: {fasta_file_path}")

import json

# Reading JSON file
with open('/content/drive/MyDrive/NCBI_Dataset/data/dataset_catalog.json', 'r') as file:
    catalog_data = json.load(file)
    print(json.dumps(catalog_data, indent=4))

# Reading JSON Lines file
with open('/content/drive/MyDrive/NCBI_Dataset/data/assembly_data_report.jsonl', 'r') as file:
    for line in file:
        record = json.loads(line)
        print(json.dumps(record, indent=4))

import pandas as pd

# Reading TSV file
tsv_file_path = '/content/drive/MyDrive/NCBI_Dataset/data/data_summary.tsv'
data_summary = pd.read_csv(tsv_file_path, sep='\t')
print(data_summary)

import os
import gzip

# Example of listing contents in a directory
directory_path = '/content/drive/MyDrive/NCBI_Dataset/data/GCA_000001405.29'
files = os.listdir(directory_path)
print("Files in directory:", files)

# Example of reading a compressed FASTA file
# Update this path to the actual compressed FASTA file within the directory
fasta_file_path = '/content/drive/MyDrive/NCBI_Dataset/data/GCA_000001405.29/sequence.fna.gz'  # Assuming the compressed file is named 'sequence.fna.gz'

# Check if the file exists before attempting to open it
if os.path.exists(fasta_file_path):
    with gzip.open(fasta_file_path, 'rt') as file:
        for line in file:
            print(line.strip())
            # Break after a few lines for brevity
            if line.startswith('>'):
                break
else:
    print(f"File not found: {fasta_file_path}") # Print an error message if the file is not found

!pip install biopython scikit-learn pandas numpy matplotlib seaborn

from Bio import SeqIO

def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        print(f"ID: {record.id}")
        print(f"Description: {record.description}")
        print(f"Sequence: {record.seq[:100]}...")  # Print first 100 bases for brevity
        sequences.append(record.seq)
    return sequences

# Example usage
file_path = "/content/drive/MyDrive/NCBI_Dataset/data/GCA_000001405.29/GCA_000001405.29_GRCh38.p14_genomic.fna"
sequences = read_fasta(file_path)

from collections import Counter

def count_codons(sequence):
    codons = [str(sequence[i:i+3]) for i in range(0, len(sequence), 3) if len(sequence[i:i+3]) == 3]
    codon_count = Counter(codons)
    return codon_count

# Example usage
for seq in sequences:
    codon_count = count_codons(seq)
    print(codon_count)

def extract_features(sequence):
    features = {}
    codon_count = count_codons(sequence)
    features['max_codon'] = max(codon_count, key=codon_count.get)
    features['min_codon'] = min(codon_count, key=codon_count.get)
    features['gc_content'] = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
    features['sequence_length'] = len(sequence)
    return features

# Example usage
features_list = [extract_features(seq) for seq in sequences]
import pandas as pd
features_df = pd.DataFrame(features_list)
print(features_df)

import numpy as np

# Generate dummy labels (for binary classification example)
# Ensure the length of labels matches the length of features_df
num_samples = len(features_df)
labels = np.random.randint(0, 2, size=num_samples)

print("Length of features_df:", len(features_df))
print("Length of labels:", len(labels))
print("Features DataFrame:")
print(features_df)
print("Labels:")
print(labels)

print("Length of features_df:", len(features_df))
print("Length of labels:", len(labels))

from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features_df, labels, test_size=0.3, random_state=42)

print("Shapes of train and test sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", len(y_train))
print("y_test:", len(y_test))
