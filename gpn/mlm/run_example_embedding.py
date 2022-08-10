from Bio import SeqIO
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import sys

import gpn.mlm

checkpoint_path = sys.argv[1]
strand = sys.argv[2]
output_path = sys.argv[3]

genome = SeqIO.to_dict(SeqIO.parse("../../data/mlm/tair10.fa", format="fasta"))
window_size = 1000000
center = 3566700
seq = genome["Chr5"][center-window_size//2:center+window_size//2].seq
if strand == "+":
    print("Positive strand")
elif strand == "-":
    print("Negative strand")
    print(seq[:10])
    seq = seq.reverse_complement()
    print(seq[:10])
seq = str(seq)

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModel.from_pretrained(checkpoint_path)
model.eval()
# # chr5:3,566,700-3,567,700
#seq = "ATAAACATATCATAAATAAGATCAATATTAATAAAATAAATAGTTTTTTTTACGGGACGGATTGGCGGGACGAGTTTAGCAGGACGTAACTTAATAACAATTGTAAACTATAAAATAAAAATATTTTATAGATAGATACAATTTGCAAACTTTTATATATACTAACTTAAAAAAAAAATATTGTCCCCTGCGGTATAAGACGGGTTAAAAATCTAGTTGTTATTATTAAAGGAAATAAAATATCCTCATAAAACAATTTGTTGTAATCTATCTTTGGGCTAATGTTCTTATCCTACAAGACGAACCCTGACCGTATTCGTCGTAGAAAAAAAATTGCTTCGATCCCATCATTGAGTTCAATAATCGGCGCACAAAGGCCGATTCATAAAAACTCTAGGCCCATTAAAGTAAAGCCCATTCTCAACCCTATCCAGTCTCCCTGTATATATATATTTACGACACCAACCCAGCGTTGATATTTAATTTTCTTCAGTCAGAGATTTCGAAACCCTAGTCGATTTCGAGATCCAACTAACTCTGCTCCTTATCTCAGGTAAAATTCTCGCTCGAGAACTCAATTGCTTATCCAAAGTTCCAACTGAAGATGCTTTCCTACTGAATCTTAGGTTAATGTTTTGGATTTGGAATCTTACCCGAAATTTCTCTGCAGCTTGTTGAATTTGCGAAGTATGGGAGACGCTAGAGACAACGAAGCCTACGAGGAGGAGCTCTTGGACTATGAAGAAGAAGACGAGAAGGTCCCAGATTCTGGAAACAAAGTTAACGGCGAAGCTGTGAAAAAGTGAGTTTTATGGTTTCCTCGATATGTTTCATGTATACTACTGTGTGTTTAAATTTGTCGATTCTTAGATTACTACTTGATAACAAGTAGCAGTATGTGTTTAATTAGTTGCTTAACATATAACAATTGACTGAGTTCTTCATTGCTATAATTCCTGAAACCCACCCAATATTAGACTGTCGTGTGTTTCTCATATTG"
print(len(seq))
input_ids = tokenizer(seq, return_tensors="pt")["input_ids"]

with torch.no_grad():
    embedding = model(input_ids=input_ids).last_hidden_state[0].numpy()
print(embedding.shape)

np.save(output_path, embedding)