# -*- coding: utf-8 -*-

from Bio import SeqIO
import random

"""
Escolha aleatória de sequências tendo por base o número de sequências de spaninas recolhidas (número de
sequências mais limitado -> 1953 sequências recolhidas após filtragem)
"""

# Determina o número de sequências contidas num ficheiro fasta
def get_num_seqs(file):
    records = SeqIO.parse(file, format="fasta")
    num_seqs = 0
    for rec in records: num_seqs += 1
    return num_seqs

# Gera um novo ficheiro fasta contendo um número de sequências definido pelo utilizador
def get_new_file(file, file_seqs, wanted_seqs, scale):
    if file_seqs < scale*wanted_seqs: return False
    rand = random.sample(range(file_seqs), scale*wanted_seqs)
    new_fname = f"{file.split('.')[0]}_sample_{scale}x.fasta"
    with open(new_fname, "w") as wfile:
        records = SeqIO.parse(file, format="fasta")
        for i, rec in enumerate(records):
            if i in rand: wfile.write(f">{rec.description}\n{rec.seq}\n\n")
    return True


# Ficheiros sobre os quais se pretende efetuar a escolha aleatória de sequências (3x negativos)
files = ["txid28883_proteins_2500_filt.fasta", "txid28883_endolysins_6000_filt.fasta",
         "txid28883_holins_6000_filt.fasta", "txid28883_negative_proteins_4000_filt.fasta",
         "txid28883_negative_proteins_4000_filt.fasta", "txid28883_negative_proteins_4000_filt.fasta"]

# Determinar o número de sequências em cada um dos ficheiros fasta (tup -> (file, counts))
file_tups = [(file, get_num_seqs(file)) for file in files]

# Retirar da lista 'files_with_counts' o ficheiro cuja contagem é mais baixa
min_file, min_counts = file_tups.pop(file_tups.index(min(file_tups, key = lambda x: x[1])))

# Scale number of sequences
scale = [1, 1, 1, 3, 6] # negatives are scaled 1x, 3x and 6x

# Geração dos novos ficheiros fasta contendo sequências escolhidas de forma aleatória
generate_files = [get_new_file(file, counts, min_counts, sc) for (file, counts), sc in zip(file_tups, scale)]
print(generate_files)
        
