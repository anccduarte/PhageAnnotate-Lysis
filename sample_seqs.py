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
def get_new_file(file, file_seqs, wanted_seqs):
    if file_seqs < wanted_seqs: return False
    rand = random.sample(range(file_seqs), wanted_seqs)
    new_fname = f"{file.split('.')[0]}_sample.fasta"
    with open(new_fname, "w") as wfile:
        records = SeqIO.parse(file, format="fasta")
        for i, rec in enumerate(records):
            if i in rand: wfile.write(f">{rec.description}\n{rec.seq}\n\n")
    return True


# Ficheiros sobre os quais se pretende efetuar a escolha aleatória de sequências
files = ["txid10239_proteins_3000_filt.fasta", "txid10239_endolysins_6000_filt.fasta",
         "txid10239_holins_6000_filt.fasta", "txid10239_negative_proteins_4000_filt.fasta"]

# Determinar o número de sequências em cada um dos ficheiros fasta (tup -> (file, counts))
file_tups = [(file, get_num_seqs(file)) for file in files]

# Retirar da lista 'files_with_counts' o ficheiro cuja contagem é mais baixa
min_file, min_counts = file_tups.pop(file_tups.index(min(file_tups, key = lambda x: x[1])))

# Geração dos novos ficheiros fasta contendo sequências escolhidas de forma aleatória
generate_files = [get_new_file(file, counts, min_counts) for file, counts in file_tups]
print(generate_files)
        