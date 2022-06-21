# -*- coding: utf-8 -*-

from Bio import Seq, SeqIO
import pandas as pd

"""
Verificar qual a percentagem de sequências de spaninas recolhidas na base de dados 'Nucleotide' do NCBI
que se encontram na base de dados manualmente curada 'Additional_S2.xlsx'
"""

# xlsx - ficheiro excel
xlsx = pd.ExcelFile("Additional_S2.xlsx")

# data1, data2 - data sheets no ficheiro excel
data1 = pd.read_excel(xlsx, "Table S1a-Sequences for 2CS")
data2 = pd.read_excel(xlsx, "Table S1b- Sequences for 1CS")

# colunas de ambas as sheets que contêm sequências
seqs1_1 = data1["i-spanin primary structure (curated)"].tolist()
seqs1_2 = data1["o-spanin primary structure (curated)"].tolist()
seqs2 = data2["u-spanin primary structure"].tolist()

# número de sequências de spaninas no ficheiro Excel
db_seqs = set(seqs1_1 + seqs1_2 + seqs2)
print(f"Number of sequences in DB: {len(db_seqs)}")

# verificar sequências que se encontam em ambos os ficheiros (Excel e fasta)
records = SeqIO.parse("txid28883_proteins_2500_filt.fasta", format="fasta")
seqs, seqs_in = 0, 0
for record in records:
    seqs += 1
    seq = record.seq
    while len(seq) % 3 != 0: seq = f"N{seq}"
    spanin = Seq.Seq(seq).translate(table=11)[:-1]
    if spanin in db_seqs: seqs_in += 1

# número de sequências de spaninas no ficheiro fasta
print(f"Number of sequences in fasta file: {seqs}")

# percentagem de sequências presentes na DB e no ficheiro fasta
print(f"Intersection: {seqs_in} ({(seqs_in/len(db_seqs))*100:.2f}%)")
