# -*- coding: utf-8 -*-

from Bio import Entrez, SeqIO, Seq
from tqdm.auto import tqdm
import os

class SeqMining:
    
    """
    Permite a recolha de sequências na base de dados 'Nucleotide' do NCBI.
    """
    
    def __init__(self, email: str, taxid: str, terms: list, num_ids: int, negatives: int = 0) -> None:
        
        """
        Inicializa uma instâcia da classe SeqMining.
        
        Parameters
        ----------
        :param email: O endereço de email do utilizador
        :param taxid: O taxid sobre o qual se pretende efetuar a pesquisa
        :param terms: Os termos a procurar na base de dados
        :param num_ids: O número de IDs a inspecionar na base de dados
        :param negatives: Indica se se pretende a recolha de sequências positivas ou negativas
        """
        
        # email - type type
        assert type(email) is str, "ERRO: O parâmetro 'email' deve ser do tipo 'str'."
        # taxid - assert type
        assert type(taxid) is str, "ERRO: O parâmetro 'taxid' deve ser do tipo 'str'."
        assert taxid.isdigit(), "ERRO: Todos os caracteres de 'taxid' devem ser dígitos."
        # terms - assert type (including inner)
        msg_terms_1 = "ERRO: O parâmetro 'terms' deve ser do tipo 'list' ou 'tuple'."
        assert type(terms) is list or type(terms) is tuple, msg_terms_1
        msg_terms_2 = "ERRO: Todos os termos contidos em 'terms' devem ser do tipo 'str'."
        assert all(type(term) is str for term in terms), msg_terms_2
        # num_ids - assert type and value
        assert type(num_ids) is int, "ERRO: O parâmetro 'num_ids' deve ser do tipo 'int'."
        msg_num_ids = "ERRO: O valor do parâmetro 'num_ids' deve estar contido em [1, 20000]."
        assert num_ids > 0 and num_ids <= 20000, msg_num_ids
        # negatives - assert type and value
        assert type(negatives) is int, "ERRO: O parâmetro 'negatives' deve ser do tipo 'int'."
        assert negatives in [0,1], "ERRO: O parâmetro 'negatives' apenas toma os valores 0 e 1."
        
        self.email = email
        self.taxid = taxid
        self.terms = terms
        self.num_ids = num_ids
        self.negatives = negatives
        
    def __get_ids(self) -> list:
        """
        Retorna uma lista de IDs tendo por base os parâmetros introduzidos pelo utilizador.
        """
        search = f"txid{self.taxid}[ORGN] AND {'NOT'*self.negatives}({' OR '.join(self.terms)})"
        print(f"NCBI search: {search}")
        Entrez.email = self.email
        # idtype: by default, ESearch returns GI numbers in its output.
        # retmax: total number of UIDs from the retrieved set to be shown in the XML output.
        handle = Entrez.esearch(db="nuccore", term=search, retmax=self.num_ids)
        record = Entrez.read(handle)
        # Lista de IDs encontrados para a expressão "search"
        idlist = record["IdList"]
        handle.close()
        return idlist
    
    def __get_fname(self):
        """
        Retorna o nome do ficheiro fasta onde as sequências de DNA serão guardadas.
        """
        descrip = min(self.terms, key=len)
        if any(descrip not in term for term in self.terms): descrip = "protein"
        fname = f"txid{self.taxid}{'_negative'*self.negatives}_{descrip}s_{self.num_ids}"
        if os.path.exists(f"{fname}.fasta"):
            i = 1
            while os.path.exists(f"{fname}({i}).fasta"): i += 1
            fname = f"{fname}({i})"
        return fname
    
    def __filter_fasta(self, fname: str) -> tuple:
        """
        Cria um novo ficheiro fasta contendo sequências não repetidas. Retorna um tuplo contendo o nome
        do ficheiro criado, o número de sequências contidas no ficheiro original, o número de sequências
        obtidas após filtragem do mesmo, e um dicionário de contagens de produtos.
        
        Parameters
        ----------
        :param file: O nome do ficheiro fasta que se pretende processar
        """
        # Criação de um novo ficheiro fasta (guardar sequências não repetidas)
        if "(" not in fname: new_fname = f"{fname}_filt"
        else:
            pref, suff = fname.split("(")
            new_fname = f"{pref}_filt({suff.split(')')[0]})"
        file = open(f"{new_fname}.fasta", "w")
        # Procura de sequências repetidas no ficheiro original e construção de dicionário de contagens
        records = SeqIO.parse(f"{fname}.fasta", format="fasta")
        nf, f, filt, products_dict = 0, 0, set(), {}
        for record in records:
            nf += 1
            description, seq = record.description, record.seq
            if seq not in filt:
                f += 1
                product = description.split(" | ")[2].strip()
                # Atualizar dicionário de contagens e adicionar sequência ao ficheiro fasta
                if product not in products_dict: products_dict[product] = 1
                else: products_dict[product] += 1
                filt.add(seq)
                file.write(f">{description}\n{seq}\n\n")
        file.close()
        ord_dict = {k: v for k, v in sorted(products_dict.items(), key = lambda x: -x[1])}
        return new_fname, nf, f, ord_dict
    
    def get_sequences(self) -> None:
        """
        Cria um ficheiro fasta contendo sequências relacionadas aos termos de pesquisa introduzidos pelo
        utilizador. Com o auxílio do método __filter_fasta(), cria outro ficheiro sem sequências repetidas.
        """
        # Recolher IDs no NCBI
        idlist = self.__get_ids()
        # Abrir ficheiro fasta para guardar sequências
        fname = self.__get_fname()
        file = open(f"{fname}.fasta", "w")
        # Inspecionar todos os IDs recolhidos anteriormente
        for k in tqdm(idlist):
            Entrez.email = self.email
            handle = Entrez.efetch(db="nuccore", id=k, rettype="gb", retmode="text")
            record = SeqIO.read(handle, format="gb")
            # Verificar se a lista record.annotations["taxonomy"] não é vazia
            if record.annotations["taxonomy"]:
                # Verificar se o organismo pertence ao taxon "Viruses"
                if record.annotations["taxonomy"][0] == "Viruses":
                    for feature in record.features:
                        # Verificar se a feature corresponde a uma região codificante
                        if feature.type == "CDS":
                            # Verifcar se "product" é uma das keys do dicionário "feature.qualifiers"
                            if "product" in feature.qualifiers:
                                product = feature.qualifiers["product"][0]
                                # Ignorar features: 
                                # 1. Caso se pretendam sequências positivas, e nenhum dos termos introduzidos
                                # pelo utilizador se encontre em "product" ou caso o termo "not" se encontre 
                                # no mesmo
                                # 2. Caso se pretendam sequências negativas, e algum dos termos introduzidos
                                # pelo utilizador se encontre em "product" ou caso a descrição de "product"
                                # seja ambígua (p.e., "hypothetical protein")
                                if not self.negatives:
                                    if all(term not in product for term in self.terms): continue
                                    if "not" in product: continue
                                else:
                                    terms = self.terms + ["hypothetical protein", "Phage protein", "unknown"]
                                    if any(term in product for term in terms): continue
                                # Adicionar sequência de DNA ao ficheiro fasta (feature não ignorada)
                                fr, to = feature.location.start, feature.location.end
                                seq = record.seq[fr:to]
                                if feature.location.strand == -1: # caso a sequência esteja na strand -1
                                    seq_bio = Seq.Seq(seq)
                                    rev_comp = seq_bio.reverse_complement()
                                    seq = rev_comp
                                file.write(f"> {record.id} | {record.annotations['source']} | {product}\n"
                                           f"{seq}\n\n")
            handle.close()
        file.close()
        # Filtrar ficheiro fasta de modo a gerar novo ficheiro sem sequências repetidas
        new_fname, nf, f, products_dict = self.__filter_fasta(fname)
        # Prints - número de sequências recolhidas e dicionário de contagens
        print(f"Filtering removed {nf - f} sequences ({f} sequences remaining in '{new_fname}.fasta')")
        len_dict = len(products_dict)
        if len_dict > 20: products_dict = {k: v for k, v in list(products_dict.items())[:20]}
        print(f"Counts (after filtering): {products_dict}{' ...' if len_dict > 20 else ''}")
