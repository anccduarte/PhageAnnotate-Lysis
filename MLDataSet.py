# -*- coding: utf-8 -*-

from Bio import Entrez, SeqIO, Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from propy import PyPro
import os
import pandas as pd
import re
import warnings

# Ignorar warnings do Biopython (related to translation)
warnings.filterwarnings("ignore")

class MLDataSet:
    
    """
    Permite a construção de um dataset a partir de um conjunto de ficheiros fasta contendo sequências
    de DNA e de labels associadas aos mesmos.
    """
    
    def __init__(self, email: str, datasets: list) -> None:
        
        """
        Inicializa uma instância da classe MLDataSet.
        
        Parameters
        ----------
        :param email: O endereço de email do utilizador
        :param datasets: O conjunto de ficheiros fasta e de labels
        """
        
        # email - assert type and value
        msg_email_1 = "O parâmetro 'email' deve ser do tipo 'str'."
        if type(email) is not str: raise TypeError(msg_email_1)
        msg_email_2 = "O endereço de email inserido não é válido."
        if not MLDataSet.__verify_email(email): raise ValueError(msg_email_2)
        
        # datasets - assert types and values
        msg_datasets_1 = "O parâmetro 'datasets' deve ser do tipo 'list'."
        if type(datasets) is not list: raise TypeError(msg_datasets_1)
        msg_datasets_2 = "Todos os elementos contidos em 'datasets' devem ser do tipo 'tuple'."
        if any(type(ds) is not tuple for ds in datasets): raise TypeError(msg_datasets_2)
        msg_datasets_3 = "Todos os elementos contidos em 'datasets' devem ter dimensão igual a 2."
        if any(len(ds) != 2 for ds in datasets): raise TypeError(msg_datasets_3)
        msg_datasets_4 = "Os elementos presentes em cada dataset (file, label) devem ser to tipo 'str'."
        if any(type(el) is not str for ds in datasets for el in ds): raise TypeError(msg_datasets_4)
        msg_datasets_5 = "Os elementos presentes em cada dataset (file, label) devem ter dimensão superior a 0."
        if any(len(el.strip()) < 1 for ds in datasets for el in ds): raise ValueError(msg_datasets_5)
        
        # assert existance of file in cwd
        for file,_ in datasets:
            if not os.path.exists(file):
                msg_datasets_6 = f"O ficheiro '{file}' não foi encontrado em '{os.getcwd()}'."
                raise FileNotFoundError(msg_datasets_6)
        
        # instance attributes
        self.email = email
        self.datasets = datasets
        self.ml_dataset = self.__build_dataset()
        
    @staticmethod
    def __verify_email(email: str) -> str:
        """
        Verifica se o endereço de email introduzido pelo utilizador é ou não válido.
        
        Parameters
        ----------
        :param email: O endereço de email a validar
        """
        valid = "^[a-zA-Z0-9]+[-._]?[a-zA-Z0-9]+[@][a-zA-Z0-9]+[.]?[a-zA-Z0-9]+[.]\w{2,3}$"
        verify = re.search(valid, email)
        return True if verify else False
        
    @staticmethod
    def __nuc_composition(seq: Seq.Seq) -> list:
        """
        Retorna a frequência relativa de cada nucleótido numa sequência de DNA.

        Parameters
        ----------
        :param seq: A sequência de DNA
        """
        comp = [round(seq.count(nuc)/len(seq), 4) for nuc in "ACGT"]
        return comp
    
    def __get_translation_table(self, file: str) -> str:
        """
        Retorna o número da tabela de tradução dado o identificador presente no nome do ficheiro fasta 
        que toma como parâmetro.
        
        Parameters
        ----------
        :param file: O nome do ficheiro fasta
        """
        taxid = file.split("_")[0]
        Entrez.email = self.email
        with Entrez.efetch(db='taxonomy', id=taxid, retmode='xml') as handle:
            record = Entrez.read(handle)
        return record[0]["GeneticCode"]["GCId"]
    
    @staticmethod
    def __translate_seq(seq: Seq.Seq, translation_table: str) -> Seq.Seq:
        """
        Retorna uma sequência de aminoácidos obtida através da tradução da sequência de DNA que toma
        como parâmetro.
        
        Parameters
        :param seq: A sequência de DNA
        :param translation_table: A tabela de tradução utilizada
        """
        while len(seq) % 3 != 0: seq = f"N{seq}"
        tseq = Seq.Seq(seq).translate(table=translation_table)[:-1]
        return tseq

    @staticmethod
    def __amino_composition(seq: Seq.Seq) -> list:
        """
        Retorna a frequência relativa de cada aminoácido numa sequência de aminoácidos.

        Parameters
        ----------
        :param seq: A sequência de aminoácidos
        """
        comp = [round(seq.count(amino)/len(seq), 4) for amino in "ACDEFGHIKLMNPQRSTVWY"]
        return comp
    
    @staticmethod
    def __aromaticity(seq: Seq.Seq) -> list:
        """
        Retorna o valor da aromaticidade da sequência (somatório das frequências relativas dos aminoácidos
        fenilalanina (F), tirosina (Y) e triptofano(W)).
        
        Parameters
        ----------
        :param seq: A sequência de aminoácidos
        """
        arom = sum([(seq.count(amino)/len(seq)) for amino in "FYW"])
        return [round(arom, 4)]
    
    @staticmethod
    def __isoelectric_point(seq: Seq.Seq) -> list:
        """
        Retorna o ponto isoelétrico da sequência de aminoácidos.
        
        Parameters
        ----------
        :param seq: A sequência de aminoácidos
        """
        X = ProteinAnalysis(str(seq))
        return [round(X.isoelectric_point(), 4)]
    
    @staticmethod
    def __secondary_structure_fraction(seq: Seq.Seq) -> list:
        """
        Retorna a fração de aminoácidos que tende a ser encontrada em três estruturas secundárias:
        alpha_helix - Val (V), Ile (I), Tyr (Y), Phe (F), Trp (W), Leu (L)
        beta_turn - Asn (N), Pro (P), Gly (G), Ser (S)
        beta_sheet - Glu (E), Met (M), Ala (A), Leu (L)
        
        Parameters
        ----------
        :param seq: A sequência de aminoácidos
        """
        helix = round(sum([(seq.count(amino)/len(seq)) for amino in "VIYFWL"]), 4)
        turn = round(sum([(seq.count(amino)/len(seq)) for amino in "NPGS"]), 4)
        sheet = round(sum([(seq.count(amino)/len(seq)) for amino in "EMAL"]), 4)
        ssf = [helix] + [turn] + [sheet]
        return ssf
    
    @staticmethod
    def __ctd_descriptors(seq: Seq.Seq) -> list:
        """
        Retorna os descritores CTD (composition, transition, distribution) da sequência de aminoácidos.
        
        Parameters
        ----------
        :param seq: A sequência de aminoácidos
        """
        DesObject = PyPro.GetProDes(seq)
        return [feat for feat in DesObject.GetCTD().values()]
    
    """
    @staticmethod
    def __dipeptide_composition(seq: Seq.Seq) -> list:
        aminoacids = "ACDEFGHIKLMNPQRSTVWY"
        dipeptides = [aa1 + aa2 for aa1 in aminoacids for aa2 in aminoacids]
        dcs = [round(len(re.findall(dp, str(seq)))/(len(seq)-1), 6) for dp in dipeptides]
        return dcs
    """
    
    @staticmethod
    def __dipeptide_composition(seq: Seq.Seq) -> list:
        """
        Retorna a composição dipeptídica da sequência de aminoácidos.
        
        Parameters
        ----------
        :param seq: A sequência de aminoácidos
        """
        DesObject = PyPro.GetProDes(seq)
        return [feat for feat in DesObject.GetDPComp().values()]
    
    def __get_fname(self) -> str:
        """
        Retorna o nome do ficheiro csv gerado a partir dos datasets de sequências de DNA.
        """
        fname = f"ml_dataset_{'_'.join([label[:3] for _,label in self.datasets])}"
        if os.path.exists(f"{fname}.csv"):
            i = 1
            while os.path.exists(f"{fname}({i}).csv"): i += 1
            fname = f"{fname}({i})"
        return fname

    def __build_dataset(self) -> pd.DataFrame:
        """
        Retorna um pd.DataFrame construído a partir dos tuplos (file, label) introduzidos pelo utilizador.
        Feature mapping: nucleotide composition (4 features), aminoacid composition (20 features), 
        aromaticity (1 feature), isoelectric point (1 feature), secondary structure fraction (3 features),
        ctd descriptors (147 features), dipeptide composition (400 features), sequence length (1 feature)
        Total number of features: 577
        """
        # Definição do nome das colunas do dataset
        ctd = [f"CTD{i+1}" for i in range(147)]
        dc = [f"DC{i+1}" for i in range(400)]
        col_names = ["Len_Prot", "A_Nuc", "C_Nuc", "G_Nuc", "T_Nuc", "A_Amino", "C_Amino", "D_Amino", 
                     "E_Amino", "F_Amino", "G_Amino", "H_Amino", "I_Amino", "K_Amino", "L_Amino", "M_Amino",
                     "N_Amino", "P_Amino", "Q_Amino", "R_Amino", "S_Amino", "T_Amino", "V_Amino", "W_Amino",
                     "Y_Amino", "Aromat", "Isoelec", "A_Helix", "B_Turn", "B_Sheet"] + ctd + dc + ["Label"]
        data_out = pd.DataFrame(columns=col_names)
        # Construção do dataset
        idx = 0
        for dataset in self.datasets:
            file, label = dataset
            translation_table = self.__get_translation_table(file)
            records = SeqIO.parse(file, format="fasta")
            for rec in records:
                tseq = MLDataSet.__translate_seq(rec.seq, translation_table)
                row = [len(tseq)]
                row += MLDataSet.__nuc_composition(rec.seq)
                row += MLDataSet.__amino_composition(tseq)
                row += MLDataSet.__aromaticity(tseq)
                row += MLDataSet.__isoelectric_point(tseq)
                row += MLDataSet.__secondary_structure_fraction(tseq)
                row += MLDataSet.__ctd_descriptors(tseq)
                row += MLDataSet.__dipeptide_composition(tseq)
                row += [label]
                data_out.loc[idx] = row
                idx += 1
        data_out.to_csv(f"{self.__get_fname()}.csv")
        return data_out
        