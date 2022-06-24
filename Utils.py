# -*- coding: utf-8 -*-

import re

class Utils:
    
    """
    Implementa métodos cuja necessidade é transversal às classes SeqMining, MLDataSet e MLModel.
    """
    
    def __init__(self, email: str = None, txid_file: str = None):
        
        """
        Inicializa uma instância da classe Utils.
        
        Parameters
        ----------
        :param email: O endereço de email que se pretende validar
        :param txid_file: O nome de ficheiro que se pretende validar
        """
        
        if email: self.email = Utils.__verify_email(email)
        else: self.email = False
        
        if txid_file: self.txid_file = Utils.__verify_file_name(txid_file)
        else: txid_file: self.txid_file = False
    
    @staticmethod
    def __verify_email(email: str) -> bool:
        """
        Verifica se o endereço de email introduzido pelo utilizador é ou não válido.
        
        Parameters
        ----------
        :param email: O endereço de email que se pretende validar
        """
        valid = "^[a-zA-Z0-9]+[-._]?[a-zA-Z0-9]+[@][a-zA-Z0-9]+[.]?[a-zA-Z0-9]+[.]\w{2,3}$"
        verify = re.search(valid, email)
        return True if verify else False
    
    @staticmethod
    def __verify_file_name(txid_file: str) -> bool:
        """
        Verifca se o nome do ficheiro 'txid_file' se inicia com a expressão 'txid<num>_'.
        
        Parameters
        ----------
        :param txid_file: O nome de ficheiro que se pretende validar
        """
        valid = "^txid[0-9]+_"
        verify = re.search(valid, txid_file)
        return True if verify else False
