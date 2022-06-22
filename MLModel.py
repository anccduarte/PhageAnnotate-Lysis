# -*- coding: utf-8 -*-

from Bio import SeqIO
from MLDataSet import MLDataSet
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel # RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import Union
import numpy as np
import os
import pandas as pd
import re
import warnings

# Ignorar warnings do Biopython (related to translation)
# Ignorar warnings do sklearn (related to MLPClassifier optimization convergence)
warnings.filterwarnings("ignore")

class MLModel:
    
    """
    Implementa algoritmos de machine learning (Random Forest, Support Vector Machine, e Artificial Neural
    Network) para a previsão de função de proteínas.
    """
    
    def __init__(self, model: str, dataset: str) -> None:
        
        """
        Inicializa uma instância da classe MLModel.
        
        Parameters
        ----------
        :param model: O modelo pretendido ({"RF", "SVM", "ANN"})
        :param dataset: Um dataset contendo features e labels relativas a sequências de DNA
        """
        
        # model - assert type and value
        msg_model_1 = "O parâmetro 'model' deve ser do tipo 'str'."
        if type(model) is not str: raise TypeError(msg_model_1)
        msg_model_2 = "O parâmetro 'model' apenas toma os valores {'RF', 'SVM', 'ANN'}."
        if model not in ["RF", "SVM", "ANN"]: raise ValueError(msg_model_2)
        
        # dataset - assert type and value
        msg1 = "O parâmetro 'dataset' deve ser do tipo 'str'."
        if type(dataset) is not str: raise TypeError(msg1)
        msg2 = "O valor do parâmetro 'dataset' deve ter dimensão superior a 4."
        if len(dataset.strip()) < 5: raise ValueError(msg2)
        
        # assert existance of dataset in cwd
        if not os.path.exists(dataset):
            msg3 = f"O ficheiro '{dataset}' não foi encontrado em '{os.getcwd()}'."
            raise FileNotFoundError(msg3)
        
        # instance attributes
        self.model = model
        self.dataset = pd.read_csv(dataset)
        self.ml_data = self.__scale_data()
        self.fitted_estimator = self.__build_model()
    
    def __call__(self) -> Union[RFC, SVC, MLPC]:
        """
        Retorna um modelo de machine learning (especificado pelo utilizador) aquando da criação de uma
        instância de da classe MLModel. 
        >>> model = MLModel('model', 'dataset')
        >>> model.predict_proteins('email', 'file')
        """
        return self.fitted_estimator
        
    def __get_train_test(self) -> tuple:
        """
        Executa a divisão em dados de treino (80%) e de teste (20%).
        """
        # separação de features e labels
        df_feats = self.dataset.iloc[:, 1:-1]
        df_labels = self.dataset.iloc[:, -1]
        # separação dos datasets em treino e teste (random_state=42 ???)
        x_train, x_test, y_train, y_test = train_test_split(df_feats, df_labels, train_size=0.8) 
        # retorna 4 numpy arrays (2 arrays de features e 2 arrays de labels)
        return x_train, x_test, y_train, y_test
    
    def __scale_data(self) -> tuple:
        """
        Retorna os dados de treino e teste normalizados ((x(i,j) - mean(j)) / std(j)).
        """
        # dados de treino e de teste
        x_train, x_test, y_train, y_test = self.__get_train_test()
        # criação de um objeto 'self.scaler' -> também será utilizado nos novos dados
        self.scaler = StandardScaler().fit(x_train)
        # normalização das features de treino e de teste
        x_train_scaled = self.scaler.transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        # retorna 4 numpy arrays (2 arrays de features e 2 arrays de labels)
        return x_train_scaled, x_test_scaled, np.ravel(y_train), np.ravel(y_test)
    
    def __optimize_hyperparameters(self, mode: str) -> dict:
        """
        Optimiza os hiperparâmetros do modelo de machine learning.
        
        Parameters
        ----------
        :param mode: Especifica o grau da robustez que se pretende da otimização dis hiperparâmetros
        """
        # data
        x_train, x_test, y_train, y_test = self.ml_data
        # seleção da param_grid dependendo do grau de robustez que se pretende na otimização
        hyper_l = {"RF": {"max_features": ["sqrt", "log2"], "n_estimators": [50, 100, 200]},
                   "SVM": {"C": [0.1, 1], "kernel": ["linear", "poly"], "gamma": ["auto", "scale"]},
                   "ANN": {"hidden_layer_sizes": [(10,), (50,)], "alpha": [0.0001, 0.001, 0.01]}}
        hyper_r = {"RF": {"max_features": ["sqrt", "log2"], "n_estimators": [50, 100, 150, 200],
                          "min_samples_split": [1, 2, 4], "min_samples_leaf": [1, 2, 4]},
                   "SVM": {"C": [0.1, 1, 10, 20], "kernel": ["linear", "poly", "rbf", "sigmoid"],
                           "degree": [3, 4], "gamma": ["auto", "scale"]},
                   "ANN": {"hidden_layer_sizes": [(10,), (25,), (50,), (80,)], "solver": ["adam", "sgd"],
                           "activation": ["tanh", "relu"], "alpha": [0.0001, 0.001, 0.01]}}
        if mode == "light": param_grid = hyper_l[self.model]
        else: param_grid = hyper_r[self.model]
        # seleção do 'estimator' para a otimização dos hiperparâmetros
        if self.model == "RF": estimator = RFC()
        elif self.model == "SVM": estimator = SVC()
        else: estimator = MLPC()
        # determinação dos hiperparâmetros ótimos (5-fold cross validation (cv) -> default)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
        grid.fit(x_train, y_train)
        if mode == "robust": print(f"{self.model}'s hyperparameters: {grid.best_params_}")
        return grid.best_params_
    
    def __get_estimator(self, mode: str) -> Union[RFC, SVC, MLPC]:
        """
        Retorna um objeto 'estimator' de acordo com o modelo de machine learning especificado pelo 
        utilizador.
        
        Parameters
        ----------
        :param mode: Especifica o grau da robustez que se pretende da otimização dis hiperparâmetros
        """
        # obter hiperparâmetros ótimos
        if mode == "light": params = self.__optimize_hyperparameters("light")
        else: params = self.__optimize_hyperparameters("robust")
        # obter o modelo de machine learning
        if self.model == "RF": estimator = RFC(**params)
        elif self.model == "SVM": estimator = SVC(**params)
        else: estimator = MLPC(**params)
        return estimator
    
    def __get_feature_mask(self, estimator: Union[RFC, SVC, MLPC]) -> np.array:
        """
        Retorna uma máscara booleana indicando os índices das features a manter no modelo.
        
        Parameters
        ----------
        :param estimator: Um objeto 'estimator' ∈ {RFC, SVC, MLPC}
        """
        x_train, x_test, y_train, y_test = self.ml_data
        selector = SelectFromModel(estimator=estimator) # RFECV(estimator)
        selector.fit(x_train, y_train)
        return selector.get_support() # selector.support_ if RFECV is used
    
    def __build_model(self) -> None:
        """
        Retorna um modelo de machine learning construído a partir do dataset de features e labels fornecido
        pelo utilizador.
        """
        print(f"Building {self.model} model...")
        # data from ml_dataset
        x_train, x_test, y_train, y_test = self.ml_data
        # get first estimator with lightly optimized parameters
        estimator1 = self.__get_estimator("light")
        # obter feature mask para a redução do número de features
        self.mask = self.__get_feature_mask(estimator1) # também é utilizada em self.predict_proteins()
        # redifinir self.ml_data de modo a otimizar os hiperparâmetros com o novo conjunto de features
        x_train, x_test = x_train[:,self.mask], x_test[:,self.mask]
        self.ml_data = x_train, x_test, y_train, y_test
        # get estimator with optimized hyperparameters and fit the model
        estimator2 = self.__get_estimator("robust")
        estimator2.fit(x_train, y_train)
        # métricas (accuracy, precision, recall) - dados de teste
        y_pred = estimator2.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        print(f"METRICS ON TESTING DATA\n"
              f"Accuracy: {accuracy*100:.2f}% | Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}%")
        # returns a fitted estimator (RF, SVM ou ANN)
        return estimator2
    
    @staticmethod
    def __verify_email(email: str) -> bool:
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
    def __verify_file_name(file: str) -> bool:
        """
        Verifca se o nome do ficheiro que contém as sequências cuja função se pretende prever se inicia com 
        a expressão 'txid<num>_'.
        
        Parameters
        ----------
        :param file: O nome do ficheiro que contém as sequências
        """
        valid = "^txid[0-9]+_"
        verify = re.search(valid, file)
        return True if verify else False
    
    def __get_fname(self) -> str:
        """
        Retorna o nome do ficheiro csv que irá conter as previsões de função.
        """
        fname = f"{self.model.lower()}_predictions"
        if os.path.exists(f"{fname}.csv"):
            i = 1
            while os.path.exists(f"{fname}({i}).csv"): i += 1
            fname = f"{fname}({i})"
        return fname
    
    def predict_proteins(self, email: str, file: str) -> None:
        """
        Parameters
        ----------
        :param email: O endereço de email do utilizador
        :param file: Um ficheiro fasta contendo sequências de nucleótidos
        """
        # email - assert type and value
        msg_email_1 = "O parâmetro 'email' deve ser do tipo 'str'."
        if type(email) is not str: raise TypeError(msg_email_1)
        msg_email_2 = "O endereço de email inserido não é válido."
        if not MLModel.__verify_email(email): raise ValueError(msg_email_2)
        
        # file assert type and value
        msg_file_1 = "O parâmetro 'file' deve ser do tipo 'str'."
        if type(file) is not str: raise TypeError(msg_file_1)
        msg_file_2 = "O nome do ficheiro 'file' deve começar com a expressão 'txid<num>'."
        if not MLModel.__verify_file_name(file): raise ValueError(msg_file_2)
        
        # assert existance of file in cwd
        if not os.path.exists(file):
            msg_file_3 = f"O ficheiro '{file}' não foi encontrado em '{os.getcwd()}'."
            raise FileNotFoundError(msg_file_3)
        
        # determinar features do novo conjunto de dados
        data = MLDataSet(email, [(file, "unknown")])
        os.remove("ml_dataset_unk.csv")
        pred_data = data.ml_dataset.iloc[:, :-1]
        # normalizar os dados novos (a partir da média e stdv dos dados de treino)
        pred_data_scaled = self.scaler.transform(pred_data)
        # anotar sequências e exportar resultados para um ficheiro csv
        y_pred_new = self.fitted_estimator.predict(pred_data_scaled[:,self.mask])
        descrips = [rec.description for rec in SeqIO.parse(file, format="fasta")]
        predictions = pd.DataFrame({"descriptions": descrips, "prediction": y_pred_new})
        predictions.to_csv(f"{self.__get_fname()}.csv")
        