# -*- coding: utf-8 -*-

from Bio import SeqIO
from MLDataSet import MLDataSet
from sklearn.ensemble import RandomForestClassifier
#from sklearn.feature_selection import SelectFromModel # RFECV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
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
        :param model: O modelo pretendido ({"RF", "SVC", "ANN"})
        :param dataset: Um dataset contendo features e labels relativas a sequências de DNA
        """
        
        # model - assert type and value
        msg_model_1 = "O parâmetro 'model' deve ser do tipo 'str'."
        if type(model) is not str: raise TypeError(msg_model_1)
        msg_model_2 = "O parâmetro 'model' apenas toma os valores 'RF', 'SVC' ou 'ANN'."
        if model not in ["RF", "SVC", "ANN"]: raise ValueError(msg_model_2)
        
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
    
    def __call__(self) -> Union[RandomForestClassifier, SVC, MLPClassifier]:
        """
        Retorna um modelo de machine learning (especificado pelo utilizador) aquando da criação de uma
        instância de da classe MLModel. 
        >>> model = MLModel('model', 'dataset')
        >>> model.predict_proteins('email', 'file')
        """
        return self.fitted_estimator
        
    def __get_train_test(self) -> tuple:
        """
        Executa o split em dados de treino e de teste.
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
    
    """
    def __get_feature_mask(self) -> np.array:
        #Retorna uma máscara booleana indicando os índices das features a manter no modelo.
        x_train, x_test, y_train, y_test = self.ml_data
        selector = SelectFromModel(estimator=DecisionTreeClassifier()) # RFECV(estimator)
        selector.fit(x_train, y_train)
        return selector.get_support() # selector.support_ if RFECV is used
    """
    
    def __optimize_hyperparameters(self) -> dict:
        """
        Optimiza os hiperparâmetros do modelo de machine learning.
        
        Parameters
        ----------
        :param modelo: O modelo especificado pelo utilizador
        """
        # data
        x_train, x_test, y_train, y_test = self.ml_data
        # escolha dos hiperparâmetros a otimizar dependendo do modelo a utilizar
        if self.model == "RF":
            param_grid = {"max_features": ["sqrt", "log2"], "n_estimators": [50, 100, 150, 200],
                          "min_samples_split": [1, 2, 4], "min_samples_leaf": [1, 2, 4]}
            estimator = RandomForestClassifier()
        elif self.model == "SVC":
            param_grid = {"C": [0.1, 1, 10, 20], "kernel": ["linear", "poly", "rbf", "sigmoid"],
                          "gamma": ["auto", "scale"]}
            estimator = SVC()
        else:
            param_grid = {"hidden_layer_sizes": [(10,), (25,), (50,), (100,)], "solver": ["adam", "sgd"],
                          "activation": ["tanh", "relu"], "alpha": [0.0001, 0.001, 0.01]}
            estimator = MLPClassifier()
        # determinação dos hiperparâmetros ótimos (5-fold cross validation (cv) -> default)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
        grid.fit(x_train, y_train)
        return grid.best_params_
    
    def __get_estimator(self) -> Union[RandomForestClassifier, SVC, MLPClassifier]:
        """
        Retorna um objeto 'estimator' de acordo com o modelo de machine learning especificado pelo 
        utilizador.
        """
        # obter hiperparâmetros ótimos
        params = self.__optimize_hyperparameters()
        # obter o modelo de machine learning
        if self.model == "RF": estimator = RandomForestClassifier(**params)
        elif self.model == "SVC": estimator = SVC(**params)
        else: estimator = MLPClassifier(**params)
        return estimator
    
    def __build_model(self) -> None:
        """
        Retorna um modelo de machine learning construído a partir do dataset de features e labels fornecido
        pelo utilizador.
        """
        # data from ml_dataset
        x_train, x_test, y_train, y_test = self.ml_data
        """
        # obter feature mask para a redução do número de features
        self.mask = self.__get_feature_mask() # também é utilizado em self.predict_proteins()
        # redifinir self.ml_data de modo a otimizar os hiperparâmetros com o novo conjunto de features
        x_train, x_test = x_train[:,self.mask], x_test[:,self.mask]
        self.ml_data = x_train, x_test, y_train, y_test
        """
        # get estimator with optimized hyperparameters and fit the model
        estimator = self.__get_estimator()
        estimator.fit(x_train, y_train)
        # métricas (accuracy, precision, recall) - dados de teste
        y_pred = estimator.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        print(f"METRICS ON TESTING DATA (USING '{self.model}')\nAccuracy: {accuracy*100:.2f}% | "
              f"Precision: {precision*100:.2f}% | Recall: {recall*100:.2f}%")
        # returns a fitted estimator (RF, SVC ou ANN)
        return estimator
    
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
        msg_file_2 = "O valor do parâmetro 'file' deve ter dimensão superior a 4."
        if len(file.strip()) < 5: raise ValueError(msg_file_2)
        
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
        y_pred_new = self.fitted_estimator.predict(pred_data_scaled) # pred_data_scaled[:,self.mask]
        descrips = [rec.description for rec in SeqIO.parse(file, format="fasta")]
        predictions = pd.DataFrame({"descriptions": descrips, "prediction": y_pred_new})
        predictions.to_csv(f"{self.__get_fname()}.csv")
        