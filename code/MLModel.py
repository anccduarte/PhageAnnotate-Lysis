# -*- coding: utf-8 -*-

from Bio import SeqIO
from MLDataSet import MLDataSet
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from typing import Union
from Utils import Utils
import numpy as np
import os
import pandas as pd

class MLModel:
    
    """
    Implementa algoritmos de machine learning (Random Forests e Support Vector Machines) para a previsão 
    de função de proteínas.
    """
    
    def __init__(self, model: str, dataset: str) -> None:
        
        """
        Inicializa uma instância da classe MLModel.
        
        Parameters
        ----------
        :param model: O modelo pretendido ({"RF", "SVM"})
        :param dataset: Um dataset contendo features e labels relativas a sequências de DNA
        """
        
        # model - assert type and value
        msg_model_1 = "O parâmetro 'model' deve ser do tipo 'str'."
        if type(model) is not str: raise TypeError(msg_model_1)
        msg_model_2 = "O parâmetro 'model' apenas toma os valores {'RF', 'SVM'}."
        if model not in ["RF", "SVM"]: raise ValueError(msg_model_2)
        
        # dataset - assert type and value
        msg_dataset_1 = "O parâmetro 'dataset' deve ser do tipo 'str'."
        if type(dataset) is not str: raise TypeError(msg_dataset_1)
        msg_dataset_2 = "O valor do parâmetro 'dataset' deve ter dimensão superior a 4."
        if len(dataset.strip()) < 5: raise ValueError(msg_dataset_2)
        
        # assert existance of dataset in cwd
        if not os.path.exists(dataset):
            msg_dataset_3 = f"O ficheiro '{dataset}' não foi encontrado em '{os.getcwd()}'."
            raise FileNotFoundError(msg_dataset_3)
        
        # instance attributes
        self.model = model
        self.__names = {"RF": "Random Forest", "SVM": "Support Vector Machine"}
        self.__dataset = pd.read_csv(dataset)
        self.__ml_data = self.__scale_data()
        self.__fitted_estimator = self.__build_model()
    
    def __call__(self) -> Union[RFC, SVC]:
        """
        Retorna um modelo de machine learning (especificado pelo utilizador) aquando da criação de uma
        instância de da classe MLModel. 
        >>> model = MLModel('model', 'dataset')
        >>> model.predict_proteins('email', 'file')
        """
        return self.__fitted_estimator
        
    def __get_train_test(self) -> tuple:
        """
        Executa a divisão em dados de treino (80%) e de teste (20%).
        """
        # separação de features e labels
        df_feats = self.__dataset.iloc[:, 1:-1]
        df_labels = self.__dataset.iloc[:, -1]
        # separação em dados de teste e treino (shuffle ???)
        x_trn, x_tst, y_trn, y_tst = train_test_split(df_feats, df_labels, train_size=0.8, random_state=42)
        # retorna 4 numpy arrays (2 arrays de features e 2 arrays de labels)
        return x_trn, x_tst, y_trn, y_tst
    
    def __scale_data(self) -> tuple:
        """
        Retorna os dados de treino e teste normalizados ((x(i,j) - mean_train(j)) / stdv_train(j)).
        """
        # dados de treino e de teste
        x_train, x_test, y_train, y_test = self.__get_train_test()
        # criação de um objeto 'self.scaler' -> também será utilizado nos novos dados
        self.__scaler = StandardScaler().fit(x_train)
        # normalização das features de treino e de teste
        x_train_scaled = self.__scaler.transform(x_train)
        x_test_scaled = self.__scaler.transform(x_test)
        # retorna 4 numpy arrays (2 arrays de features e 2 arrays de labels)
        return x_train_scaled, x_test_scaled, np.ravel(y_train), np.ravel(y_test)
    
    def __optimize_hyperparameters(self, mode: str) -> dict: # <--------------------
        """
        Optimiza os hiperparâmetros do modelo de machine learning.
        
        Parameters
        ----------
        :param mode: Especifica o grau de robustez que se pretende na otimização dos hiperparâmetros
        """
        # data
        x_train, x_test, y_train, y_test = self.__ml_data
        # seleção da param_grid dependendo do grau de robustez que se pretende na otimização
        dfs, be = "decision_function_shape", "base_estimator__"
        hyper_l = {"RF": {"n_estimators": [50, 150], "max_features": ["sqrt", "log2"], 
                          "min_samples_split": [1, 2], "min_samples_leaf": [1, 2]},
                   "SVM": {"C": [0.01, 0.1, 1], "kernel": ["linear"]}, dfs: ["ovo", "ovr"]}
        hyper_r = {"RF": {"criterion": ["gini", "entropy"], "n_estimators": [50, 100, 150, 200], 
                          "min_samples_split": [1, 2, 4], "min_samples_leaf": [1, 2, 4],
                          "max_features": ["sqrt", "log2"], "bootstrap": [True, False]},
                   "SVM": {"C": [0.01, 0.1, 1, 10], "kernel": ["linear", "poly", "rbf", "sigmoid"], 
                           "degree": [3, 4, 5], "gamma": ["auto", "scale"], dfs: ["ovo", "ovr"]}}
        if mode == "light": param_grid = hyper_l[self.model]
        elif mode == "robust": param_grid = hyper_r[self.model]
        # seleção do 'estimator' para a otimização dos hiperparâmetros
        if self.model == "RF": estimator = RFC(random_state=42)
        elif self.model == "SVM": estimator = SVC()
        # determinação dos hiperparâmetros ótimos (5-fold cross validation (cv) -> default)
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid)
        grid.fit(x_train, y_train)
        if mode == "robust": print(f"{self.model} hyperparameters: {grid.best_params_}")
        return grid.best_params_
    
    def __get_estimator(self, mode: str) -> Union[RFC, SVC]:
        """
        Retorna um objeto 'estimator' de acordo com o modelo de machine learning especificado pelo 
        utilizador.
        
        Parameters
        ----------
        :param mode: Especifica o grau de robustez que se pretende na otimização dos hiperparâmetros
        """
        # obter hiperparâmetros ótimos
        params = self.__optimize_hyperparameters(mode)
        # obter o modelo de machine learning
        if self.model == "RF": estimator = RFC(**params, random_state=42)
        # SVC -> # random_state ignored when 'probability' is False
        elif self.model == "SVM": estimator = SVC(**params)
        return estimator
    
    def __get_feature_mask(self, estimator: Union[RFC, SVC]) -> np.array:
        """
        Retorna uma máscara booleana indicando os índices das features a manter no modelo.
        
        Parameters
        ----------
        :param estimator: Um objeto 'estimator' ∈ {RFC, SVC}
        """
        x_train, x_test, y_train, y_test = self.__ml_data
        selector = SelectFromModel(estimator=estimator)
        selector.fit(x_train, y_train)
        feature_mask = selector.get_support()
        print(f"Number of selected features: {len(feature_mask[feature_mask == True])}")
        return feature_mask
    
    def __build_model(self) -> None:
        """
        Retorna um modelo de machine learning construído a partir do dataset de features e labels fornecido
        pelo utilizador.
        """
        print(f"Building {self.__names[self.model]} ({self.model}) model...")
        # data from ml_dataset
        x_train, x_test, y_train, y_test = self.__ml_data
        # get first estimator with lightly optimized parameters
        estimator1 = self.__get_estimator("light")
        # obter feature mask para a redução do número de features
        self.__mask = self.__get_feature_mask(estimator1) # também é utilizada em self.predict_proteins()
        # redifinir self.ml_data de modo a otimizar os hiperparâmetros com o novo conjunto de features
        x_train, x_test = x_train[:,self.__mask], x_test[:,self.__mask]
        self.__ml_data = x_train, x_test, y_train, y_test
        # get estimator with optimized hyperparameters and fit the model
        estimator2 = self.__get_estimator("robust")
        estimator2.fit(x_train, y_train)
        # confusion matrix e métricas (accuracy, precision, recall) - dados de teste
        y_pred = estimator2.predict(x_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.precision = precision_score(y_test, y_pred, average="macro")
        self.recall = recall_score(y_test, y_pred, average="macro")
        self.conf_mat = confusion_matrix(y_test, y_pred, labels=["spanin", "endolysin", "holin", "other"])
        print(f"Metrics on testing data: accuracy_score = {self.accuracy*100:.2f}% | "
              f"precision_score = {self.precision*100:.2f}% | recall_score = {self.recall*100:.2f}%")
        # returns a fitted estimator (RF ou SVM)
        return estimator2
    
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
        msg_email_2 = "O endereço de email introduzido não é válido."
        if not Utils(email=email).email: raise ValueError(msg_email_2)
        
        # file assert type and value
        msg_file_1 = "O parâmetro 'file' deve ser do tipo 'str'."
        if type(file) is not str: raise TypeError(msg_file_1)
        msg_file_2 = "O nome do ficheiro 'file' deve começar com a expressão 'txid<num>_'."
        if not Utils(txid_file=file).txid_file: raise ValueError(msg_file_2)
        
        # assert existance of file in cwd
        if not os.path.exists(file):
            msg_file_3 = f"O ficheiro '{file}' não foi encontrado em '{os.getcwd()}'."
            raise FileNotFoundError(msg_file_3)
        
        # determinar features do novo conjunto de dados
        data = MLDataSet(email, [(file, "unknown")])
        os.remove("ml_dataset_unk.csv")
        to_predict = data.ml_dataset.iloc[:, :-1]
        # normalizar os dados novos (a partir da média e stdv dos dados de treino)
        to_predict_scaled = self.__scaler.transform(to_predict)
        # anotar sequências e exportar resultados para um ficheiro csv
        y_pred = self.__fitted_estimator.predict(to_predict_scaled[:,self.__mask])
        descrips = [rec.description for rec in SeqIO.parse(file, format="fasta")]
        predictions = pd.DataFrame({"descriptions": descrips, "prediction": y_pred})
        predictions.to_csv(f"{self.__get_fname()}.csv")
        