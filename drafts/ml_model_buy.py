import numpy as np
import pandas as pd
import os
import time
import joblib
import requests
import math
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor


class BaseAlgorithm:
    def __init__(self, model_class, weights_file, target_col, **model_kwargs):
        """
        Базовый класс для всех алгоритмов машинного обучения.

        :param model_class: Класс модели (LinearRegression, RandomForest, и т. д.)
        :param weights_file: Файл для сохранения/загрузки модели.
        :param target_col: Название столбца-цели.
        :param model_kwargs: Дополнительные параметры для модели.
        """
        directory = "models"
        self.ensure_directory_exists(directory)
        self.weights_file = f"{directory}/{weights_file}"
        self.target_col = target_col
        self.model = model_class(**model_kwargs)

    @staticmethod
    def ensure_directory_exists(directory_path="models"):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Директория '{directory_path}' создана.")
        else:
            print(f"Директория '{directory_path}' уже существует.")

    def train(self, X, y):
        """
        Обучает модель на предоставленных данных и сохраняет веса.

        :param data: Датафрейм с входными признаками и целевой переменной.
        """
        # df = data.copy()
        # X = df.drop(columns=[self.target_col])
        # y = df[self.target_col].values
        start_time = time.time()
        # Обучение модели
        self.model.fit(X, y)

        # Сохранение модели
        joblib.dump(self.model, self.weights_file)
        print(f"Модель сохранена в {self.weights_file}")
        model_title = self.weights_file.split("/")[1].split("_")[0]
        return [model_title, round(time.time() - start_time, 2)]

    def predict(self, data: pd.DataFrame, pas: float = 0.2, level: int = 0) -> int:
        """
        Загружает модель и делает предсказание.

        :param data: Датафрейм с входными признаками.
        :return: Датафрейм с предсказаниями и торговыми сигналами.
        """
        # pas = 0.4 if pas > 0.4 else pas
        df = data.copy()
        # X = df.drop(columns=[self.target_col])
        X = df.copy()
        # X = df.drop(columns=[self.target_col]).values.astype('float32')
        # X = np.expand_dims(X, axis=1)
        # X = df

        # Загружаем модель
        if os.path.exists(self.weights_file):
            self.model = joblib.load(self.weights_file)
            print(f"Загружены веса из {self.weights_file}")
        else:
            raise FileNotFoundError(f"Файл с весами {self.weights_file} не найден!")

        preds = self.model.predict(X)
        df['PredictedValue'] = preds
        print(df)
        # Генерация торговых сигналов
        if isinstance(self.model, (RandomForestClassifier, GradientBoostingClassifier, LogisticRegression,
                                   KNeighborsClassifier, GaussianNB, SVC, DecisionTreeClassifier)):
            df['Signal'] = np.where(df['PredictedValue'] == 1, 1, -1)
        else:
            df['Signal'] = np.where(df['PredictedValue'] > df["close"], 1, -1)
        return df


class AlgorithmLinearRegression(BaseAlgorithm):
    def __init__(self, weights_file="linear_regression_class.pkl", target_col="close", fit_intercept=True):
        super().__init__(LinearRegression, weights_file, target_col, fit_intercept=fit_intercept)


class AlgorithmLasso(BaseAlgorithm):
    def __init__(self, weights_file="lasso_class.pkl", target_col="close", alpha=1.0):
        super().__init__(Lasso, weights_file, target_col, alpha=alpha)


class AlgorithmRidge(BaseAlgorithm):
    def __init__(self, weights_file="ridge_class.pkl", target_col="close", alpha=1.0):
        super().__init__(Ridge, weights_file, target_col, alpha=alpha)


class AlgorithmElasticNet(BaseAlgorithm):
    def __init__(self, weights_file="elasticnet_class.pkl", target_col="close", alpha=1.0, l1_ratio=0.5):
        super().__init__(ElasticNet, weights_file, target_col, alpha=alpha, l1_ratio=l1_ratio)


class AlgorithmBayesianRidge(BaseAlgorithm):
    def __init__(self, weights_file="bayesian_ridge_class.pkl", target_col="close"):
        super().__init__(BayesianRidge, weights_file, target_col)


class AlgorithmSGDRegressor(BaseAlgorithm):
    def __init__(self, weights_file="sgd_regressor_class.pkl", target_col="close", max_iter=1000, tol=1e-3):
        super().__init__(SGDRegressor, weights_file, target_col, max_iter=max_iter, tol=tol)


class AlgorithmDecisionTreeRegressor(BaseAlgorithm):
    def __init__(self, weights_file="decision_tree_regressor_class.pkl", target_col="close", max_depth=None, random_state=42):
        super().__init__(DecisionTreeRegressor, weights_file, target_col, max_depth=max_depth, random_state=random_state)


class AlgorithmRandomForestRegressor(BaseAlgorithm):
    def __init__(self, weights_file="random_forest_regressor_class.pkl", target_col="close", n_estimators=100, random_state=42):
        super().__init__(RandomForestRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmExtraTreesRegressor(BaseAlgorithm):
    def __init__(self, weights_file="extra_trees_regressor_class.pkl", target_col="close", n_estimators=100, random_state=42):
        super().__init__(ExtraTreesRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmGradientBoostingRegressor(BaseAlgorithm):
    def __init__(self, weights_file="gradient_boosting_regressor_class.pkl", target_col="close", n_estimators=100, learning_rate=0.1, random_state=42):
        super().__init__(GradientBoostingRegressor, weights_file, target_col, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)


class AlgorithmHistGradientBoostingRegressor(BaseAlgorithm):
    def __init__(self, weights_file="hist_gradient_boosting_regressor_class.pkl", target_col="close", max_iter=100):
        super().__init__(HistGradientBoostingRegressor, weights_file, target_col, max_iter=max_iter)


class AlgorithmSVR(BaseAlgorithm):
    def __init__(self, weights_file="svr_class.pkl", target_col="close", kernel="rbf", C=1.0):
        super().__init__(SVR, weights_file, target_col, kernel=kernel, C=C)


class AlgorithmMLPRegressor(BaseAlgorithm):
    def __init__(self, weights_file="mlp_regressor_class.pkl", target_col="close", hidden_layer_sizes=(100,), max_iter=500):
        super().__init__(MLPRegressor, weights_file, target_col, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)


class AlgorithmBaggingRegressor(BaseAlgorithm):
    def __init__(self, weights_file="bagging_regressor_class.pkl", target_col="close", n_estimators=10, random_state=42):
        super().__init__(BaggingRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmVotingRegressor(BaseAlgorithm):
    def __init__(self, weights_file="voting_regressor_class.pkl", target_col="close", estimators=None):
        if estimators is None:
            estimators = [("rf", RandomForestRegressor(n_estimators=50)), ("gb", GradientBoostingRegressor(n_estimators=50))]
        super().__init__(VotingRegressor, weights_file, target_col, estimators=estimators)


class AlgorithmLightGBMRegressor(BaseAlgorithm):
    def __init__(self, weights_file="lightgbm_regressor_class.pkl", target_col="close", n_estimators=100, learning_rate=0.1):
        super().__init__(LGBMRegressor, weights_file, target_col, n_estimators=n_estimators, learning_rate=learning_rate)


class AlgorithmCatBoostRegressor(BaseAlgorithm):
    def __init__(self, weights_file="catboost_regressor_class.pkl", target_col="close", iterations=100, learning_rate=0.1):
        super().__init__(CatBoostRegressor, weights_file, target_col, iterations=iterations, learning_rate=learning_rate, verbose=0)


class AlgorithmXGBoostRegressor(BaseAlgorithm):
    def __init__(self, weights_file="xgboost_regressor_class.pkl", target_col="close", n_estimators=100, learning_rate=0.1):
        super().__init__(XGBRegressor, weights_file, target_col, n_estimators=n_estimators, learning_rate=learning_rate)


def train_models():
    df = pd.read_csv("new_dataset_buy.csv")
    x = df[["open", "high", "low", "close", "volume"]]
    y = df[["class"]]
    model_lr = AlgorithmLinearRegression()
    model_lasso = AlgorithmLasso()
    model_ridge = AlgorithmRidge()
    model_en = AlgorithmElasticNet()
    model_br = AlgorithmBayesianRidge()
    model_sgd = AlgorithmSGDRegressor()
    model_dtr = AlgorithmDecisionTreeRegressor()
    model_rfr = AlgorithmRandomForestRegressor()
    model_etr = AlgorithmExtraTreesRegressor()
    model_gbr = AlgorithmGradientBoostingRegressor()
    # model_hgbr = AlgorithmHistGradientBoostingRegressor()
    model_svr = AlgorithmSVR()
    model_mlpr = AlgorithmMLPRegressor()
    model_bagr = AlgorithmBaggingRegressor()
    model_vt = AlgorithmVotingRegressor()
    model_xgr = AlgorithmXGBoostRegressor()
    model_cr = AlgorithmCatBoostRegressor()
    model_lgr = AlgorithmLightGBMRegressor()
    #
    # # Обучение
    train_time = list()

    train_time.append(model_lr.train(x, y))
    train_time.append(model_lasso.train(x, y))
    train_time.append(model_ridge.train(x, y))
    train_time.append(model_en.train(x, y))
    train_time.append(model_br.train(x, y))
    train_time.append(model_sgd.train(x, y))
    train_time.append(model_dtr.train(x, y))
    train_time.append(model_rfr.train(x, y))
    train_time.append(model_etr.train(x, y))
    train_time.append(model_gbr.train(x, y))
    # train_time.append(model_hgbr.train(x, y))
    train_time.append(model_svr.train(x, y))
    train_time.append(model_mlpr.train(x, y))
    train_time.append(model_bagr.train(x, y))
    train_time.append(model_vt.train(x, y))
    train_time.append(model_xgr.train(x, y))
    train_time.append(model_cr.train(x, y))
    train_time.append(model_lgr.train(x, y))

    train_time = np.array(train_time)
    stat = pd.DataFrame(train_time, columns=["Model", "Train Time (s)"])
    stat.to_csv("Stat_train_ml_buy.csv")


if __name__ == "__main__":
    train_models()
    token = "8038576871:AAFAQ60J11zxws3FlhFMZ4lgzSq8Rdn8iMw"
    chat_id = "961023982"
    message = "Обучение моделей закончилось: МЛ покупка"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message
    }

    response = requests.post(url, json=payload)

