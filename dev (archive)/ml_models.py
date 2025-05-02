import numpy as np
import pandas as pd
import os
import time
import joblib
import requests
import math
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, ElasticNet, BayesianRidge, SGDRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier


class BaseAlgorithm:
    def __init__(self,
                 model_class,
                 weights_file,
                 target_col,
                 self_work=False,
                 **model_kwargs):
        """
        Базовый класс для всех алгоритмов машинного обучения.

        :param model_class: Класс модели (LinearRegression, RandomForest, и т. д.)
        :param weights_file: Файл для сохранения/загрузки модели.
        :param target_col: Название столбца-цели.
        :param model_kwargs: Дополнительные параметры для модели.
        """
        directory = "modelsdfasfasfd"
        # self.ensure_directory_exists(directory)
        self.weights_file = f"{directory}/{weights_file}"
        self.target_col = target_col
        self.model = model_class(**model_kwargs)
        self.self_work = self_work

    @staticmethod
    def ensure_directory_exists(directory_path="modelsdfasfasfd"):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Директория '{directory_path}' создана.")
        else:
            print(f"Директория '{directory_path}' уже существует.")

    def fit(self, X, y):
        """
        Обучает модель на предоставленных данных и сохраняет веса.

        :param data: Датафрейм с входными признаками и целевой переменной.
        """
        if self.self_work:
            start_time = time.time()
            # Обучение модели
            self.model.fit(X, y)
            # Сохранение модели
            joblib.dump(self.model, self.weights_file)
            print(f"Модель сохранена в {self.weights_file}")
            model_title = self.weights_file.split("/")[1].split("_")[0]
            return [model_title, round(time.time() - start_time, 2)]
        else:
            self.model.fit(X, y)

    def predict(self, data: pd.DataFrame):
        """
        Загружает модель и делает предсказание.

        :param data: Датафрейм с входными признаками.
        :return: Датафрейм с предсказаниями и торговыми сигналами.
        """
        X = data.copy()
        if self.self_work:

            # Загружаем модель
            if os.path.exists(self.weights_file):
                self.model = joblib.load(self.weights_file)
                print(f"Загружены веса из {self.weights_file}")
            else:
                raise FileNotFoundError(f"Файл с весами {self.weights_file} не найден!")

            preds = self.model.predict(X)
            return preds
        else:
            pred = self.model.predict(X)
            return pred


class ClAlgo(BaseAlgorithm):
    def __init__(self,
                 model_class,
                 weights_file,
                 target_col,
                 self_work=False,
                 **model_kwargs):
        super().__init__(model_class, weights_file, target_col, self_work, **model_kwargs)


class ReAlgo(BaseAlgorithm):
    def __init__(self,
                 model_class,
                 weights_file,
                 target_col,
                 self_work=False,
                 **model_kwargs):
        super().__init__(model_class, weights_file, target_col, self_work, **model_kwargs)


class AlgorithmLinearRegression(ReAlgo):
    def __init__(self, weights_file="linear_regression_buy.pkl", target_col="close", fit_intercept=True):
        super().__init__(LinearRegression, weights_file, target_col, fit_intercept=fit_intercept)


class AlgorithmLasso(ReAlgo):
    def __init__(self, weights_file="lasso_buy.pkl", target_col="close", alpha=1.0):
        super().__init__(Lasso, weights_file, target_col, alpha=alpha)


class AlgorithmRidge(ReAlgo):
    def __init__(self, weights_file="ridge_buy.pkl", target_col="close", alpha=1.0):
        super().__init__(Ridge, weights_file, target_col, alpha=alpha)


class AlgorithmElasticNet(ReAlgo):
    def __init__(self, weights_file="elasticnet_buy.pkl", target_col="close", alpha=1.0, l1_ratio=0.5):
        super().__init__(ElasticNet, weights_file, target_col, alpha=alpha, l1_ratio=l1_ratio)


class AlgorithmBayesianRidge(ReAlgo):
    def __init__(self, weights_file="bayesian_ridge_buy.pkl", target_col="close"):
        super().__init__(BayesianRidge, weights_file, target_col)


class AlgorithmSGDRegressor(ReAlgo):
    def __init__(self, weights_file="sgd_regressor_buy.pkl", target_col="close", max_iter=1000, tol=1e-3):
        super().__init__(SGDRegressor, weights_file, target_col, max_iter=max_iter, tol=tol)


class AlgorithmDecisionTreeRegressor(ReAlgo):
    def __init__(self, weights_file="decision_tree_regressor_buy.pkl", target_col="close", max_depth=None, random_state=42):
        super().__init__(DecisionTreeRegressor, weights_file, target_col, max_depth=max_depth, random_state=random_state)


class AlgorithmRandomForestRegressor(ReAlgo):
    def __init__(self, weights_file="random_forest_regressor_buy.pkl", target_col="close", n_estimators=100, random_state=42):
        super().__init__(RandomForestRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmExtraTreesRegressor(ReAlgo):
    def __init__(self, weights_file="extra_trees_regressor_buy.pkl", target_col="close", n_estimators=100, random_state=42):
        super().__init__(ExtraTreesRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmGradientBoostingRegressor(ReAlgo):
    def __init__(self, weights_file="gradient_boosting_regressor_buy.pkl", target_col="close", n_estimators=100, learning_rate=0.1, random_state=42):
        super().__init__(GradientBoostingRegressor, weights_file, target_col, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)


class AlgorithmHistGradientBoostingRegressor(ReAlgo):
    def __init__(self, weights_file="hist_gradient_boosting_regressor_buy.pkl", target_col="close", max_iter=100):
        super().__init__(HistGradientBoostingRegressor, weights_file, target_col, max_iter=max_iter)


class AlgorithmSVR(ReAlgo):
    def __init__(self, weights_file="svr_buy.pkl", target_col="close", kernel="rbf", C=1.0):
        super().__init__(SVR, weights_file, target_col, kernel=kernel, C=C)


class AlgorithmMLPRegressor(ReAlgo):
    def __init__(self, weights_file="mlp_regressor_buy.pkl", target_col="close", hidden_layer_sizes=(100,), max_iter=500):
        super().__init__(MLPRegressor, weights_file, target_col, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)


class AlgorithmBaggingRegressor(ReAlgo):
    def __init__(self, weights_file="bagging_regressor_buy.pkl", target_col="close", n_estimators=10, random_state=42):
        super().__init__(BaggingRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmVotingRegressor(ReAlgo):
    def __init__(self, weights_file="voting_regressor_buy.pkl", target_col="close", estimators=None):
        if estimators is None:
            estimators = [("rf", RandomForestRegressor(n_estimators=50)), ("gb", GradientBoostingRegressor(n_estimators=50))]
        super().__init__(VotingRegressor, weights_file, target_col, estimators=estimators)


class AlgorithmLightGBMRegressor(ReAlgo):
    def __init__(self, weights_file="lightgbm_regressor_buy.pkl", target_col="close", n_estimators=100, learning_rate=0.1):
        super().__init__(LGBMRegressor, weights_file, target_col, n_estimators=n_estimators, learning_rate=learning_rate)


class AlgorithmCatBoostRegressor(ReAlgo):
    def __init__(self, weights_file="catboost_regressor_buy.pkl", target_col="close", iterations=100, learning_rate=0.1):
        super().__init__(CatBoostRegressor, weights_file, target_col, iterations=iterations, learning_rate=learning_rate, verbose=0)


class AlgorithmXGBoostRegressor(ReAlgo):
    def __init__(self, weights_file="xgboost_regressor_buy.pkl", target_col="close", n_estimators=100, learning_rate=0.1):
        super().__init__(XGBRegressor, weights_file, target_col, n_estimators=n_estimators, learning_rate=learning_rate)

class AlgorithmLogisticRegressionClf(ClAlgo):
    def __init__(self,
                 weights_file="logistic_regression_clf.pkl",
                 target_col="class",
                 self_work=False,
                 C=1.0,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для LogisticRegression (классификация).
        """
        super().__init__(
            LogisticRegression,
            weights_file,
            target_col,
            self_work,
            C=C,
            random_state=random_state,
            **kwargs
        )

class AlgorithmDecisionTreeClassifierClf(ClAlgo):
    def __init__(self,
                 weights_file="decision_tree_clf.pkl",
                 target_col="class",
                 self_work=False,
                 max_depth=None,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для DecisionTreeClassifier.
        """
        super().__init__(
            DecisionTreeClassifier,
            weights_file,
            target_col,
            self_work,
            max_depth=max_depth,
            random_state=random_state,
            **kwargs
        )

class AlgorithmRandomForestClassifierClf(ClAlgo):
    def __init__(self,
                 weights_file="random_forest_clf.pkl",
                 target_col="class",
                 self_work=False,
                 n_estimators=100,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для RandomForestClassifier.
        """
        super().__init__(
            RandomForestClassifier,
            weights_file,
            target_col,
            self_work,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )

class AlgorithmSVCClf(ClAlgo):
    def __init__(self,
                 weights_file="svc_clf.pkl",
                 target_col="class",
                 self_work=False,
                 kernel="rbf",
                 C=1.0,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для SVC (Support Vector Classifier).
        """
        super().__init__(
            SVC,
            weights_file,
            target_col,
            self_work,
            kernel=kernel,
            C=C,
            random_state=random_state,
            **kwargs
        )

class AlgorithmNaiveBayesClf(ClAlgo):
    def __init__(self,
                 weights_file="naive_bayes_clf.pkl",
                 target_col="class",
                 self_work=False,
                 **kwargs):
        """
        Класс-обёртка для GaussianNB.
        """
        super().__init__(
            GaussianNB,
            weights_file,
            target_col,
            self_work,
            **kwargs
        )

class AlgorithmKNeighborsClf(ClAlgo):
    def __init__(self,
                 weights_file="kneighbors_clf.pkl",
                 target_col="class",
                 self_work=False,
                 n_neighbors=5,
                 **kwargs):
        """
        Класс-обёртка для KNeighborsClassifier.
        """
        super().__init__(
            KNeighborsClassifier,
            weights_file,
            target_col,
            self_work,
            n_neighbors=n_neighbors,
            **kwargs
        )

class AlgorithmGradientBoostingClassifierClf(ClAlgo):
    def __init__(self,
                 weights_file="gradient_boosting_clf.pkl",
                 target_col="class",
                 self_work=False,
                 n_estimators=100,
                 learning_rate=0.1,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для GradientBoostingClassifier (sklearn).
        """
        super().__init__(
            GradientBoostingClassifier,
            weights_file,
            target_col,
            self_work,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )

class AlgorithmXGBoostClassifierClf(ClAlgo):
    def __init__(self,
                 weights_file="xgboost_clf.pkl",
                 target_col="class",
                 self_work=False,
                 n_estimators=100,
                 learning_rate=0.1,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для XGBClassifier (XGBoost).
        """
        super().__init__(
            XGBClassifier,
            weights_file,
            target_col,
            self_work,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            use_label_encoder=False,   # Часто требуется, чтобы подавить warning
            eval_metric='logloss',     # Для классификации по умолчанию
            **kwargs
        )

class AlgorithmCatBoostClassifierClf(ClAlgo):
    def __init__(self,
                 weights_file="catboost_clf.pkl",
                 target_col="class",
                 self_work=False,
                 iterations=100,
                 learning_rate=0.1,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для CatBoostClassifier.
        """
        super().__init__(
            CatBoostClassifier,
            weights_file,
            target_col,
            self_work,
            iterations=iterations,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=0,
            **kwargs
        )

class AlgorithmLightGBMClassifierClf(ClAlgo):
    def __init__(self,
                 weights_file="lightgbm_clf.pkl",
                 target_col="class",
                 self_work=False,
                 n_estimators=100,
                 learning_rate=0.1,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для LGBMClassifier (LightGBM).
        """
        super().__init__(
            LGBMClassifier,
            weights_file,
            target_col,
            self_work,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            **kwargs
        )

class AlgorithmMLPClassifierClf(ClAlgo):
    def __init__(self,
                 weights_file="mlp_classifier.pkl",
                 target_col="class",
                 self_work=False,
                 hidden_layer_sizes=(100,),
                 max_iter=500,
                 random_state=42,
                 **kwargs):
        """
        Класс-обёртка для MLPClassifier (искусственные нейронные сети).
        """
        super().__init__(
            MLPClassifier,
            weights_file,
            target_col,
            self_work,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )

















#########################################
#########################################
#########################################
#########################################
#########################################
class AlgorithmLinearRegression(ReAlgo):
    def __init__(self, weights_file="linear_regression_class.pkl", target_col="close", fit_intercept=True):
        super().__init__(LinearRegression, weights_file, target_col, fit_intercept=fit_intercept)


class AlgorithmLasso(ReAlgo):
    def __init__(self, weights_file="lasso_class.pkl", target_col="close", alpha=1.0):
        super().__init__(Lasso, weights_file, target_col, alpha=alpha)


class AlgorithmRidge(ReAlgo):
    def __init__(self, weights_file="ridge_class.pkl", target_col="close", alpha=1.0):
        super().__init__(Ridge, weights_file, target_col, alpha=alpha)


class AlgorithmElasticNet(ReAlgo):
    def __init__(self, weights_file="elasticnet_class.pkl", target_col="close", alpha=1.0, l1_ratio=0.5):
        super().__init__(ElasticNet, weights_file, target_col, alpha=alpha, l1_ratio=l1_ratio)


class AlgorithmBayesianRidge(ReAlgo):
    def __init__(self, weights_file="bayesian_ridge_class.pkl", target_col="close"):
        super().__init__(BayesianRidge, weights_file, target_col)


class AlgorithmSGDRegressor(ReAlgo):
    def __init__(self, weights_file="sgd_regressor_class.pkl", target_col="close", max_iter=1000, tol=1e-3):
        super().__init__(SGDRegressor, weights_file, target_col, max_iter=max_iter, tol=tol)


class AlgorithmDecisionTreeRegressor(ReAlgo):
    def __init__(self, weights_file="decision_tree_regressor_class.pkl", target_col="close", max_depth=None, random_state=42):
        super().__init__(DecisionTreeRegressor, weights_file, target_col, max_depth=max_depth, random_state=random_state)


class AlgorithmRandomForestRegressor(ReAlgo):
    def __init__(self, weights_file="random_forest_regressor_class.pkl", target_col="close", n_estimators=100, random_state=42):
        super().__init__(RandomForestRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmExtraTreesRegressor(ReAlgo):
    def __init__(self, weights_file="extra_trees_regressor_class.pkl", target_col="close", n_estimators=100, random_state=42):
        super().__init__(ExtraTreesRegressor, weights_file, target_col, n_estimators=n_estimators, random_state=random_state)


class AlgorithmGradientBoostingRegressor(ReAlgo):
    def __init__(self, weights_file="gradient_boosting_regressor_class.pkl", target_col="close", n_estimators=100, learning_rate=0.1, random_state=42):
        super().__init__(GradientBoostingRegressor, weights_file, target_col, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)


class AlgorithmHistGradientBoostingRegressor(ReAlgo):
    def __init__(self, weights_file="hist_gradient_boosting_regressor_class.pkl", target_col="close", max_iter=100):
        super().__init__(HistGradientBoostingRegressor, weights_file, target_col, max_iter=max_iter)


class AlgorithmSVR(ReAlgo):
    def __init__(self, weights_file="svr_class.pkl", target_col="close", kernel="rbf", C=1.0):
        super().__init__(SVR, weights_file, target_col, kernel=kernel, C=C)


class AlgorithmMLPRegressor(ReAlgo):
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

