import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import ml_model_price, ml_model_buy, nn_model_buy, nn_model_price
import time


def combine_timeseries_csv(csv_files, time_col="time", sort=True, interval="1m", count=None):
    """
    Функция, которая читает несколько CSV-файлов со столбцами:
    [time, open, close, high, low, volume, ...]
    и объединяет их в единый DataFrame.

    :param csv_files: список путей к CSV-файлам (str).
    :param time_col: название столбца с датой/временем. По умолчанию 'time'.
    :param sort: нужно ли сортировать конечный DataFrame по времени? По умолчанию True.
    :return: объединённый DataFrame со всеми строками.
    """

    # Пустой список для DataFrame
    df_list = []
    counter = 0
    for file_path in csv_files:
        if interval not in file_path:
            continue
        if count is not None:
            counter += 1
            if counter > count:
                break
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден, пропускаем.")
            continue

        df = pd.read_csv(file_path)
        df["symbol"] = file_path.split("\\")[-1].split("_")[0]

        # Проверяем наличие нужных столбцов
        required_cols = [time_col, "open", "high", "low", "close", "volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Файл {file_path} не содержит столбцы {missing}, пропускаем.")
            continue

        # Убедимся, что столбец time_col в формате datetime, если нужно
        # (или оставим, если уже datetime)
        # df[time_col] = pd.to_datetime(df[time_col])  # если формат строк

        df_list.append(df)

    if not df_list:
        print("Нет валидных файлов для объединения.")
        return pd.DataFrame()
    # Объединяем все DataFrame в один (по сути, просто вертикальный concat)
    combined_df = pd.concat(df_list, ignore_index=True)

    # Если нужно сортировать по времени
    if sort and time_col in combined_df.columns:
        combined_df.sort_values(by=time_col, inplace=True)

    # Сбрасываем индекс
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.to_csv(f"dataset_{interval}.csv")
    return combined_df


def make_dataset_from_file(filename):
    ff = 5
    window = int(filename.split("_")[3].replace("w", ""))
    df = pd.read_csv(filename)
    new_dict = dict()
    for ww in range(window):
        new_dict[f"open_s{ww + 1}"] = list()
        new_dict[f"high_s{ww + 1}"] = list()
        new_dict[f"low_s{ww + 1}"] = list()
        new_dict[f"close_s{ww + 1}"] = list()
        new_dict[f"volume_s{ww + 1}"] = list()
        new_dict[f"target"] = list()

    def make_batch(data):
        for ww in range(window):
            line = data[f"step{ww}"].split("|")
            new_dict[f"open_s{ww + 1}"].append(line[0])
            new_dict[f"high_s{ww + 1}"].append(line[1])
            new_dict[f"low_s{ww + 1}"].append(line[2])
            new_dict[f"close_s{ww + 1}"].append(line[3])
            new_dict[f"volume_s{ww + 1}"].append(line[4])
        new_dict["target"].append(data["target"])
        # print("UNIQUE: ", data["target"].unique())

    for index, elem in df.iterrows():
        make_batch(elem)
    new_df = pd.DataFrame(new_dict)
    return new_df


def list_files_in_directory(directory):
    try:
        files = list()
        # Получаем список файлов и папок в указанной директории
        items = os.listdir(directory)

        # Проходим по всем элементам и проверяем, являются ли они файлами
        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                # print(item)  # Выводим имя файла
                # item_path = item_path.replace("\\", "\")
                files.append(item_path)
        return files
    except Exception as e:
        print(f"Ошибка: {e}")


def count_class(filename, limit=800):
    target_counts = {}
    if filename.endswith('.csv'):
        # file_path = os.path.join(directory_path, filename)
        # print(file_path)

        # Читаем CSV файл
        df = pd.read_csv(filename)
        # Проверяем, существует ли столбец 'target'
        if 'target' in df.columns:
            # Считаем количество каждого уникального значения в столбце 'target'
            counts = df['target'].value_counts()

            # Обновляем общий словарь с результатами
            for value, count in counts.items():
                if value in target_counts:
                    target_counts[value] += count
                else:
                    target_counts[value] = count
    # print("_________________________________")
    # print(filename)
    # print(target_counts)
    nums = list()
    for value, count in target_counts.items():
        nums.append(count)
        # print(f"Значение: {value}, Количество: {count}")
    if min(nums) < limit:
        return False
    return min(nums)


def balance_classes(df, class_column):
    # Подсчет количества элементов каждого класса
    class_counts = df[class_column].value_counts()

    # Наименьшее количество элементов среди классов
    min_count = class_counts.min()

    # Создание списка для хранения сбалансированных данных
    balanced_data = []
    # Для каждого класса выбираем min_count случайных элементов
    for class_label in class_counts.index:
        class_data = df[df[class_column] == class_label]
        balanced_data.append(class_data.sample(min_count, random_state=42))

    # Объединяем все сбалансированные данные в один DataFrame
    balanced_df = pd.concat(balanced_data, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    x_full = balanced_df.drop(["target"], axis=1)
    y_full = balanced_df["target"]
    class_counts = balanced_df[class_column].value_counts()
    return x_full, y_full


def tran_pred(data, border=0.5):
    new_data = list()
    for elem in data:
        if elem + border > 1:
            new_data.append(1)
        elif elem - border < -1:
            new_data.append(-1)
        else:
            new_data.append(0)
    return new_data


def train_and_evaluate_models(
        models_dict: dict,
        target_col="target",
        output_dir="trained_models"
):
    """
    Функция обучает несколько моделей (из словаря), с учётом заданного интервала / шага предсказания,
    и возвращает информацию о весах и статистику обучения.

    :param models_dict: словарь вида {"LinearRegression": model_obj, "XGB": model_obj, ...}
    :param df: DataFrame со столбцами признаков и target_col (регрессия).
    :param interval: (start, end) или иной способ ограничить данные. Если None – берём все.
    :param forecast_step: на сколько баров/шагов вперёд делаем прогноз. По умолчанию 1 (следующий шаг).
    :param target_col: название столбца-цели.
    :param output_dir: директория для сохранения обученных моделей.
    :return: словарь вида:
        {
          "model_name": {
            "weights_file": "...",
            "mse": float,
            "rmse": float,
            "r2": float,
            "coefficients": (при наличии, например для линрега)
          },
          ...
        }
    """

    results = {}
    statistic = {
        "model": list(),
        "train_time": list(),
        "pred_time": list(),
        "MSE": list(),
        "RMSE": list(),
        "R2": list(),
        "MAE": list(),
        "Accuracy": list(),
        "Precision": list(),
        "Recall": list(),
        "F1": list(),
        "title": list()
    }
    csv_files = list_files_in_directory(directory="parsing\datasets")
    counter = 0
    limit = 5
    old_data = pd.read_csv("file_stat.csv")
    educated_model = list(old_data["title"])

    for filename in csv_files:
        save_flag = True
        # if "class" in file:
        #     continue
        # counter += 1
        # if counter == limit:
        #     break

        data = make_dataset_from_file(filename)

        # 1. Убедимся, что директория для сохранения весов существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data = data.reset_index(drop=True)

        task_type = filename.split("\\")[2][:2]
        if task_type == "cl":
            count_elem = count_class(filename)
            X_full, y_full = balance_classes(data, "target")

            split_index = int(count_elem * 0.9)

            X_train, X_test = X_full[:split_index], X_full[split_index:count_elem]
            y_train, y_test = y_full[:split_index], y_full[split_index:count_elem]
        else:
            X_full = data.drop(columns=[target_col, "time"], errors="ignore")
            y_full = data[target_col]
            X_full = X_full.values
            y_full = y_full.values
            split_index = int(len(X_full) * 0.8)
            X_train, X_test = X_full[:split_index], X_full[split_index:]
            y_train, y_test = y_full[:split_index], y_full[split_index:]
        print(f"Dataset ({filename}) is ready.")
        # 5. Для каждой модели: обучаем, делаем предсказание, сохраняем
        for model_name, model_obj in models_dict.items():
            if "class" in filename:
                type_file = "cl"
            else:
                type_file = "re"
            postfix = filename.split("\\")[2].replace("_dataset", "").replace("regress", "R")
            postfix = postfix.replace("class", "C").replace(".csv", "")
            weights_file = os.path.join(output_dir, f"{model_name}_{postfix}.pkl")
            title = weights_file.split("\\")[1]
            if title in educated_model:
                print("Model is exist.")
                save_flag = False
                continue

            params = models_dict[model_name]["params"]
            model_type = models_dict[model_name]["type_task"]
            if type_file not in model_type:
                continue
            model_obj = models_dict[model_name]["model"](**params)
            # Обучение
            start_time = time.time()
            model_obj.fit(X_train, y_train)
            train_time = round(time.time() - start_time, 2)

            start_time = time.time()
            # Предсказание
            y_pred = model_obj.predict(X_test)
            y_pred = tran_pred(y_pred)
            pred_time = round(time.time() - start_time, 4)
            if pred_time > 0.1:
                title = "File didn't saved, because model is very slowly."
            else:
                joblib.dump(model_obj, weights_file)
            # Метрики
            if task_type == "re" and "re" in model_type:
                mse = mean_squared_error(y_test, y_pred)
                rmse = mse ** 0.5
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                info = {
                    "weights_file": weights_file,
                    "mse": mse,
                    "rmse": rmse,
                    "r2": r2
                }
                statistic["model"].append(model_name)
                statistic["train_time"].append(train_time)
                statistic["pred_time"].append(pred_time)
                statistic["MSE"].append(round(mse, 4))
                statistic["RMSE"].append(round(rmse, 4))
                statistic["R2"].append(round(r2, 4))
                statistic["MAE"].append(round(mae, 4))
                statistic["Accuracy"].append(" ")
                statistic["Precision"].append(" ")
                statistic["Recall"].append(" ")
                statistic["F1"].append(" ")
                statistic["title"].append(title)
            elif task_type == "cl" and "cl" in model_type:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                statistic["model"].append(model_name)
                statistic["train_time"].append(train_time)
                statistic["pred_time"].append(pred_time)
                statistic["MSE"].append(" ")
                statistic["RMSE"].append(" ")
                statistic["R2"].append(" ")
                statistic["MAE"].append(" ")
                statistic["Accuracy"].append(round(accuracy, 4))
                statistic["Precision"].append(round(precision, 4))
                statistic["Recall"].append(round(recall, 4))
                statistic["F1"].append(round(f1, 4))
                statistic["title"].append(title)
                info = {
                    "weights_file": weights_file,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            else:
                continue
            print(f"Model: {model_name} - was evolved.")

            results[model_name] = info
        if save_flag:
            data = pd.DataFrame(statistic)
            all_data = pd.concat([old_data, data], axis=0)
            all_data.to_csv("file_stat.csv", index=False)
            print("file saved.")

    return results


# Пример использования
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor

    models_dict = {
        # "LinearRegression": {"type_task": ["re", "cl"],
        #                      "model": LinearRegression,
        #                      "params": {}},
        # "RandomForest_15": {"type_task": ["re", "cl"],
        #                     "model": RandomForestRegressor,
        #                     "params": {"n_estimators": 15, "random_state": 42}},
        # "RandomForest_10": {"type_task": ["re", "cl"],
        #                     "model": RandomForestRegressor,
        #                     "params": {"n_estimators": 10, "random_state": 42}},
        # "RandomForest_5": {"type_task": ["re", "cl"],
        #                    "model": RandomForestRegressor,
        #                    "params": {"n_estimators": 5, "random_state": 42}},
        "Lass_05": {"type_task": ["re"],
                    "model": ml_model_price.AlgorithmLasso,
                    "params": {"alpha": 0.5}},
        "Lass_1": {"type_task": ["re"],
                   "model": ml_model_price.AlgorithmLasso,
                   "params": {"alpha": 1.0}},
        "Lass_15": {"type_task": ["re"],
                    "model": ml_model_price.AlgorithmLasso,
                    "params": {"alpha": 1.5}},
        "ElasticNet_1_1": {"type_task": ["re"],
                           "model": ml_model_price.AlgorithmElasticNet,
                           "params": {"alpha": 1.0, "l1_ratio": 1.0}},
        "ElasticNet_1_05": {"type_task": ["re"],
                            "model": ml_model_price.AlgorithmElasticNet,
                            "params": {"alpha": 1.0, "l1_ratio": 0.5}},
        "ElasticNet_05_1": {"type_task": ["re"],
                            "model": ml_model_price.AlgorithmElasticNet,
                            "params": {"alpha": 0.5, "l1_ratio": 1.0}},
        "Ridge": {"type_task": ["re"],
                  "model": ml_model_price.AlgorithmElasticNet,
                  "params": {}},
        "SGD_tole3": {"type_task": ["re"],
                      "model": ml_model_price.AlgorithmSGDRegressor,
                      "params": {"max_iter": 1000, "tol": 1e-3}},
        "SGD_tole4": {"type_task": ["re"],
                      "model": ml_model_price.AlgorithmSGDRegressor,
                      "params": {"max_iter": 1000, "tol": 1e-4}},
        "DecisionTree": {"type_task": ["re"],
                         "model": ml_model_price.AlgorithmDecisionTreeRegressor,
                         "params": {}},
        "RandomForest_n20": {"type_task": ["re"],
                             "model": ml_model_price.AlgorithmRandomForestRegressor,
                             "params": {"n_estimators": 20, "random_state": 42}},
        "RandomForest_n40": {"type_task": ["re"],
                             "model": ml_model_price.AlgorithmRandomForestRegressor,
                             "params": {"n_estimators": 40, "random_state": 42}},
        "RandomForest_n80": {"type_task": ["re"],
                             "model": ml_model_price.AlgorithmRandomForestRegressor,
                             "params": {"n_estimators": 80, "random_state": 42}},
        "ExtraTrees_n20": {"type_task": ["re"],
                           "model": ml_model_price.AlgorithmExtraTreesRegressor,
                           "params": {"n_estimators": 20, "random_state": 42}},
        "ExtraTrees_n50": {"type_task": ["re"],
                           "model": ml_model_price.AlgorithmExtraTreesRegressor,
                           "params": {"n_estimators": 50, "random_state": 42}},
        "GradientBoosting_n20_l01": {"type_task": ["re"],
                                     "model": ml_model_price.AlgorithmGradientBoostingRegressor,
                                     "params": {"n_estimators": 20, "learning_rate": 0.1, "random_state": 42}},
        "GradientBoosting_n40_l01": {"type_task": ["re"],
                                     "model": ml_model_price.AlgorithmGradientBoostingRegressor,
                                     "params": {"n_estimators": 40, "learning_rate": 0.1, "random_state": 42}},
        "GradientBoosting_n20_l001": {"type_task": ["re"],
                                      "model": ml_model_price.AlgorithmGradientBoostingRegressor,
                                      "params": {"n_estimators": 20, "learning_rate": 0.01, "random_state": 42}},
        "HistGradientBoosting_m100": {"type_task": ["re"],
                                      "model": ml_model_price.AlgorithmHistGradientBoostingRegressor,
                                      "params": {"max_iter": 100}},
        "HistGradientBoosting_m200": {"type_task": ["re"],
                                      "model": ml_model_price.AlgorithmHistGradientBoostingRegressor,
                                      "params": {"max_iter": 200}},
        "SVR_c1": {"type_task": ["re"],
                   "model": ml_model_price.AlgorithmSVR,
                   "params": {"C": 1.0}},
        "SVR_c05": {"type_task": ["re"],
                    "model": ml_model_price.AlgorithmSVR,
                    "params": {"C": 0.5}},
    }

    res = train_and_evaluate_models(
        models_dict=models_dict,
        target_col="target",
        output_dir="trained_models"
    )

    print("RESULTS:", res)
