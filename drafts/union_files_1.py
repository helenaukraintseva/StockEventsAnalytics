import os
import pandas as pd

# Папка, где лежат .csv файлы
directory = ""

# Список нужных моделей
models = [
    "TorchRNN", "TorchLSTM", "TorchGRU",
    "RandomForest", "LogisticRegression", "DecisionTree",
    "GradientBoosting", "AdaBoost", "GaussianNB", "KNN"
]

# Собираем все .csv файлы
summary = []

for model_name in models:
    filename = os.path.join(directory, f"{model_name}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, header=None, names=["Metric", "Value"])
        df = df.set_index("Metric").T
        df["Model"] = model_name
        summary.append(df)

# Объединяем всё в один DataFrame
result_df = pd.concat(summary, ignore_index=True)

# Переставим "Model" первой колонкой
cols = ["Model"] + [col for col in result_df.columns if col != "Model"]
result_df = result_df[cols]
result_df.to_csv("total_stat.csv")

# Покажем результат
# import ace_tools as tools; tools.display_dataframe_to_user(name="Сводная статистика по моделям", dataframe=result_df)
