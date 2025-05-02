import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_csv(file_path):
    # Загрузка данных
    df = pd.read_csv(file_path)

    # Краткая структурная информация (столбцы, типы данных, пропуски)
    print("===== Info о данных =====")
    df.info()

    # Описание числовых столбцов (count, mean, std, min, max и др.)
    print("\n===== Описание числовых столбцов =====")
    print(df.describe())

    # Подсчёт пропусков (NaN) в каждом столбце
    print("\n===== Пропуски в каждом столбце =====")
    print(df.isna().sum())

    # Количество уникальных значений для каждого столбца
    print("\n===== Количество уникальных значений =====")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} уникальных значений")

    # Отбираем числовые столбцы (float, int)
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    if len(numeric_cols) > 0:
        # Гистограммы для числовых столбцов
        df[numeric_cols].hist(figsize=(12, 6))
        plt.suptitle("Гистограммы числовых столбцов")
        plt.show()

        # Boxplot (ящиковые диаграммы) для числовых столбцов
        plt.figure(figsize=(12,6))
        sns.boxplot(data=df[numeric_cols])
        plt.title("Boxplot для числовых столбцов")
        plt.show()
    else:
        print("Нет числовых столбцов для гистограммы/boxplot.")

    # Поиск потенциально категориальных столбцов (object, bool и т.п.)
    categorical_cols = df.select_dtypes(include=['object','bool','category']).columns
    for col in categorical_cols:
        # Проверяем, не слишком ли много уникальных значений (например, лимит 30)
        if df[col].nunique() <= 30:
            plt.figure(figsize=(8,4))
            # Считаем частоту каждого уникального значения и строим barplot
            df[col].value_counts().plot(kind='bar')
            plt.title(f"Распределение столбца '{col}'")
            plt.xlabel(col)
            plt.ylabel("Частота")
            plt.show()
        else:
            print(f"Столбец '{col}' имеет {df[col].nunique()} уникальных значений, пропускаем бар-чарт.")

if __name__ == "__main__":
    # Замените 'file_stat.csv' на нужный путь к вашему CSV-файлу
    analyze_and_plot_csv("file_stat.csv")
