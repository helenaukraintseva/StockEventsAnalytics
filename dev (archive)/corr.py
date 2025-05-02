import os
import pandas as pd
from collections import deque
import time


def group_tokens_by_correlation(csv_files,
                                correlation_threshold=0.7,
                                close_col="close",
                                method: str = "pearson",
                                file_data: str = "1m"):
    """
    Функция, которая читает временные ряды из списка CSV-файлов,
    вычисляет корреляцию между токенами и формирует «классы» (группы),
    у которых взаимная корреляция > correlation_threshold.

    :param csv_files: список путей к CSV-файлам,
                      например ["BTC.csv", "ETH.csv", ...]
    :param correlation_threshold: порог корреляции (например, 0.7)
    :param close_col: название столбца цены (по умолчанию 'Close')
    :return: словарь групп вида:
             {
                "group1": ["BTC", "ETH"],
                "group2": ["BNB", "XRP"]
             }
    """
    # 1) Считываем все файлы в dict: {token_name: Series[float]}
    #    Предположим, имя токена = basename без .csv
    token_series = {}
    for file_path in csv_files:
        check_file = file_path.split(".")[0].split("_")[-1]
        if check_file != file_data:
            continue
        token_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)
        # Проверим, что столбец close_col есть
        if close_col not in df.columns:
            raise ValueError(f"В файле {file_path} нет столбца {close_col}.")
        # Преобразуем в Series, где индекс — это datetime или integer
        # Для простоты пусть будет индекс без парсинга дат,
        #  но можно сделать parse_dates=["time"] / set_index("time").
        s = df[close_col].copy().dropna()
        token_series[token_name] = s

    # 2) Сформируем общий DataFrame (возможно, с outer join),
    #    но для корреляции часто берут пересечение дат (inner join).
    #    Однако, чтобы упростить, сделаем pairwise корреляцию
    #    (без создания общего DataFrame).
    tokens = list(token_series.keys())
    n = len(tokens)

    # 3) Вычислим матрицу корреляций (n x n)
    #    Для каждой пары (i, j) => Series.corr(Series_j, method='pearson')
    corr_matrix = pd.DataFrame(index=tokens, columns=tokens, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix.iloc[i, j] = 1.0
            elif j > i:
                # Приведём Series к общему индексу (inner join)
                s1 = token_series[tokens[i]]
                s2 = token_series[tokens[j]]
                joined = pd.concat([s1, s2], axis=1, join="inner", keys=["s1", "s2"]).dropna()
                # Корреляция
                corr_val = joined["s1"].corr(joined["s2"], method=method)
                corr_matrix.iloc[i, j] = corr_val
                corr_matrix.iloc[j, i] = corr_val

    # 4) Определяем «граф» (adjacency), где вершины = tokens,
    #    ребро (i, j) если corr_matrix[i,j] >= threshold (i != j).
    adjacency = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                if corr_matrix.iloc[i, j] >= correlation_threshold:
                    adjacency[i].append(j)

    # 5) Поиск связных компонент => группы
    visited = [False] * n
    groups = []

    def bfs(start):
        queue = deque([start])
        visited[start] = True
        component = [tokens[start]]
        while queue:
            v = queue.popleft()
            for w in adjacency[v]:
                if not visited[w]:
                    visited[w] = True
                    queue.append(w)
                    component.append(tokens[w])
        return component

    for i in range(n):
        if not visited[i]:
            group = bfs(i)
            groups.append(group)

    # 6) Формируем словарь с названиями групп: group1, group2, ...
    result = {}
    for idx, grp in enumerate(groups, start=1):
        name = f"group{idx}"
        result[name] = grp
    return result


# Пример использования
if __name__ == "__main__":
    files = os.listdir("crypto_data")
    csv_list = [f"crypto_data/{title}" for title in files]

    # Допустим, у нас есть ["BTC.csv", "ETH.csv", "XRP.csv", ...]
    # csv_list = ["BTC.csv", "ETH.csv", "XRP.csv"]
    start_time = time.time()
    out = group_tokens_by_correlation(csv_list,
                                      correlation_threshold=0.95,
                                      close_col="close",
                                      method="pearson",
                                      file_data="1m")
    print(out)
    print(f"Time: {round(time.time() - start_time, 2)}sec")

    out = group_tokens_by_correlation(csv_list,
                                      correlation_threshold=0.95,
                                      close_col="close",
                                      method="pearson",
                                      file_data="5m")
    print(out)
    print(f"Time: {round(time.time() - start_time, 2)}sec")
    out = group_tokens_by_correlation(csv_list,
                                      correlation_threshold=0.95,
                                      close_col="close",
                                      method="pearson",
                                      file_data="30m")
    print(out)
    print(f"Time: {round(time.time() - start_time, 2)}sec")
    out = group_tokens_by_correlation(csv_list,
                                      correlation_threshold=0.95,
                                      close_col="close",
                                      method="pearson",
                                      file_data="1h")
    print(out)
    print(f"Time: {round(time.time() - start_time, 2)}sec")
    # Например:
    # {
    #   "group1": ["BTC", "ETH"],
    #   "group2": ["XRP"]
    # }
