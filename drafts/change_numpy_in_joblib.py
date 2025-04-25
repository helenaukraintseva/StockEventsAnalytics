import os
import joblib


def resave_pickles(folder_path, suffix="_compatible", compress=3):
    """
    Пересохраняет все .pkl файлы в папке в совместимом формате.

    :param folder_path: Путь к папке с .pkl файлами
    :param suffix: Суффикс для новых файлов (по умолчанию '_compatible')
    :param compress: Уровень компрессии (0–9, где 0 — без сжатия)
    """
    for file in os.listdir(folder_path):
        if file.endswith(".pkl"):
            original_path = os.path.join(folder_path, file)
            # new_filename = file.replace(".pkl", f"{suffix}.pkl")
            new_path = os.path.join("new_pkls", file)

            try:
                print(f"📥 Загрузка: {original_path}")
                obj = joblib.load(original_path)

                print(f"💾 Сохранение как: {new_path}")
                joblib.dump(obj, new_path, compress=compress)

            except Exception as e:
                print(f"❌ Ошибка при обработке {file}: {e}")


# === Пример использования ===
if __name__ == "__main__":
    resave_pickles("NLP")
    # resave_pickles("trained_signal_models_3")
