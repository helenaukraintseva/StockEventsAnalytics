import os
import random

# Путь к папке с изображениями
dir_list = ["archive (2)/train/cats", "archive (2)/train/dogs", "archive (2)/test/cats", "archive (2)/test/dogs",
            "archive (2)/dogs_vs_cats/train/dogs", "archive (2)/dogs_vs_cats/train/cats",
            "archive (2)/dogs_vs_cats/test/dogs", "archive (2)/dogs_vs_cats/test/cats", ]


# Сколько файлов нужно оставить


def func(dir, keep_count=100):
    # Допустимые расширения изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

    # Получаем список всех изображений в папке
    all_images = [f for f in os.listdir(dir) if f.lower().endswith(image_extensions)]

    # Если изображений меньше, чем нужно оставить — ничего не удаляем
    if len(all_images) <= keep_count:
        print(f"В папке всего {len(all_images)} изображений — ничего не удалено.")
    else:
        # Выбираем случайные изображения, которые нужно оставить
        keep_images = set(random.sample(all_images, keep_count))

        # Удаляем остальные изображения
        for image in all_images:
            if image not in keep_images:
                os.remove(os.path.join(dir, image))

        print(f"Оставлено {keep_count} изображений из {len(all_images)}.")


if __name__ == "__main__":
    for dir in dir_list:
        func(dir)
