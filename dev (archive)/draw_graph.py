import matplotlib.pyplot as plt

# Данные из таблицы
industries = [
    "Manufacturing",
    "Finance",
    "Healthcare",
    "Transportation",
    "Security",
    "Business & legal services",
    "Others",
    "Energy",
    "Media & Entertainment",
    "Retail",
    "Semiconductor"
]

market_shares = [
    18.88,
    15.42,
    12.23,
    10.63,
    10.10,
    9.86,
    5.83,
    5.58,
    5.19,
    4.67,
    1.61
]

# Настраиваем размер и стиль графика
plt.figure(figsize=(8, 6))
# Делаем горизонтальные полосы
plt.barh(industries, market_shares, color='skyblue')

# Подписываем оси и заголовок
plt.xlabel("Market Share (%)")
plt.title("Market share distribution by industry, 2022")

# Разворачиваем чтобы самая большая доля была сверху
plt.gca().invert_yaxis()

# Добавляем сетку по оси X
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.pie(
    market_shares,
    labels=industries,
    autopct='%1.1f%%',
    startangle=140
)
plt.title("Market share distribution by industry, 2022")
plt.show()
