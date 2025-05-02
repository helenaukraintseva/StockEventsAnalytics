import matplotlib.pyplot as plt
import numpy as np

# Годы, по которым у нас есть приблизительные оценки
years = [2020, 2021, 2022]

# Общий рынок ML (в млрд $), приблизительно
ml_market = [15, 20, 30]

# Минимальные и максимальные оценки рынка AutoML (в млрд $)
automl_min = [0.2, 0.5, 0.7]
automl_max = [0.4, 0.7, 1.0]

# Считаем примерную (среднюю) оценку для AutoML
automl_avg = [(lo + hi) / 2 for lo, hi in zip(automl_min, automl_max)]

# Доля AutoML в общем рынке (в процентах)
# (используем среднее значение automl_avg и делим на ml_market)
share_automl_percent = [
    (automl_avg[i] / ml_market[i]) * 100
    for i in range(len(years))
]

fig, ax1 = plt.subplots(figsize=(8, 5))

# Построим столбиковую диаграмму для общего рынка ML
x = np.arange(len(years))  # [0, 1, 2] для 3-х лет
width = 0.35

bars1 = ax1.bar(x - width/2, ml_market, width, label='Общий рынок ML (млрд $)', alpha=0.7)

# Построим столбики для AutoML (средняя оценка)
bars2 = ax1.bar(x + width/2, automl_avg, width, label='AutoML (млрд $) - средняя оценка', alpha=0.7)

# Подписи осей
ax1.set_xlabel('Год')
ax1.set_ylabel('Объём рынка (млрд $)')
ax1.set_title('Приблизительная динамика рынка ML и AutoML')

# Устанавливаем подписи по оси X
ax1.set_xticks(x)
ax1.set_xticklabels(years)

# Включаем легенду
ax1.legend()

# Создадим вторую ось Y для отображения процента (доли AutoML)
ax2 = ax1.twinx()
ax2.set_ylabel('Доля AutoML в рынке ML (%)')

# Построим график (линию) для доли AutoML
ax2.plot(x, share_automl_percent, marker='o', color='red', label='Доля AutoML (%)')

# Отображаем сетку на второй оси (по желанию)
ax2.grid(False)  # Чтобы сетка второй оси не дублировалась с первой
ax2.set_ylim(0, max(share_automl_percent)*1.2)

# Добавляем легенду ко второй оси
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
