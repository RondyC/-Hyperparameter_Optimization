# Genetic Algorithm for Neural Network Hyperparameter Optimization

## 🧠 Описание проекта

Этот проект демонстрирует использование **генетического алгоритма (ГА)** для автоматической оптимизации гиперпараметров трёх различных архитектур нейронных сетей:

- **MLP (Multi-Layer Perceptron)**
- **1D CNN (Сверточная сеть по признакам)**
- **RNN (на базе LSTM)**

Все модели обучаются на одном и том же датасете `load_wine()` из библиотеки `sklearn`, содержащем **13 числовых признаков** и **3 класса** вин.

---

## 🎯 Цель

- Исследовать, какая архитектура наиболее эффективно справляется с задачей классификации вина.
- Автоматически подобрать оптимальные гиперпараметры с помощью ГА:
  - Тип модели (`mlp`, `cnn`, `rnn`)
  - Learning rate
  - Количество скрытых слоёв
  - Размер слоя (нейронов/фильтров)
  - Dropout

---

## ⚙️ Как работает

1. **Инициализация популяции** — случайно создаются особи (наборы гиперпараметров).
2. **Оценка** — каждая особь представляет собой модель, которая обучается и оценивается по точности на тестовой выборке.
3. **Селекция** — отбор лучших с помощью турниров.
4. **Кроссовер и мутация** — создаются новые особи, которые заменяют старое поколение.
5. **Эволюция** — повторяется в течение нескольких поколений.
6. **Дообучение** — финальная модель дообучается на полной тренировочной выборке.
7. **Сохранение** — лучшая модель сохраняется в файл `best_wine_model.keras`.

---

## 📈 Визуализация

По ходу обучения сохраняется CSV-файл с историей точности по типам моделей (`ga_wine_history.csv`). Также строится график прогресса:

- Линии точности для MLP, CNN, RNN по поколениям.
- Видно, какая архитектура стабильно побеждает.

---

▶️ Запуск
Вы можете запустить код в любом Jupyter Notebook, Google Colab или локально через main.py.

🧪 Результаты
RNN достигла точности 1.0000 на тесте.

MLP показала стабильность и аналогично вышла на топовые результаты.

CNN показала нестабильность — требует улучшенной настройки.

Финальная модель дообучена и сохранена, готова к использованию.
