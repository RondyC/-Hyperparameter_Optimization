# Оптимизация гиперпараметров с использованием нескольких моделей

* Задача:
 * Оптимизация гиперпараметров с использованием генетического алгоритма

* Цель:
 * Использовать ГА для подбора наилучшей комбинации гиперпараметров среди трёх разных архитектур нейронных сетей на одном и том же датасете.

* Описание:
 * На вход подаётся датасет `load_wine()` из библиотеки `sklearn`, содержащий 13 числовых признаков и 3 класса (категории вина).
Каждая особь в популяции кодирует архитектуру и гиперпараметры одной из моделей:
    - MLP (полносвязная сеть)
    - 1D-CNN (сверточная сеть)
    - LSTM (рекуррентная сеть)

* Оптимизируемые гиперпараметры:
    1. Тип модели: 'mlp', 'cnn', 'rnn'
    2. Learning rate (для оптимизатора Adam)
    3. Количество скрытых слоёв
    4. Размер скрытого слоя (нейроны или фильтры)
    5. Уровень Dropout

* Процесс:
 * Генетический алгоритм генерирует популяции из таких конфигураций, обучает модели и оценивает точность на тестовой выборке.
 * По мере эволюции отбираются лучшие особи, происходит кроссовер и мутация, и в итоге находится модель с наилучшей точностью.

* Результат:
 * Выводится прогресс обучения по каждому типу модели, сохраняется история, а лучшая найденная модель дополнительно дообучается и сохраняется на диск.

# Импорт необходимых библиотек и подавление предупреждений
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras.layers import (
    Input, Dense, Dropout,
    Conv1D, MaxPooling1D, Flatten,
    LSTM
)
from keras.optimizers import Adam

"""# Детали воспроизводимости

---

Фиксирует начальные значения генераторов случайных чисел (random и numpy), чтобы каждый запуск эксперимента давал одинаковые результаты. Это особенно важно для стабильности обучения и корректного сравнения моделей.
"""

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

"""# Загрузка и подготовка Wine

---

Загрузка датасета Wine из sklearn, разделение на обучающую и тестовую выборки с сохранением сбалансированности классов, нормализация признаков и подготовка данных в нужных форматах для моделей MLP, CNN и RNN. Также определяется количество классов для финального слоя нейросетей.
"""

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Форматы входа для трёх моделей
X_train_mlp = X_train
X_test_mlp  = X_test

X_train_seq = X_train[..., np.newaxis]
X_test_seq  = X_test[...,  np.newaxis]

num_classes = len(np.unique(y))

"""# Генерация случайного индивида

---

Функция создаёт одну случайную «особь» — набор гиперпараметров и тип модели (MLP, CNN или RNN), который будет использоваться в популяции генетического алгоритма. Каждый параметр выбирается случайно из заданного диапазона.
"""

def random_individual():
    return (
        random.choice(['mlp','cnn','rnn']),   # тип модели
        random.uniform(1e-4, 1e-1),           # learning rate
        random.randint(1, 3),                 # число слоёв
        random.randint(16, 128),              # units / filters
        random.uniform(0.0, 0.5)              # dropout rate
    )

"""# Фитнесс-функция

---

Функция оценивает качество одной особи — строит и обучает модель (MLP, CNN или RNN) с заданными гиперпараметрами, затем измеряет её точность (accuracy) на тестовой выборке. Это значение используется как приспособленность (fitness) при отборе особей в генетическом алгоритме.
"""

def fitness_function(ind):
    model_type, lr, n_layers, n_units, drop = ind

    model = Sequential()
    if model_type == 'mlp':
        model.add(Input(shape=(13,)))
        for _ in range(n_layers):
            model.add(Dense(n_units, activation='relu'))
            model.add(Dropout(drop))
        model.add(Dense(num_classes, activation='softmax'))
        x_tr, x_te = X_train_mlp, X_test_mlp

    elif model_type == 'cnn':
        model.add(Input(shape=(13,1)))
        for _ in range(n_layers):
            model.add(Conv1D(n_units, 3, padding='same', activation='relu'))
            model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dropout(drop))
        model.add(Dense(num_classes, activation='softmax'))
        x_tr, x_te = X_train_seq, X_test_seq

    else:  # rnn
        model.add(Input(shape=(13,1)))
        for i in range(n_layers):
            rs = (i < n_layers - 1)
            model.add(LSTM(n_units, return_sequences=rs))
            model.add(Dropout(drop))
        model.add(Dense(num_classes, activation='softmax'))
        x_tr, x_te = X_train_seq, X_test_seq

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(x_tr, y_train, epochs=20, batch_size=32, verbose=0)
    preds = model.predict(x_te, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    return accuracy_score(y_test, y_pred)

"""# Кроссовер и мутация

---

Здесь реализованы два ключевых оператора генетического алгоритма:
* Кроссовер (crossover) — комбинирует параметры двух родителей для создания новых особей, обмениваясь их гиперпараметрами.
* Мутация (mutate) — с заданной вероятностью случайно изменяет один из гиперпараметров особи, чтобы поддерживать разнообразие популяции и избежать преждевременной сходимости.
"""

def crossover(p1, p2):
    return (
        (p1[0], p2[1], p1[2], p2[3], p1[4]),
        (p2[0], p1[1], p2[2], p1[3], p2[4])
    )

def mutate(ind, rate):
    if random.random() < rate:
        genes = list(ind)
        i = random.randrange(len(genes))
        if i == 0:
            genes[0] = random.choice(['mlp','cnn','rnn'])
        elif i == 1:
            genes[1] = random.uniform(1e-4, 1e-1)
        elif i == 2:
            genes[2] = random.randint(1, 3)
        elif i == 3:
            genes[3] = random.randint(16, 128)
        else:
            genes[4] = random.uniform(0.0, 0.5)
        return tuple(genes)
    return ind

"""# Генетический алгоритм

---

Главная функция, реализующая цикл генетического алгоритма.
На каждом поколении:
* Оцениваются все особи с помощью фитнесс-функции;
* Выводится информация о точности каждой модели;
* Сохраняется лучший результат и история по типам моделей (MLP, CNN, RNN);
* Выполняется отбор (турнир), кроссовер и мутация для формирования следующего поколения.

В конце возвращаются параметры и точность наилучшей найденной модели, а также история для визуализации.
"""

def genetic_algorithm(pop_size, generations, mutation_rate):
    population = [random_individual() for _ in range(pop_size)]
    best = (None, 0.0)
    history = []

    for gen in range(1, generations + 1):
        print(f"\nПоколение {gen}/{generations}")
        scores = [fitness_function(ind) for ind in population]

        # Вывод каждой особи
        for i, (ind, acc) in enumerate(zip(population, scores), start=1):
            model_type, lr, n_layers, n_units, _ = ind
            print(f"Особь {i}: Модель={model_type.upper():<3}, LR={lr:.4f}, "
                  f"Слои={n_layers}, Нейроны={n_units}, Точность={acc:.4f}")

        # Проверка нового рекорда
        idx_best = int(np.argmax(scores))
        if scores[idx_best] > best[1]:
            best = (population[idx_best], scores[idx_best])
            model_type, best_lr, best_layers, best_units, _ = best[0]
            best_acc = best[1]
            print("\nНовый лучший результат!")
            print(f"Модель: {model_type.upper()}")
            print(f"Параметры: LR={best_lr:.4f}, Слои={best_layers}, Нейроны={best_units}")
            print(f"Точность: {best_acc:.4f}")

        # Сохранение истории
        mlp_acc = max((s for ind, s in zip(population, scores) if ind[0]=='mlp'), default=0.0)
        cnn_acc = max((s for ind, s in zip(population, scores) if ind[0]=='cnn'), default=0.0)
        rnn_acc = max((s for ind, s in zip(population, scores) if ind[0]=='rnn'), default=0.0)
        history.append({'gen': gen, 'mlp': mlp_acc, 'cnn': cnn_acc, 'rnn': rnn_acc})

        # Турнир и формирование нового поколения
        selected = [
            max(random.sample(list(zip(population, scores)), 3), key=lambda x: x[1])[0]
            for _ in range(pop_size)
        ]
        new_pop = []
        for i in range(0, pop_size, 2):
            c1, c2 = crossover(selected[i], selected[i+1])
            new_pop += [mutate(c1, mutation_rate), mutate(c2, mutation_rate)]
        population = new_pop

    return best, history

"""# Запуск обучения

---

Главная функция, реализующая цикл генетического алгоритма для оптимизации гиперпараметров нейросетей.

На каждом поколении:
* Оцениваются все особи текущей популяции: каждая модель обучается и тестируется;
* В консоль выводятся параметры и точность каждой особи (включая тип модели);
* Обновляется глобальный лучший результат, если найдена более точная модель;
* Сохраняются максимальные точности по каждой архитектуре (MLP, CNN, RNN) для построения графика прогресса;
* С помощью турнира отбираются кандидаты, которые дают начало следующему поколению;
* Применяются операции кроссовера и мутации для формирования новой популяции.

По завершении:
* Распечатываются параметры и точность лучшей найденной модели с поясняющими подписями;
* Результаты сохраняются в файл ga_wine_history.csv для последующего анализа и визуализации;
* Обеспечивается наглядное и удобное представление итогов оптимизации.
"""

POP_SIZE, GENS, MUT_RATE = 10, 6, 0.3

(best_ind, best_acc), history = genetic_algorithm(POP_SIZE, GENS, MUT_RATE)

# Распаковка параметров лучшей особи
final_model, final_lr, final_layers, final_units, final_dropout = best_ind

print("\n=== Финал ===")
print("Лучшая модель:")
print(f"  Тип архитектуры:     {final_model.upper()}")
print(f"  Learning rate:       {final_lr:.5f}")
print(f"  Скрытых слоёв:       {final_layers}")
print(f"  Нейронов/фильтров:   {final_units}")
print(f"  Dropout:             {final_dropout:.2f}")
print(f"  Accuracy на тесте:   {best_acc:.4f}")

# История в DataFrame
df = pd.DataFrame(history)
df.to_csv('ga_wine_history.csv', index=False)
print("\nИстория сохранена в файл: ga_wine_history.csv")

"""# Отрисовка прогресса обучения лучших моделей

---

График, отображающий изменение максимальной точности по поколениям отдельно для каждой архитектуры — MLP, CNN и RNN.
Это позволяет наглядно сравнить, какая модель стабильно показывает лучшие результаты в процессе эволюции.
"""

plt.plot(df['gen'], df['mlp'], marker='o', label='MLP')
plt.plot(df['gen'], df['cnn'], marker='s', label='CNN')
plt.plot(df['gen'], df['rnn'], marker='^', label='RNN')
plt.xlabel('Поколение')
plt.ylabel('Лучший Accuracy')
plt.title('Прогресс по типам моделей')
plt.legend()
plt.grid(True)
plt.show()

"""# Восстановление и сохранение лучшей модели

---

После завершения работы генетического алгоритма, здесь по найденным оптимальным гиперпараметрам создаётся соответствующая модель (MLP, CNN или RNN).
Она повторно обучается на всей обучающей выборке для достижения максимального качества и затем сохраняется в файл best_wine_model.keras для дальнейшего использования или развёртывания.
"""

model_type, lr, n_layers, n_units, drop = best_ind

best_model = Sequential()
if model_type == 'mlp':
    best_model.add(Input(shape=(13,)))
    for _ in range(n_layers):
        best_model.add(Dense(n_units, activation='relu'))
        best_model.add(Dropout(drop))
    best_model.add(Dense(num_classes, activation='softmax'))
    x_tr_full = X_train_mlp
elif model_type == 'cnn':
    best_model.add(Input(shape=(13,1)))
    for _ in range(n_layers):
        best_model.add(Conv1D(n_units, 3, padding='same', activation='relu'))
        best_model.add(MaxPooling1D(2))
    best_model.add(Flatten())
    best_model.add(Dropout(drop))
    best_model.add(Dense(num_classes, activation='softmax'))
    x_tr_full = X_train_seq
else:  # rnn
    best_model.add(Input(shape=(13,1)))
    for i in range(n_layers):
        rs = (i < n_layers - 1)
        best_model.add(LSTM(n_units, return_sequences=rs))
        best_model.add(Dropout(drop))
    best_model.add(Dense(num_classes, activation='softmax'))
    x_tr_full = X_train_seq

best_model.compile(
    optimizer=Adam(learning_rate=lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Дообучение на полной тренировочной выборке
best_model.fit(x_tr_full, y_train,
               epochs=20, batch_size=32, verbose=1)

# Сохранение модели
best_model.save('best_wine_model.keras')
print("Лучшая модель сохранена в best_wine_model.keras")

"""# Итоги
---

1) Ход эволюции (6 поколений)

Поколение 1
 * Сразу наблюдается высокая точность у RNN (0.9722), MLP (0.9444) и CNN (0.9444).
 * Первая особь с точностью 0.9722 (RNN) установила лучший результат.

Поколения 2–4
 * MLP и RNN стабильно демонстрируют высокие значения (до 0.9722 и 1.0000).
 * В поколении 4 RNN достигла точности 1.0000 — это стало новым рекордом.

Поколения 5–6
 * MLP догоняет RNN, также показывая 1.0000 в нескольких особях.
 * CNN после 2 поколения полностью теряет точность (падение до 0.0), вероятно, из-за плохих гиперпараметров, выбранных при кроссовере и мутации.

2) Визуализация прогресса по типам моделей
* MLP: уверенно росла и стабилизировалась на уровне 1.0000.
* RNN: лидировала с самого начала и удерживала высокую точность.
* CNN: после второго поколения полностью «выпала» — признак того, что комбинации параметров оказались неудачными и не восстанавливались.

3) Финальные параметры лучшей модели:

| Параметр            | Значение |
| ------------------- | -------- |
| **Модель**          | RNN      |
| **Learning rate**   | 0.04232  |
| **Скрытых слоёв**   | 2        |
| **Нейронов**        | 35       |
| **Dropout**         | 0.18     |
| **Точность (test)** | 1.0000   |

4) Дообучение лучшей модели
После выбора лучшей конфигурации, модель была дообучена на всех тренировочных данных в течение 20 эпох.

Прогресс
* Accuracy вырос с ~60% до 99.8%
* Loss снизился до 0.0176
* Итоговая точность после 20 эпох — 94.15% (на части тренировочных данных)

5) Выводы
* Генетический алгоритм эффективно отобрал лучшие архитектуры: MLP и RNN достигли идеальной точности.
* CNN требует дополнительной настройки: в этом запуске она не смогла восстановиться после деградации в 3-м поколении.
* RNN оказалась финальным победителем, но MLP показала сравнимую стабильность, что может указывать на её эффективность при меньшей сложности.
* Графики и история обучения дали полную картину эволюции — это удобно как для анализа, так и для отчётности.
"""