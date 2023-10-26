#Задание 1

import numpy as np
import matplotlib.pyplot as plt

def create_random_points(num_points):
    X = np.random.random((num_points, 2))
    y = np.zeros(num_points, dtype=int)
    y[(X[:, 0] > 0.5) & (X[:, 1] > 0.5)] = 1
    y[(X[:, 0] <= 0.5) & (X[:, 1] <= 0.5)] = 1
    return X, y

def generate_random_circles_and_quadrants():
    np.random.seed()
        
    outer_radius = np.random.uniform(0.7, 1.3)
    inner_radius = np.random.uniform(0.3, 0.6)

    num_points = np.random.randint(100, 201)
    theta = np.linspace(0, 2 * np.pi, num_points)
    x_outer = outer_radius * np.cos(theta) + np.random.uniform(-0.1, 0.1, num_points)
    y_outer = outer_radius * np.sin(theta) + np.random.uniform(-0.1, 0.1, num_points)
    x_inner = inner_radius * np.cos(theta) + np.random.uniform(-0.05, 0.05, num_points)
    y_inner = inner_radius * np.sin(theta) + np.random.uniform(-0.05, 0.05, num_points)

    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    x = np.linspace(-1.5, 1.5, 100)
    parabola1_x = x
    parabola1_x = x - 1
    parabola1_y = -x**2 + 2
    parabola2_x = x
    parabola2_x = x + 1
    parabola2_y = x**2 - 2 

    axes[0].scatter(x_outer, y_outer, color='yellow', s=50)
    axes[0].scatter(x_inner, y_inner, color='purple', s=50)
    axes[0].grid(True)
    axes[0].axis('equal')

    n2 = np.random.randint(100, 201)
    X2, y2 = create_random_points(n2)

    axes[1].scatter(X2[y2 == 0][:, 0], X2[y2 == 0][:, 1], color='yellow', s=50)
    axes[1].scatter(X2[y2 == 1][:, 0], X2[y2 == 1][:, 1], color='purple', s=50)
    axes[1].grid(True)
    axes[1].axis('equal')

    axes[2].scatter(parabola1_x, parabola1_y, color='purple', s=5)
    axes[2].scatter(parabola2_x, parabola2_y, color='yellow', s=5)
    axes[2].grid(True)
    axes[2].axis('equal')

    plt.tight_layout()
    plt.show()
generate_random_circles_and_quadrants()

#Задание 2

import numpy as np

# Истинные значения и предсказанные значения
y_true = np.array(['Cat', 'Cat', 'Cat', 'Cat', 'Cat', 'Cat', 'Fish', 'Fish', 'Fish', 'Fish', 'Fish', 'Fish', 'Fish', 'Fish', 'Fish', 'Fish', 'Hen', 'Hen', 'Hen', 'Hen', 'Hen', 'Hen', 'Hen', 'Hen', 'Hen'])
y_pred = np.array(['Cat', 'Cat', 'Cat', 'Cat', 'Hen', 'Fish', 'Cat', 'Cat', 'Cat', 'Cat', 'Cat', 'Cat', 'Fish', 'Fish', 'Fish', 'Fish', 'Cat', 'Cat', 'Cat', 'Hen', 'Hen', 'Hen', 'Hen', 'Hen', 'Hen'])

# Создание матрицы ошибок
unique_labels = np.unique(y_true)
num_classes = len(unique_labels)
confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

for i in range(len(y_true)):
    true_class = y_true[i]
    pred_class = y_pred[i]
    true_class_index = np.where(unique_labels == true_class)[0][0]
    pred_class_index = np.where(unique_labels == pred_class)[0][0]
    confusion_matrix[true_class_index][pred_class_index] += 1

# Вычисление метрик precision, recall и f1-score для каждого класса
precision = np.zeros(num_classes)
recall = np.zeros(num_classes)
f1_score = np.zeros(num_classes)
for i in range(num_classes):
    true_positives = confusion_matrix[i][i]
    false_positives = np.sum(confusion_matrix[:, i]) - true_positives
    false_negatives = np.sum(confusion_matrix[i, :]) - true_positives
    
    precision[i] = true_positives / (true_positives + false_positives)
    recall[i] = true_positives / (true_positives + false_negatives)
    f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

# Вычисление accuracy
accuracy = np.trace(confusion_matrix) / len(y_true)

# Вычисление макро-усредненных (macro avg) значений
macro_avg_precision = np.mean(precision)
macro_avg_recall = np.mean(recall)
macro_avg_f1_score = np.mean(f1_score)

# Вычисление взвешенных (weighted avg) значений
weights = np.sum(confusion_matrix, axis=1) / len(y_true)
weighted_avg_precision = np.sum(precision * weights)
weighted_avg_recall = np.sum(recall * weights)
weighted_avg_f1_score = np.sum(f1_score * weights)

# Вывод результатов
print(confusion_matrix)
print("{:<15} {:<15} {:<15} {:<15}".format("", "precision", "recall", "f1-score"))
for i, cl in enumerate(unique_labels):
    print("{:<15} {:.3f}          {:.3f}         {:.3f}".format(cl, precision[i], recall[i], f1_score[i]))
print("\n")
print("{:<15} {:.3f}".format("accuracy", accuracy))
print("{:<15} {:.3f}          {:.3f}         {:.3f}".format("macro avg", macro_avg_precision, macro_avg_recall, macro_avg_f1_score))
print("{:<15} {:.3f}          {:.3f}         {:.3f}".format("weighted avg", weighted_avg_precision, weighted_avg_recall, weighted_avg_f1_score))

#Задание 3


import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv("artists.csv")  # Замените на путь к файлу с данными

# Выбор признаков и целевой переменной
X = data[['Followers', 'Popularity', 'Acousticness']]
y = data['Streams']

# Разбиваем данные на обучающий и тестовый наборы (например, 80% обучающих данных и 20% тестовых данных)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Обучение и оценка модели для переменной 'Followers'
model_followers = LinearRegression()
model_followers.fit(X_train[['Followers']], y_train)
y_pred_followers_train = model_followers.predict(X_train[['Followers']])
y_pred_followers_test = model_followers.predict(X_test[['Followers']])
r2_followers_train = r2_score(y_train, y_pred_followers_train)
r2_followers_test = r2_score(y_test, y_pred_followers_test)

# Обучение и оценка модели для переменной 'Popularity'
model_popularity = LinearRegression()
model_popularity.fit(X_train[['Popularity']], y_train)
y_pred_popularity_train = model_popularity.predict(X_train[['Popularity']])
y_pred_popularity_test = model_popularity.predict(X_test[['Popularity']])
r2_popularity_train = r2_score(y_train, y_pred_popularity_train)
r2_popularity_test = r2_score(y_test, y_pred_popularity_test)

# Обучение и оценка модели для переменной 'Acousticness'
model_acousticness = LinearRegression()
model_acousticness.fit(X_train[['Acousticness']], y_train)
y_pred_acousticness_train = model_acousticness.predict(X_train[['Acousticness']])
y_pred_acousticness_test = model_acousticness.predict(X_test[['Acousticness']])
r2_acousticness_train = r2_score(y_train, y_pred_acousticness_train)
r2_acousticness_test = r2_score(y_test, y_pred_acousticness_test)

# Выводим результаты
print("R^2 для Followers (обучение):", r2_followers_train)
print("R^2 для Followers (тест):", r2_followers_test)
print("R^2 для Popularity (обучение):", r2_popularity_train)
print("R^2 для Popularity (тест):", r2_popularity_test)
print("R^2 для Acousticness (обучение):", r2_acousticness_train)
print("R^2 для Acousticness (тест):", r2_acousticness_test)
