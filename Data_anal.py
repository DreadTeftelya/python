import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Функция для вычисления оптимального числа главных компонент
def find_optimal_num_components(X):
    max_components = min(X.shape[0], X.shape[1])
    explained_variance_ratio = np.zeros(max_components)

    for num_components in range(1, max_components + 1):
        scores = ScoresPCA(X, PC=num_components)
        explained_variance_ratio[num_components - 1] = np.sum(scores ** 2) / np.sum(np.var(X**2, axis=0))

    return explained_variance_ratio

def LoadingsPCA(X, PC=None, CentWeightX=3):
    # Центрирование и/или шкалирование переменных X
    if CentWeightX == 1:  # Только центрирование
        X = X - np.mean(X, axis=0)
    elif CentWeightX == 2:  # Только шкалирование
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif CentWeightX == 3:  # Центрирование и шкалирование
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Вычисление PCA
    pca = PCA(n_components=PC)
    pca.fit(X)

    # Получение нагрузок P для калибровочного набора X
    loadings = pca.components_.T

    return loadings


def ScoresPCA(X, PC=None, CentWeightX=4, Xnew=None):
    # Центрирование и/или шкалирование переменных X
    if CentWeightX == 1:  # Только центрирование
        X = X - np.mean(X, axis=0)
    elif CentWeightX == 2:  # Только шкалирование
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    elif CentWeightX == 3:  # Центрирование и шкалирование
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Вычисление PCA
    pca = PCA(n_components=PC)
    pca.fit(X)

    # Вычисление счетов T для калибровочного набора X
    T = pca.transform(X)

    if Xnew is not None:
        # Применение модели PCA к новому набору значений Xnew
        if CentWeightX == 1:  # Только центрирование
            Xnew = Xnew - np.mean(X, axis=0)
        elif CentWeightX == 2:  # Только шкалирование
            Xnew = (Xnew - np.mean(X, axis=0)) / np.std(X, axis=0)
        elif CentWeightX == 3:  # Центрирование и шкалирование
            Xnew = (Xnew - np.mean(X, axis=0)) / np.std(X, axis=0)

        T_new = pca.transform(Xnew)
        return T, T_new

    return T


data = pd.read_csv('DATA.csv', delimiter=';')

#2.2
label_encoder = LabelEncoder()

data['student_id'] = label_encoder.fit_transform(data['student_id'])

#2.3

scaler = MinMaxScaler(feature_range=(-1, 1))

data[:] = scaler.fit_transform(data[:])
# Вычисление оптимального числа главных компонент
explained_variance_ratio = find_optimal_num_components(data)

# Построение графика объясненной дисперсии
components = np.arange(1, len(explained_variance_ratio) + 1)
plt.plot(components, explained_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()

# Вычисление PCA-счетов
T = ScoresPCA(data, PC=2)

# Построение графика счетов
plt.scatter(T[:, 0], T[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Scores Plot')
plt.grid(True)
plt.show()

# Вычисление PCA-нагрузок
loadings = LoadingsPCA(data, PC=2)

# Построение графика нагрузок
feature_names = np.arange(1, data.shape[1] + 1)
num_components = loadings.shape[1]

plt.figure(figsize=(8, 6))
for i in range(num_components):
    plt.plot(feature_names, loadings[:, i], label='PC{}'.format(i + 1))

plt.xlabel('Features')
plt.ylabel('Loadings')
plt.title('Loadings Plot')
plt.legend()
plt.grid(True)
plt.show()

data.to_csv('updated_file.csv', index=False)
