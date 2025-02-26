import pandas as pd
import numpy as np
from sklearn import tree, metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder

# Carrega o ARFF
data, meta = arff.loadarff('./PartidasTenis.arff')

# Converte bytes para string
tempo_str = np.array([x.decode('utf-8') for x in data['Tempo']]).reshape(-1, 1)
temperatura_str = np.array([x.decode('utf-8') for x in data['Temperatura']]).reshape(-1, 1)
umidade_str = np.array([x.decode('utf-8') for x in data['Umidade']]).reshape(-1, 1)
vento_str = np.array([x.decode('utf-8') for x in data['Vento']]).reshape(-1, 1)
target_str = np.array([x.decode('utf-8') for x in data['Partida']])

# Encoders separados
le_tempo = LabelEncoder()
le_temperatura = LabelEncoder()
le_umidade = LabelEncoder()
le_vento = LabelEncoder()

tempo_encoded = le_tempo.fit_transform(tempo_str.ravel()).reshape(-1, 1)
temperatura_encoded = le_temperatura.fit_transform(temperatura_str.ravel()).reshape(-1, 1)
umidade_encoded = le_umidade.fit_transform(umidade_str.ravel()).reshape(-1, 1)
vento_encoded = le_vento.fit_transform(vento_str.ravel()).reshape(-1, 1)

# Concatena as features
features = np.concatenate((tempo_encoded, 
                           temperatura_encoded, 
                           umidade_encoded, 
                           vento_encoded), axis=1)

# Target (aqui deixamos como string)
target = target_str

# Cria e treina a árvore
Arvore = DecisionTreeClassifier(criterion='entropy').fit(features, target)

# Plot da árvore
plt.figure(figsize=(10, 6.5))
tree.plot_tree(
    Arvore,
    feature_names=['Tempo', 'Temperatura', 'Umidade', 'Vento'],
    class_names=Arvore.classes_,  # Pega as classes que o modelo reconheceu
    filled=True,
    rounded=True
)
plt.show()

# Matriz de confusão
fig, ax = plt.subplots(figsize=(25, 10))
metrics.ConfusionMatrixDisplay.from_estimator(
    Arvore,
    features,
    target,
    display_labels=Arvore.classes_,
    values_format='d',
    ax=ax
)
plt.show()
