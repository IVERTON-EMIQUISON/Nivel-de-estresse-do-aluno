import shap
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance


# Carregando o Dataset a partir de um arquivo local
df = pd.read_csv("./dataset/model/StressLevelDataset.csv", delimiter=';')

# Dicionário para mapear nomes de colunas em inglês para português
colunas_traduzidas = {
    'anxiety_level': 'nível de ansiedade',
    'self_esteem': 'autoestima',
    'mental_health_history': 'histórico de saúde mental',
    'depression': 'depressão',
    'headache': 'dor de cabeça',
    'blood_pressure': 'pressão sanguínea',
    'sleep_quality': 'qualidade do sono',
    'breathing_problem': 'problema de respiração',
    'noise_level': 'nível de ruído',
    'living_conditions': 'condições de vivência',
    'safety': 'segurança',
    'basic_needs': 'necessidades básicas',
    'academic_performance': 'desempenho acadêmico',
    'study_load': 'carga de estudo',
    'teacher_student_relationship': 'relação professor aluno',
    'future_career_concerns': 'preocupações com a carreira futura',
    'social_support': 'apoio social',
    'peer_pressure': 'pressão dos colegas',
    'extracurricular_activities': 'atividades extracurriculares',
    'bullying': 'bullying',
    'stress_level': 'nível de estresse'
}

# Renomeando as colunas do DataFrame
df.rename(columns=colunas_traduzidas, inplace=True)

# Verificando se as colunas estão corretas
print(df.columns)  # Verifica os nomes das colunas
print(df.head())  # Exibe as primeiras linhas do DataSet
print(df.info())  # Tipos de dados e valores nulos
print(df.describe())  # Descreve estatísticas básicas dos dados

# Removendo valores nulos
df = df.dropna()

# Codificação de variáveis categóricas
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Definição das features e target
target_column_name = 'nível de estresse'
if target_column_name not in df.columns:
    raise KeyError(f"A coluna '{target_column_name}' não foi encontrada no DataFrame.")

X = df.drop(target_column_name, axis=1).values
y = df[target_column_name].values

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizando as features para terem média 0 e desvio padrão 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Treinamento do modelo MLP
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# Previsões
y_pred = model.predict(X_test)

# Avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Imprimindo o cálculo das métricas de avaliação: acurácia, precisão, recall e F1 score
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='viridis')
plt.xlabel('Classe Predita', fontsize=14)
plt.ylabel('Classe Verdadeira', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Matriz de Confusão', fontsize=16)
plt.show()
feature_names = X.columns
#Cálculo da permutação de importância
result = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

# Extraindo as importâncias médias e seus desvios padrão
importances = result.importances_mean
std = result.importances_std

# Ordenando as importâncias
indices = np.argsort(importances)

# Plotando as importâncias
plt.figure(figsize=(12, 8))
plt.title('Importância das Features usando Permutação de Importância', fontsize=16)
plt.barh(range(len(indices)), importances[indices], xerr=std[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=12)
plt.xlabel('Redução Média na Acurácia', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.tight_layout()
plt.show()

# Ajusta os limites do eixo x
plt.xlim(min(feature_importances[indices]) * 0.9, max(feature_importances[indices]) * 1.1)

# Ajusta o layout para evitar corte das labels
plt.subplots_adjust(left=0.3, right=0.9)
plt.show()

# Selecionando as features mais importantes 
num_top_features = 5
top_features = [features[i] for i in indices[-num_top_features:]]

print(f'Top {num_top_features} features: {top_features}')

# Usando SHAP para explicar o modelo
explainer = shap.KernelExplainer(model.predict, X_test)
shap_values = explainer.shap_values(X_test)

# Visualizando a importância das features com SHAP
shap.summary_plot(shap_values, X_test, feature_names=features)

# Plote o gráfico de força para uma amostra do conjunto de teste
shap.force_plot(explainer.expected_value[1], shap_values[0][1], X_test[1].reshape(1, -1), feature_names=features)
print(f"Number of features in X_test[1]: {X_test[1].shape[0]}")
print(f"Number of SHAP values in shap_values[1][1]: {shap_values[1][1].shape[0]}")
