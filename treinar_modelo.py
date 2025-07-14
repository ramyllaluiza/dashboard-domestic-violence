# treinar_modelo.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Carregar os dados tratados
try:
    df = pd.read_csv('dados_tratados.csv')
    print("Arquivo 'dados_tratados.csv' carregado com sucesso.")
except FileNotFoundError:
    print("Erro: 'dados_tratados.csv' não encontrado. Certifique-se de que seu script de pré-processamento foi executado e salvou o arquivo.")
    exit()

# 2. Definir Features (X) e Target (y)
features = ['age', 'education', 'employment', 'income', 'marital_status']
target = 'violence'

X = df[features]
y = df[target]

# 3. Tratamento de Dados Ausentes no Target
# Remove linhas onde 'violence' é None (pois não podemos treinar com um target desconhecido)
df_clean = df.dropna(subset=[target])
X = df_clean[features]
y = df_clean[target]

# Converter 'violence' para int (True=1, False=0)
y = y.astype(int)

# 4. Dividir Dados em Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nDados divididos: Treino={len(X_train)} amostras, Teste={len(X_test)} amostras.")

# 5. Treinar o Modelo
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("\nModelo RandomForestClassifier treinado com sucesso.")

# 6. Avaliar o Modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do Modelo no conjunto de teste: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Não Violência', 'Violência']))

# 7. Salvar o Modelo Treinado
model_filename = 'modelo_classificacao_violencia.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"\nModelo salvo como '{model_filename}'")

# Salvar a lista de features (colunas) usadas para treinar o modelo
features_filename = 'features_modelo.pkl'
with open(features_filename, 'wb') as file:
    pickle.dump(features, file)
print(f"Lista de features salva como '{features_filename}'")