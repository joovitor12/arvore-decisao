import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# declaração de colunas do csv
columns = [
    'Wifes age',
    'Wifes education',
    'Husbands education',
    'Number of children ever born',
    'Wifes religion',
    'Wifes now working?',
    'Husbands occupation',
    'Standard-of-living index',
    'Media exposure',
    'Contraceptive method used'
]

# leitura da base de dados
df = pd.read_csv('cmc.data', header=None)
df.columns = columns

# substituir valores faltantes pela média da idade das esposas
df['Wifes age'].fillna(df['Wifes age'].mean(), inplace=True)

# Converter variáveis categóricas em numéricas
df = pd.get_dummies(df)

# Separar dados de treinamento e teste
X = df.drop('Contraceptive method used', axis=1)
y = df['Contraceptive method used']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Criar modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo com dados de treinamento
rf.fit(X_train, y_train)

# Avaliar o modelo usando dados de teste
accuracy = rf.score(X_test, y_test)
accuracy_rounded = round(accuracy, 4)
print("Acurácia arredondada do modelo: ", accuracy_rounded * 100, "%")
print("Acurácia exata do modelo: ", accuracy)
