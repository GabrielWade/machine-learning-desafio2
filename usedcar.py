import tensorflow as tf

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/usedcars.csv')

df_enum = pd.get_dummies(df).fillna(0)
print(df_enum)

dados = df_enum.drop(columns=['price'])
alvo = df_enum['price']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
dados = scaler.fit_transform(dados)

print(dados)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dados, alvo, test_size=0.20, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(32),
    tf.keras.layers.Dense(16),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.R2Score()])

model.fit(x_train, y_train, epochs=1000)
model.evaluate(x_test, y_test)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_pred_train, color='blue', label='Dados Reais (Treinamento) x previsão')
plt.scatter(y_test, y_pred_test, color='green', label='Dados Reais (Teste)')
plt.plot(y_train, y_train, color='red', linewidth=2, label='Linha de Regressão')
plt.legend()

plt.tight_layout()
plt.show()

