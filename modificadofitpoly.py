import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


dados = pd.read_csv('C:/Users/maluu/Downloads/ORB04_EUR_EPHIO_SPACECRAFT_EVENT_TIME.csv', sep=',')

col = dados.columns
dados.set_index(pd.to_datetime(dados[col[0]]), inplace=True)

dados['R'] = np.sqrt(dados['X'].values**2 + dados['Y'].values**2 + dados['Z'].values**2)

dados['mask'] = 'Próximo'

dados.loc[dados['R'] > 3.2, 'mask'] = 'Longe'



dados_longe = dados[ dados['mask'] == 'Longe' ]

x = dados_longe['X'].values.reshape(-1, 1)
y = dados_longe['BX'].values

poly = PolynomialFeatures(degree=4, include_bias=True)
x_trans = poly.fit_transform(x)

lr = LinearRegression()
lr.fit(x_trans, y)

X_ = dados['X'].values
X_ = X_.reshape(X_.shape[0],1)

X_new_poly = poly.transform(X_)
y_new  = lr.predict(X_new_poly)

dados['fit_bx'] = y_new
plt.figure()
sn.scatterplot(x='X', y='BX', hue='mask', data=dados)
sn.lineplot(x='X', y='fit_bx', data=dados)
dados['BX_Eu'] = dados['BX'] - dados['fit_bx']

#plt.figure()
dados.plot(y=['BX_Eu'])