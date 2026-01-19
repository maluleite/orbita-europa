import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.metrics import r2_score

#from scipy.optimize import curve_fit

from symfit import variables, parameters, Model, Fit, sin, cos
#from symfit.core.minimizers import DifferentialEvolution

dados = pd.read_csv(r'C:\Users\malú\orbitasephio\ORB04_EUR_EPHIO_SPACECRAFT_EVENT_TIME.csv', sep=',')
col = dados.columns

dados.set_index(pd.to_datetime(dados[col[0]]), inplace=True)

RE = 1560. # KM
mo4PI = 1e-10 / RE
R_G = 180./np.pi

distancia_corte = 3.9

#dados['X'] *= RE
#dados['Y'] *= RE
#dados['Z'] *= RE

dados['R'] = np.sqrt( dados['X'].values**2 + 
        dados['Y'].values**2 + dados['Z'].values**2)

dados['RHO'] = np.sqrt( dados['X'].values**2 + dados['Y'].values**2)

dados['THETA'] = np.arctan2( dados['RHO'].values, dados['Z'].values )
dados['LAT'] = 0.5*np.pi - dados['THETA']

dados['LON'] = np.arctan2(dados['Y'].values, dados['X'].values)

#dados['LON'].loc[dados['LON'] < 0 ] += 2*np.pi

# Mostrando as figuras para a cada fit
time = pd.to_datetime(dados[col[0]])


plt.rcParams['font.size'] = 11
fig, ax = plt.subplots(3,1, figsize=[12,6], sharex=True)
fig.suptitle("Campo magnético medido pela Galileo na proximidade de Europa (órbita 04)")

line1, = ax[0].plot(time, dados['LAT']*R_G, label='THETA')
#line2, = ax[0].plot(time, dados['fit_bx'], label='Bx fit')
#ax[0].legend(handles=[line1, line2])
ax[0].set_ylabel('Latitude [°]')

line1, = ax[1].plot(time, dados['LON']*R_G, label='LON')
#line2, = ax[1].plot(time, dados['fit_by'], label='By fit')
#ax[1].legend(handles=[line1, line2])
ax[1].set_ylabel('Longitude [°]')

line1, = ax[2].plot(time, dados['R'], label='R')
#line2, = ax[2].plot(time, dados['fit_bz'], label='Bz fit')
#ax[2].legend(handles=[line1, line2])
ax[2].set_xlabel('Tempo')
ax[2].set_ylabel('$Distância [R_e]$')


dados['mask'] = "Próximo"

dados.loc[dados['R'] > distancia_corte, 'mask'] = "Longe"

dados_longe = dados[ dados['mask'] == "Longe" ]

# para a componente X
x = dados_longe.index.values.reshape(-1, 1)
y = dados_longe['BX'].values

poly = PolynomialFeatures(degree=6, include_bias=True)
x_trans = poly.fit_transform(x)

lr = LinearRegression()
lr.fit(x_trans, y)

X_ = dados.index.values
X_ = X_.reshape(X_.shape[0],1)

X_new_poly = poly.transform(X_)
y_new  = lr.predict(X_new_poly)

dados['fit_bx'] = y_new

dados['BX_Eu'] = dados['BX'] - dados['fit_bx']

# para a componente Y
y = dados_longe['BY'].values

poly = PolynomialFeatures(degree=6, include_bias=False)
x_trans = poly.fit_transform(x)

lr = LinearRegression()
lr.fit(x_trans, y)

X_new_poly = poly.transform(X_)
y_new  = lr.predict(X_new_poly)

dados['fit_by'] = y_new

dados['BY_Eu'] = dados['BY'] - dados['fit_by']

# para a componente Z
y = dados_longe['BZ'].values

poly = PolynomialFeatures(degree=6, include_bias=True)
x_trans = poly.fit_transform(x)

lr = LinearRegression()
lr.fit(x_trans, y)

X_new_poly = poly.transform(X_)
y_new  = lr.predict(X_new_poly)

dados['fit_bz'] = y_new

dados['BZ_Eu'] = dados['BZ'] - dados['fit_bz']

# campo total
BT = np.sqrt(dados['BX'].values**2 + dados['BY'].values**2 + dados['BZ'].values**2)
fitT = np.sqrt(dados['fit_bx'].values**2 + dados['fit_by'].values**2 + dados['fit_bz'].values**2)

dados['BT'] = BT
dados['BT_Eu'] = BT - fitT

# Mostrando as figuras para a cada fit
time = pd.to_datetime(dados[col[0]])

dados_longe = dados[ dados['mask'] == "Longe" ]
time_longe = pd.to_datetime(dados_longe[col[0]])

r, te, fi, bx, by, bz = variables('r, te, fi, bx, by, bz')
mx, my, mz = parameters('mx, my, mz')

mx.value = 153
my.value = 269
mz.value = -496

model = Model({
    bx: (3.*(mx*sin(te)*cos(fi) + my*sin(te)*sin(fi) + mz*cos(te) )*sin(te)*cos(fi) - mx) / r**3, 
    by: (3.*(mx*sin(te)*cos(fi) + my*sin(te)*sin(fi) + mz*cos(te) )*sin(te)*sin(fi) - my) / r**3, 
    bz: (3.*(mx*sin(te)*cos(fi) + my*sin(te)*sin(fi) + mz*cos(te) )*cos(te) - mz) / r**3
})

dados_P = dados[ dados['mask'] == "Próximo" ]
#sigma = dados_P['BX_Eu'].values * 0 + 0.5

fit = Fit(model, r=dados_P['R'].values, te=dados_P['THETA'].values,
          fi=dados_P['LON'].values,
          bx=dados_P['BX_Eu'].values, by=dados_P['BY_Eu'].values, 
          bz=dados_P['BZ_Eu'].values)

fit_result = fit.execute()

print(fit_result) 
print('mx = ', fit_result.value(mx) ) #* 1e3 / 1560.
print('my = ', fit_result.value(my) )
print('mz = ', fit_result.value(mz) )

dados_P = dados#[ dados['mask'] == "Próximo" ]

bv = model(r=dados['R'].values, te=dados['THETA'].values,
          fi=dados['LON'].values, mx=fit_result.value(mx), 
          my=fit_result.value(my),  mz=fit_result.value(mz))

dados['BX_mod'] = bv.bx
dados['BY_mod'] = bv.by
dados['BZ_mod'] = bv.bz

dados['Bx_FIT'] = dados['BX_mod'] + dados['fit_bx']
dados['By_FIT'] = dados['BY_mod'] + dados['fit_by']
dados['Bz_FIT'] = dados['BZ_mod'] + dados['fit_bz']

dados['Bx_Longe'] = dados['BX']
dados['By_Longe'] = dados['BY']
dados['Bz_Longe'] = dados['BZ']

dados['Bx_Longe'].loc[ dados['mask'] == "Próximo" ] = np.nan
dados['By_Longe'].loc[ dados['mask'] == "Próximo" ] = np.nan
dados['Bz_Longe'].loc[ dados['mask'] == "Próximo" ] = np.nan

fig, ax = plt.subplots(3,1, figsize=[12,6], sharex=True)
fig.suptitle("Campo Magnetico medido pela Galileo na proximidade de Europa")

plt.rcParams['font.size'] = 13  # Sets global font size to 14
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

line1, = ax[0].plot(time, dados['BX'], label='medido')
line4, = ax[0].plot(time, dados['Bx_Longe'], label='Corte')
line2, = ax[0].plot(time, dados['fit_bx'], label='Ajuste')
line3, = ax[0].plot(time, dados['Bx_FIT'], label='Modelado')
#ax[0].legend(handles=[line1, line4, line2, line3], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
ax[0].set_ylabel('Bx[nT]')
#ax[0].add_axes()


line1, = ax[1].plot(time, dados['BY'], label='Medido')
line4, = ax[1].plot(time, dados['By_Longe'], label='Corte')
line2, = ax[1].plot(time, dados['fit_by'], label='Ajuste')
line3, = ax[1].plot(time, dados['By_FIT'], label='Modelado')
ax[1].legend(handles=[line1, line4, line2, line3], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
ax[1].set_ylabel('By[nT]')

line1, = ax[2].plot(time, dados['BZ'], label='medido')
line4, = ax[2].plot(time, dados['Bz_Longe'], label='Corte')
line2, = ax[2].plot(time, dados['fit_bz'], label='ajustado')
line3, = ax[2].plot(time, dados['Bz_FIT'], label='modelado')
#ax[2].legend(handles=[line1, line4, line2, line3], bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
ax[2].set_ylabel('Bz[nT]')
ax[2].set_xlabel('Tempo')

fig.savefig('Campo_magnetico.png', dpi=350)