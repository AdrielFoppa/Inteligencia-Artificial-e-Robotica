import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor




print('Carregando Arquivo de teste')
arquivo = np.load('teste5.npy')
x = arquivo[0]
y = np.ravel(arquivo[1])

lista = []
menorErro  = 100

for i in range(10):
    iteracoes = 500

    regr = MLPRegressor(hidden_layer_sizes=(900,900,900,900,900),
                        max_iter=iteracoes,
                        activation='tanh', #{'identity', 'logistic', 'tanh', 'relu'},
                        solver='adam',
                        learning_rate = 'adaptive',
                        n_iter_no_change=iteracoes)
    print('Treinando RNA')
    regr = regr.fit(x,y)



    print('Preditor')
    y_est = regr.predict(x)
    print(regr.best_loss_)
    lista.append(regr.best_loss_)
    if(regr.best_loss_ < menorErro):
        menorErro = regr.best_loss_ 


plt.figure(figsize=[14,7])

#plot curso original
plt.subplot(1,3,1)
plt.plot(x,y)

#plot aprendizagem
plt.subplot(1,3,2)
plt.plot(regr.loss_curve_)


#plot regressor
plt.subplot(1,3,3)
plt.plot(x,y,linewidth=1,color='yellow')
plt.plot(x,y_est,linewidth=2)
plt.show()


media = sum(lista) / len(lista)

variancia = sum((x - media) ** 2 for x in lista) / (len(lista) - 1)
desvio_padrao = variancia ** 0.5

print("Média: ", media)
print("Desvio Padrão: ", desvio_padrao)
print("Menor erro: ", menorErro)
