import numpy as np
import skfuzzy as fuzz 
from skfuzzy import control as ctrl
import networkx as nx
import matplotlib.pyplot as plt


#variaveis de entrada
calorias = ctrl.Antecedent(np.arange(0,11,1),'calorias') #1000 kcal

#variaveis de saida
peso = ctrl.Consequent(np.arange(0,12,1),'peso')

#funcao de pertinencia trapezoidal 
# calorias['pouco'] = fuzz.trapmf(calorias.universe,[-1,0,2,4])
# calorias['razoavel'] = fuzz.trapmf(calorias.universe,[2,4,6,8])
# calorias['bastante'] = fuzz.trapmf(calorias.universe,[4,6,10,15])
# # #calorias.view()
# peso['leve'] = fuzz.trapmf(peso.universe,[-1,0,4,6])
# peso['medio'] = fuzz.trapmf(peso.universe,[4,6,8,10])
# peso['pesado'] = fuzz.trapmf(peso.universe,[8,10,12,13])
#peso.view()


#funcao de pertinencia triangular 
# calorias['pouco'] = fuzz.trimf(calorias.universe,[-1,1,4])
# calorias['razoavel'] = fuzz.trimf(calorias.universe,[2,5,8])
# calorias['bastante'] = fuzz.trimf(calorias.universe,[4,8,15])

# peso['leve'] = fuzz.trimf(peso.universe,[-1,2,6])
# peso['medio'] = fuzz.trimf(peso.universe,[4,7,10])
# peso['pesado'] = fuzz.trimf(peso.universe,[8,11,13])


#funcao de pertinencia gaussiana
calorias['pouco'] = fuzz.gaussmf(calorias.universe, 1,2)
calorias['razoavel'] = fuzz.gaussmf(calorias.universe, 5,2)
calorias['bastante'] = fuzz.gaussmf(calorias.universe, 8,2)
peso['leve'] = fuzz.gaussmf(peso.universe, 2,2)
peso['medio'] = fuzz.gaussmf(peso.universe, 7,2)
peso['pesado'] = fuzz.gaussmf(peso.universe, 11,2)


#criando as regras
regra1 = ctrl.Rule(calorias['pouco'],peso['leve'])
regra2 = ctrl.Rule(calorias['razoavel'],peso['medio'])
regra3 = ctrl.Rule(calorias['bastante'],peso['pesado'])

controlador = ctrl.ControlSystem([regra1,regra2,regra3])


#simulando
calculoPeso = ctrl.ControlSystemSimulation(controlador)

calculoPeso.input['calorias'] = 4

calculoPeso.compute()

print(calculoPeso.output['peso'])
calorias.view(sim=calculoPeso)
peso.view(sim=calculoPeso)

plt.show(block=True)
