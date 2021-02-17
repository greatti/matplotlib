# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:58:16 2021

@author: great
"""

import matplotlib.pyplot as plt
import numpy as np

#### SUBPLOTS ####

plt.figure()
plt.subplot(1,2,1) # subplot with 1 row, 2 columns, and current axis is 1st subplot axes

linear_data = np.array([1,2,3,4,5,6,7,8])

plt.plot(linear_data, '-o')

exponential_data = linear_data**2
plt.subplot(1,2,2) # subplot with 1 row, 2 columns, and current axis is 2nd subplot axes
plt.plot(exponential_data, '-o')

#E podemos adicionar mais graficos a um subplot

plt.subplot(1,2,1)
plt.plot(exponential_data, '-x')

# Para resolver o problema da proporção
plt.figure()
ax1 = plt.subplot(1,2,1)
plt.plot(linear_data, '-o')
#E vamos compartilhar o eixo y 
ax2 = plt.subplot(1,2,2, sharey = ax1)
plt.plot(exponential_data, '-x')

'''
vamos lembrar que antes o que acontecia é que:
    linear_data ia de 0 a 8 
    exponential_data ia de 0 a 60
isso fazia parecer que ambos tinham o mesmo "tamanho"
'''

# plt.subplot(1,2,1) == plt.subplot(121)
#plt.subplot(1,2,1, sharey = ax1) == plt.subplot(121, sharey = ax1)


#Agora vamos criar uma grade 3x3 de plots#

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7,ax8, ax9)) = plt.subplots(3,3, 
                                                                       sharex = True, 
                                                                       sharey = True)
ax5.plot(linear_data, '-')
ax9.plot(exponential_data, '-x')
# O método é um pouco dificil de entender mas o resultado é muito legal


########## HISTOGRAMAS ##########

#vamos criar uma grade 2x2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,
                                             sharex = True) 

#Não vamos usar sharey = True porque queremos ver com igual qualidade todos os histogramas
#Se ligarmos sharey - True veremos, por exemplo, o primeiro histograma muito pequeno

axs = [ax1, ax2, ax3, ax4]

for n in range(0, len(axs)): 
    sample_size = 10**(n+1)
    sample = np.random.normal(loc = 0.0, scale = 1.0, size = sample_size)
    axs[n].hist(sample)
    axs[n].set_title('n={}'.format(sample_size))
    
''' Veja que o estudo foi feito usando apenas com 10 barras, mas podemos
aumentar esse numero para facilitar a visualização da normal '''

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,
                                             sharex = True)
axs = [ax1, ax2, ax3, ax4]
for n in range(0, len(axs)): 
    sample_size = 10**(n+1)
    sample = np.random.normal(loc = 0.0, scale = 1.0, size = sample_size)
    axs[n].hist(sample, bins = 100)
    axs[n].set_title('n={}'.format(sample_size))

''' veja como a gaussiana é muito mais suave e perceptivel '''

#Vamos fazer uma distribuição usando SCATTER

plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
plt.scatter(X,Y)

#Vamos montar uma grade
import matplotlib.gridspec as gridspec

plt.figure()
gspec = gridspec.GridSpec(3,3)

top_histogram = plt.subplot(gspec[0, 1:]) #Ele ocupa a linha 1 mas apenas a coluna 1 e 2
side_histogram = plt.subplot(gspec[1:, 0 ]) #Ele ocupa a linha 2 e 3 coluna 1
lower_right = plt.subplot(gspec[1:, 1:]) #Ele ocupa a linha 2 e 3 coluna 2 e 3

#Agora vamos preencher esses graficos com dados

Y = np.random.normal(loc = 0.0, scale = 1.0, size = 10000)
X = np.random.random(size = 10000)

lower_right.scatter(X, Y) #Um grafico de Scatter aleatorio
top_histogram.hist(X, bins = 100) #Um histograma com 100 barras aleatorio
s = side_histogram.hist(Y, bins = 100, orientation = 'horizontal') #um histograma horizontal aleatorio

#Não importa o valor de X para 'side_histogram' e Não importa o valor de Y para 'top_histogram'

#O bom é que podemos limpar um grafico e refaze-lo
top_histogram.clear()
top_histogram.hist(X, bins = 100)

side_histogram.clear()
side_histogram.hist(Y, bins = 100, orientation = 'horizontal')
side_histogram.invert_xaxis()

for ax in [top_histogram, lower_right]: 
    ax.set_xlim(0,1)
for ax in [side_histogram, lower_right]: 
    ax.set_ylim(-5, 5)
    
############## BOX PLOT ##############

import pandas as pd

normal_sample = np.random.normal(loc = 0.0, scale = 1.0, size = 10000)
random_sample = np.random.random(size = 10000)
gamma_sample = np.random.gamma(2, size = 10000)

df = pd.DataFrame({'normal' : normal_sample, 
                  'random' : random_sample, 
                  'gamma' : gamma_sample})

#e vamos observar algumas estatisticas sobre o nosso dataframe

print(df.describe())

plt.figure()
_ = plt.boxplot(df['normal'], whis = 'range') #E isso vai te dar um boxplot basico

#Vamos agora plotar todos os boxplots

plt.clf()
_ = plt.boxplot([df['normal'], df['random'], df['gamma'] ], whis = 'range')
#Vemos que dos 3 plots o df['gamma'] é o que possui um maior braço, mas porque?
#Vamos dar uma olhada no histograma

plt.figure()
_ = plt.hist(df['gamma'], bins = 100)
#Por isso! gamma é uma distribuição que possui maior concentração em valores "maiores"

#Vamos adicionar esse histograma ao boxplot?
import mpl_toolkits.axes_grid1.inset_locator as mpl_il

plt.figure()
plt.boxplot([df['normal'], df['random'], df['gamma'] ], whis = 'range')

ax2 = mpl_il.inset_axes(plt.gca(), width = '60%', height = '40%', loc = 2)
ax2.hist(df['gamma'], bins = 100)
ax2.margins(x=0.5)
ax2.yaxis.tick_right()

#Se nao dermos informações obre o parametro 'whis' o que acontece?
plt.figure()
_ = plt.boxplot([df['normal'], df['random'], df['gamma']])
#E isso pode ser um metodo para detectar OUTLINERS, por exemplo
#Ainda podemos adicionar um intervalo de confiança, temos que olhar os documentos

############### HEATMAPS ###############

#É um tipo de grafico muito bem utilizado quando queremos representar a "cocentração"
#ou "intensidade" através de cores 

plt.figure()

Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
_ = plt.hist2d(X, Y, bins=25) #e a qualidade do histograma aumenta conforme aumenta bins


plt.figure()
_ = plt.hist2d(X, Y, bins=300)

#Vamos adicionar uma legenda
plt.colorbar()  

############### ANIMATION ###############

import matplotlib.animation as animation

n = 100
x = np.random.randn(n)

#A gente vai tentar fazer a contrução animada de um histograma indo de n = 10 para n = 100

def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n: 
        a.event_source.stop() #Quando n alcançar 100 a função é interrompida
    plt.cla()
    bins = np.arange(-4, 4, 0.5) #Estamos organizando um bins variável por um passo
    plt.hist(x[:curr], bins=bins)
    plt.axis([-4,4,0,30]) 
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr), [3,27]) #estamos adicionando um "subtitulo"
    
fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100) #interval está em ms

############# INTERACTIVITY ##############

plt.figure()
data = np.random.rand(10)
plt.plot(data)

def onclick(event):
    plt.cla()
    plt.plot(data)
    plt.gca().set_title('Event at pixels {},{} \nand data {},{}'.format(event.x, event.y, event.xdata, event.ydata))

# tell mpl_connect we want to pass a 'button_press_event' into onclick when the event is detected
plt.gcf().canvas.mpl_connect('button_press_event', onclick)

#Quando clicamos no plot o titulo indica onde o mouse está

#Vamos criar um plot que selecione o dado quando clicamos em cima
from random import shuffle
origins = ['China', 'Brazil', 'India', 'USA', 'Canada', 'UK', 'Germany', 'Iraq', 'Chile', 'Mexico']

shuffle(origins)

df = pd.DataFrame({'height': np.random.rand(10),
                       'weight': np.random.rand(10),
                       'origin': origins})
#print(df)

plt.figure()
# picker=5 means the mouse doesn't have to click directly on an event, but can be up to 5 pixels away
plt.scatter(df['height'], df['weight'], picker=5)
plt.gca().set_ylabel('Weight')
plt.gca().set_xlabel('Height')

def onpick(event):
    origin = df.iloc[event.ind[0]]['origin']
    plt.gca().set_title('Selected item came from {}'.format(origin))

# tell mpl_connect we want to pass a 'pick_event' into onpick when the event is detected
plt.gcf().canvas.mpl_connect('pick_event', onpick)