# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:33:21 2021

@author: great
"""

import matplotlib as mpl
mpl.get_backend()
import matplotlib.pyplot as plt


plt.plot(3,2, '.')

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
fig = Figure()
canvas = FigureCanvasAgg(fig)
ax = fig.add_subplot(111)
ax.plot(3,2,'.')
canvas.print_png('test.png')

plt.figure() # create a new figure
plt.plot(3,2,'o') # plot the point (3,2) using the circle marker
ax = plt.gca() # get the current axes
ax.axis([0,6,0,10])  # Set axis properties [xmin, xmax, ymin, ymax]

# create a new figure
plt.figure()
# plot the point (1.5, 1.5) using the circle marker
plt.plot(1.5, 1.5, 'o')
# plot the point (2, 2) using the circle marker
plt.plot(2, 2, 'o')
# plot the point (2.5, 2.5) using the circle marker
plt.plot(2.5, 2.5, 'o')


# get current axes
ax = plt.gca()
# get all the child objects the axes contains
ax.get_children()

''' ATÉ AQUI APRENDEMOS A FAZER O BASICO DE UM PLOT '''

#############    SCATTERPLOTS

import numpy as np
x = np.array([1,2,3,4,5,6,7,8])
y = x

plt.figure()
plt.scatter(x,y)

x = np.array([1,2,3,4,5,6,7,8])
y = x
colors = ['green']*(len(x) - 1) 
colors.append('red')
# ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'red']
plt.figure()
plt.scatter(x, y, s = 100, c = colors)


# convert the two lists into a list of pairwise tuples
zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
print(list(zip_generator)) # [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]

zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
# The single star * unpacks a collection into positional arguments
print(*zip_generator) # (1, 6) (2, 7) (3, 8) (4, 9) (5, 10)

# let's turn the data back into 2 lists
zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
x, y = zip(*zip_generator)
print(x)
print(y)


plt.figure()
# plot a data series 'Tall students' in red using the first two elements of x and y
plt.scatter(x[:2], y[:2], s=100, c='red', label='Tall students')
# plot a second data series 'Short students' in blue using the last three elements of x and y 
plt.scatter(x[2:], y[2:], s=100, c='blue', label='Short students')

# add a label to the x axis
plt.xlabel('The number of times the child kicked a ball')
plt.ylabel('The grade of the student')
plt.title('Relationship between ball kicking and grades')

plt.legend(loc=4, frameon=False, title='Legend')
# add the legend to loc=4 (the lower right hand corner), also gets rid of the frame and adds a title

####### LINE PLOTS

import numpy as np
import matplotlib as mpl
mpl.get_backend()
import matplotlib.pyplot as plt

linear_data = np.array([1,2,3,4,5,6,7,8])
exponencial_data = linear_data**2

plt.figure() #criar uma figura em branco
plt.plot(linear_data, '-o', exponencial_data, '-o') #plotar ambos na mesma figura

'''
perceba que não tivemos que fornecer nenhum valor de 'x', ao inves disso, o python
entendeu que esses valores eram de 'y' e que 'x' era um index positivo igualmente
espaçados 
'''

#E se do nada quisermos plotar mais alguma coisa nessa figura?
plt.plot([22,44,55], '--r')
#E então podemos atribuir nomes aos eixos
plt.xlabel('Index')
plt.ylabel('Some data')
plt.title('A title')
plt.legend(['Baseline', 'Competition', 'Us'])

''' podemos fazer uma coisa muito daora: 
                                    preencher o espaço entre duas funções '''
plt.gca().fill_between(range(len(linear_data)),
                       linear_data, exponencial_data, 
                       facecolor = 'blue', 
                       alpha = 0.25)


#Vamos começar a trabalhar com datas!
plt.figure()
observation_dates = np.arange('2017-01-01', '2017-01-09', dtype = 'datetime64[D]')
#lembre que esse dtype faz com que ele identifique as nossas strings como datas

plt.plot(observation_dates, linear_data, '-o', observation_dates, exponencial_data, '-o')
''' Nossa mas ficou muito ruim pqp '''

import pandas as pd
'''
plt.figure()
observation_dates = np.arange('2017-01-01', '2017-01-09', dtype = 'datetime64[D]')
observation_dates = map(pd.to_datetime, observation_dates)
plt.plot(observation_dates, linear_data, '-o',
         observation_dates, exponencial_data, '-o') ##### ERROR #####
'''
''' o problema é que map() retorna um iterador, e o matplotlib não consegue ler
um iterador


por isso, precisamos primeiro converter map() em uma lista
'''

plt.figure()
observation_dates = np.arange('2017-01-01', '2017-01-09', dtype='datetime64[D]')
observation_dates = list(map(pd.to_datetime, observation_dates)) # convert the map to a list to get rid of the error
plt.plot(observation_dates, linear_data, '-o',
         observation_dates, exponencial_data, '-o')

# para melhorar esse eixo'x' podemos retirar 2017 que é comum a todos os dados
x = plt.gca().xaxis
for item in x.get_ticklabels():
    item.set_rotation(45) #rotacionar 45graus
    
plt.subplots_adjust(bottom = 0.25)#Ajustar para que as datas não saiam da tela

############ BAR CHARTS

plt.figure()

xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.3)
# plot another set of bars, adjusting the new xvals to make up for the first set of bars plotted
new_xvals = []
for item in xvals: 
    new_xvals.append(item + 0.3)

plt.bar(new_xvals, exponencial_data, width = 0.3, color = 'red')

from random import randint
linear_err = [randint(0,15) for x in range(len(linear_data))]
# This will plot a new set of bars with errorbars using the list of random error values
plt.bar(xvals, linear_data, width = 0.3, yerr=linear_err)


# stacked bar charts are also possible
plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.3, color = 'b')
plt.bar(xvals, exponencial_data, width = 0.3, bottom = linear_data, color = 'r')

#e podemos fazer a mesma coisa so que na horizontal: 

plt.figure()
xvals = range(len(linear_data))
plt.barh(xvals, linear_data, height = 0.3, color = 'b')
plt.barh(xvals, exponencial_data, height = 0.3, left = linear_data, color = 'r')

#EXEMPLO: 

plt.figure()

languages = ['Python', 'Java', 'C++', 'JS', 'SQL']
pos = np.arange(len(languages))
pop = [56, 34, 34, 29, 39]

bars = plt.bar(pos, pop, align = 'center') #para criar o grafico de barra
#vamos mudar a cor das ultimas 4 barras para cinza
for i in range (0,5):
    if i > 0:
        bars[i].set_color('#BFBCD6')
        
plt.xticks(pos, languages) #vai trocar os ticks numericos pelo nome dos elementos na lista
plt.ylabel('% Popularity')
plt.title('Top 5 Languages for Math & Data \nby % popularity on Stack Overflow', alpha = 0.8)
#Até aqui adicionamos os grafico de pos x pop, substituimos os nomes do pos pelo
#nome das linguagens, adicionamos uma desc. para o eixo Y e adicionamos um titulo

#Mas e se quisermos tirar esses tracinhos que poluem o grafico?
plt.tick_params(top = False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)
#Ou seja, a unica coisa que vamos deixar é a desc do eixo x, nao sei pq nao funcionou
#Tambem podemos remover toda a borda quadrada da imagem
for spine in plt.gca().spines.values():
    spine.set_visible(False)

    #Vamos adicionar as porcentagens às proprias barras
for bar in bars:
    plt.gca().text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, str(int(bar.get_height())) + '%', 
                 ha='center', color='w', fontsize=11)
plt.show()

