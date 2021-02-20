# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:20:49 2021

@author: great
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('white') #vamos mudar o estilo

df = pd.read_csv('C:/Users/great/Desktop/py4e/Plotting/inv.csv')

df= df.rename(columns = {'Invest. Esp' : 'INVEST'})
df.ANO = pd.to_numeric(df.ANO , errors='coerce')
df.INVEST = pd.to_numeric(df.INVEST , errors='coerce')

df = df.set_index('ANO')
print(df.head())
print(df.columns)

titles = ['Criminalidade ao longo dos anos', 'Investimento ao longo dos anos']
print(len(df.columns))

df.plot(kind = 'line', subplots = True, grid = True, title = 'Dados por Ano', 
        layout = (3,1), sharex = True, legend = True, 
        colormap = 'cividis');

df = df.reset_index()
dfc = df.copy()
del dfc['ANO']


dfc = dfc.set_index('INVEST')
dfc = dfc.sort_index()
dfc = dfc.reset_index()

dfc.plot.line(y = 'Crim/100mil', x = 'INVEST',
             title = 'A influência do investimento em esportes em Maringá/PR na criminalidade anual.',
             ylabel = 'Criminalidade a cada  100mil Habitantes',
             grid = True,
             colormap = 'viridis'
             );

dfc.plot.line(y = 'Bolsas_Atleta', x = 'INVEST',
             title = 'A influência do investimento em esportes em Maringá/PR na quantidade de bolsas atleta fornecidas',
             ylabel = 'Numero de Bolsas Atleta fornecidas',
             grid = True,  
             colormap = 'plasma'
             );












