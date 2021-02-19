# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 19:20:49 2021

@author: great
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Teremos que criar 7 dataframes de duas colunas e 12 linhas

record1 = pd.Series({'Name' : 'Alice', 
                         'Class' : 'Physics',
                         'Score' : 85})
record2 = pd.Series({'Name' : 'Jack', 
                         'Class' : 'Chemistry',
                         'Score' : 82})
record3 = pd.Series({'Name' : 'Helen', 
                         'Class' : 'Biology', 
                         'Score' : 90})
''' aqui temos informações sobre 3 pessoas, com suas respectivas materias e notas
para juntar as 3 series precisamos chamar uma função chamada DataFrame, geralmente representada por df
'''
df = pd.DataFrame([record1, record2, record3], 
                  index = ['school1', 'school2', 'school3'])
''' veja que adicionamos indexes arbitrarios porque nao queriamos os padroes 0 1 2 '''
print(df)