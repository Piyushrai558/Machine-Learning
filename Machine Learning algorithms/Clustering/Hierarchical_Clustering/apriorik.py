# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 19:08:24 2018

@author: Piyush
"""
#Apriori
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

transactions = []
for i in range(0 , 7501):
	transactions.append([str(dataset.values[i,j]) for j in range(0 , 20)])


#training apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support =0.003 , min_confidence=0.2 , min_lift=3 , min_length = 2)	
	
#Visualizing the result 
result = list(rules)



