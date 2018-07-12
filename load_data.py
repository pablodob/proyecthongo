
# coding: utf-8

# # Project_hongo
# 
# ## Introducción
# 
# ### 1- Trabajar los datos
# transformar los datos de ingreso, a variables de ingreso.
# 
# ### 2- Diseño de la red
# Pensar en la estructura.
# 
# ### 3- Costo
# Definir la función de costo para entrenar la red.
# 
# ### 4- Definir el entrenamiento
# Manera a modificar los parametros (pesos)
# 
# ### 5- Prueba
# entrenamiento y testeo.
# 

import pandas as pd
import numpy as np
from collections import Counter

def load_data()
    # Load data
    data = pd.read_csv("mushrooms.csv")

    # Define variables
    planedata = []
    all_possible_values = []
    planedata=np.reshape(np.array(planedata),(len(data),0))

    num_values = 0


    #Load data in planedata variable
    for atr in data.columns:

        counter = Counter(data[atr])

        parcial_matrix = np.zeros((len(data),len(counter)))


        possible_values_parcial = list(counter.keys())
        all_possible_values += possible_values_parcial

        values = list(data[atr])

        n_reg=0
        for reg in values:
            index = possible_values_parcial.index(reg)
            parcial_matrix[n_reg,index] = 1
            n_reg += 1

        planedata = np.append(planedata,parcial_matrix,axis=1)

    return(planedata)