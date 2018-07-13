import pandas as pd
import numpy as np
from collections import Counter

def load_data():
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