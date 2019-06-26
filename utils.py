import numpy as np
import pandas as pd


def euclidean(p1, p2):
    return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])


def getData(file_path='assets/Sprint7ToroideMixto.csv'):
    data = pd.read_csv(file_path)
    data['tiempo_en_tienda'] = data['demanda'] * data['frecuencia']
    stores = np.array(
        list(
            zip(
                data.lat.values, data.lon.values, data.tiempo_en_tienda.values, np.zeros(len(data.lat.values))
            )
        )
    )
    return stores

