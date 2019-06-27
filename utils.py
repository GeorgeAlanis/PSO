import numpy as np
import pandas as pd
from sklearn import preprocessing


def euclidean(p1, p2):
    a = (p1[0] - p2[0]) * (p1[0] - p2[0])
    b = (p1[1] - p2[1]) * (p1[1] - p2[1])
    return a + b


def get_data(file_path='assets/Sprint7ToroideMixto.csv'):
    data = pd.read_csv(file_path)
    data['tiempo_en_tienda'] = data['demanda'] * data['frecuencia']

    # Create x, where x the 'scores' column's values as floats
    x = data[['lat']].values.astype(float)
    y = data[['lon']].values.astype(float)
    z = data[['tiempo_en_tienda']].values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)
    y_scaled = min_max_scaler.fit_transform(y)
    z_scaled = min_max_scaler.fit_transform(z)

    # Run the normalizer on the dataframe
    #df_normalized = pd.DataFrame(x_scaled)
    stores = np.array(
        list(
            zip(
                x_scaled, y_scaled, z_scaled, np.array([None]*len(data.lat.values))
            )
        )
    )
    return stores


def get_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

