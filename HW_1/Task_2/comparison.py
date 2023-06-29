import numpy as np
import pandas as pd

data = pd.read_csv('AB_NYC_2019.csv')
mean = np.mean(data["price"])
print(mean)

var = np.var(data[["price"]])
print(var)
