from pandas import read_csv, DataFrame
import numpy as np
from os import getcwd
import matplotlib.pyplot as plt
from seaborn import set as sns_set
from seaborn import pairplot as sns_pairplot
from seaborn import violinplot as sns_violinplot

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

from lineal_tools import get_error_model, plot_twenty_points
from lineal_tools import train_model, print_alpha_errors


current_path = getcwd()

# read the archive
archive = read_csv(
    current_path + "/db/airfoil_self_noise.dat", 
    sep="\t",
    header=None
    )

# how the columns are in the archive
columns = [
    "frequency",
    "angle",
    "chord",
    "velocity",
    "suction",
    "sound"
    ]

# set the columns
archive.columns = columns
x = archive.iloc[:, :5]
y = archive.iloc[:, 5]

# split the data in train and test
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=1
    )


alphas = np.arange(0.001, 0.1, 0.001)

train_data = [x_train, y_train]
test_data = [x_test, y_test]

train_model(LinearRegression, train_data, test_data)
train_model(Lasso, train_data, test_data)
train_model(Ridge, train_data, test_data)

print_alpha_errors(Lasso, train_data, test_data, alphas)
print_alpha_errors(Ridge, train_data, test_data, alphas)
