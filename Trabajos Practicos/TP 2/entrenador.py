import pandas as pd
from sklearn import tree
from joblib import dump

df = pd.read_csv("dataset.csv")
X = df.iloc[:, :7].values
Y = df.iloc[:, -1].values

clasificador = tree.DecisionTreeClassifier().fit(X, Y)
dump(clasificador, "modelo.joblib")