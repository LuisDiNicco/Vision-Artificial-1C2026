import pandas as pd
from sklearn import tree
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump

df = pd.read_csv("dataset.csv")
X = df.iloc[:, :7].values
Y = df.iloc[:, -1].values

clasificador = tree.DecisionTreeClassifier().fit(X, Y)

y_pred = cross_val_predict(clasificador, X, Y, cv=5)
print("Matriz de confusión:")
print(confusion_matrix(Y, y_pred))
print("Informe de clasificación:")
print(classification_report(Y, y_pred))

dump(clasificador, "modelo.joblib")