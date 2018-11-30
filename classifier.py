import numpy as np  
import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  

def get_classifier():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign column names to the dataset
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(url, names=names)  

    # Set features (x) and labels (y)
    X = dataset.iloc[:, :-1].values  
    y = dataset.iloc[:, 4].values  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifier = KNeighborsClassifier(n_neighbors=11)  
    classifier.fit(X_train, y_train)  

    return scaler, classifier