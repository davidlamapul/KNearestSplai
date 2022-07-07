from sklearn import datasets, neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import numpy
import pandas


def iris():
    iris = datasets.load_iris()

    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2020)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print("TEST:", round(clf.score(X_test, y_test), 2))


def asteroids():
    data = pandas.read_csv('data/neo_v2.csv')

    data.drop(columns=['id', 'name', 'orbiting_body', 'sentry_object'], inplace=True)
    y = data['hazardous']
    X = data.drop(columns=['hazardous'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2020)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    print("TEST:", round(clf.score(X_test, y_test), 2))

asteroids()