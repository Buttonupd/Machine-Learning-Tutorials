#KNN

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv('car.data')

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
persons = le.fit_transform(list(data["persons"]))
door = le.fit_transform(list(data["door"]))
cls = le.fit_transform(list(data["class"]))
safety = le.fit_transform(list(data["safety"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))

predict = "class"

X = list(zip(lug_boot, safety, door, maint, persons, buying))

y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

predicted = model.predict(x_test)
names = ["unacc", "acc", "vgood", "good"]

for x in range(len(predicted)):
    print("Predicted : ", names[predicted[x]], "Data : ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N : ", n)





















