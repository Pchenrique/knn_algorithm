import numpy as np
from numpy.core.fromnumeric import mean
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

data = np.genfromtxt("colorrectal_2_classes_formatted.txt", delimiter=",")

classes = data[:, 142]
attributes = np.delete(data,(142), axis=1)

min_max_scaler = MinMaxScaler()
attributes_norm = min_max_scaler.fit_transform(attributes)

l_acc = []
l_prec = []
l_f1 = []
l_recall = []

for i in range(10):
  x_train, x_test, y_train, y_test = train_test_split(attributes_norm, classes, test_size=0.2)

  knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")

  knn.fit(x_train, y_train)
  y_pred = knn.predict(x_test)


  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)

  l_acc.append(acc)
  l_prec.append(prec)
  l_f1.append(f1)
  l_recall.append(recall)

print("acc: ", round(mean(l_acc), 2))
print("prec: ", round(mean(l_prec), 2))
print("f1: ", round(mean(l_f1), 2))
print("recall: ", round(mean(l_recall), 2))