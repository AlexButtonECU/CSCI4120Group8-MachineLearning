# CSCI 4120 Homework 2
# By: Group 8

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,20))

visualizer.fit(X)
visualizer.show()
k = visualizer.elbow_value_
print(k)

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(X)
kmeans.cluster_centers_.shape

from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(10):
    mask = (clusters == i)
    labels[mask] = mode(y_true[mask], keepdims=True)[0]

print(labels.shape)
print(labels)


accuracy = accuracy_score(y_true, labels)
print("Accuracy", accuracy)

mat = confusion_matrix(y_true, labels)
#need to change 'digits.target_names' but idk what to
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()




# TODO determine the best k for k-means (Done)
# TODO calculate accuracy for best K
# TODO draw a confusion matrix