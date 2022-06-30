
import numpy as np
from sklearn.cluster import KMeans


def selection(img):

kmeans_model = KMeans(n_clusters=3, random_state=1).fit(img)
# labels = kmeans_model.labels_
# KMeans.metrics.silhouette_score(img, labels, metric='euclidean')
