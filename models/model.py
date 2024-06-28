import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

data = pd.read_csv("Dataset/data.csv")
songClusterPipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20, verbose=False))], verbose=False)
X = data.select_dtypes(np.number)
numberCols = list(X.columns)
songClusterPipeline.fit(X)
songClusterLabels = songClusterPipeline.predict(X)
data['clusterLabel'] = songClusterLabels

with open('pipeline.pkl', 'wb') as file:
    pickle.dump(songClusterPipeline, file)

