import numpy as np 
from sklearn.cluster import KMeans



def kmeans() -> KMeans:
    kmeans = KMeans(n_clusters=7,
                init='random',
                tol=1e-4, 
                random_state=170,
                verbose=True)
    
    return kmeans

def cnn():
    pass