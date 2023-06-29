from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
import argparse
from to_numpy import load_data

cluster_save_path = ""
# X = load_data()
X = None


def agglomorative_cluster(c : int):
    Y = AgglomerativeClustering(n_clusters=c).fit(X)
    np.savez("../cluster/agglomorative/{}_clus.npz".format(c), X=X, Y=Y.labels_)
 
def load_agglo(c: int):
    n = np.load("../cluster/agglomorative/{}_clus.npz".format(c), allow_pickle=True)
    return n['X'], n['Y'] 

def gmm_cluster(c : int):
    Y = GaussianMixture(n_components=c,
                        covariance_type="full",
                        random_state=100,
                        ).fit(X)
    np.savez("../cluster/gaussian/{}_clus.npz".format(c), X=X, Y=Y.predict(X))

def load_gmm(c: int):
    n = np.load("../cluster/gaussian/{}_clus.npz".format(c), allow_pickle=True)
    return n['X'], n['Y'] 
    
if __name__ == "__main__":
    # for num_c in range(3,9):
    #     agglomorative_cluster(num_c)
    #     gmm_cluster(num_c)
    x, y = load_agglo(4)
    print(x.shape, y.shape)