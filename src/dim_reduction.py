import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D
 
def load_agglo(c: int):
    n = np.load("../cluster/agglomorative/{}_clus.npz".format(c), allow_pickle=True)
    return n['X'], n['Y'] 

def load_gmm(c: int):
    n = np.load("../cluster/gaussian/{}_clus.npz".format(c), allow_pickle=True)
    return n['X'], n['Y'] 

def load_Unlabeled():
    n = np.load("../cluster/gaussian/{}_clus.npz".format(3), allow_pickle=True)
    return n['X']
    
class dataset():
    def __init__(self,class_num=3, cluster = "agglomorative"):
        # init 3d figure
        self.number_of_classes = class_num
        self.color_list = ['b', 'g', 'r', 'y', 'black', 'c', 'k', 'w']
        
        self.classes = []
        
        for j in range(self.number_of_classes):
            self.classes.append([])
        
        match cluster:
            case "agglomorative":
                self.X, self.Y = load_agglo(self.number_of_classes)
            case "gaussian":
                self.X, self.Y = load_gmm(self.number_of_classes)
            case "NoLabeled":
                self.X = load_Unlabeled()
            case _:
                Exception("Not defined dataset has been input. Exiting the program.")
                exit(-1)
        
        self.fig = plt.figure(figsize=(9, 6))
        # self.ax = self.fig.add_subplot(111, projection='3d')        
                
                
        self.trans_X = None
        self.trans_Y = None
        
    def add_plot(self, n_row, n_col,iter,max_img):
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def graph_show(self):
        nums = [elem for elem in range(self.components)]
        comb = list(combinations(nums, 3))
        
        set_even = self.components+1 if self.components % 2 == 1 else self.components
        
        print(comb)
        
        nrows = 2
        ncol = set_even // 2

        for i, idxs in enumerate(comb):
            idxs = list(idxs)
            for n in range(self.number_of_classes):
                self.ax.scatter([row[idxs[0]] for row in self.classes[n]], [row[idxs[1]] for row in self.classes[n]], [row[idxs[2]] for row in self.classes[n]], color=self.color_list[n])
        plt.show()
        
        
    def LDA(self, components, write = False):
        self.components = components
        Model = LinearDiscriminantAnalysis(n_components=components)
        self.trans_X = Model.fit_transform(self.X, self.Y)
        # print(Model.coef_)
            
        for i in range(len(self.Y)):
            self.classes[self.Y[i]].append(list(self.trans_X[i]))
            
        self.graph_show()
        
        print(self.trans_X.shape)
        
        if write:
            with open('parameters.txt', mode='a') as f:
                f.write("{} # of class = {}, # of components = {}\n coefficients  = \n{}\n\n {}".format('#'*10,self.number_of_classes, self.components, list(Model.coef_), '#'*10))
        
    def Manifold_TSNE(self, components):
        self.components = components
        method = 'barnes_hut'
        if components > 3:
            method = 'exact'
        
        Model = TSNE(n_components=components, random_state=42, n_iter=1000, method=method)
        self.trans_X = np.array(Model.fit_transform(self.X))
        # print(Model.embedding_)
        
        for i in range(len(self.Y)):
            self.classes[self.Y[i]].append(list(self.trans_X[i]))
            
        self.graph_show()
        
        
t = dataset(class_num= 5,cluster="gaussian")
print("LDA start")
t.LDA(components=4, write = False)
# print("Manifold start")
# p = dataset(class_num= 7, cluster="gaussian")
# p.Manifold_TSNE(components=4)