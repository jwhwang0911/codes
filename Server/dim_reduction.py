import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import export_graphviz

from mpl_toolkits.mplot3d import Axes3D
from features import key_indexs 

import graphviz
 
star_list = [
    10363,
3114,
2609,
1158,
2825,
679,
2618,
1335,
9838,
10710,
3281,
1759,
9032,
6218,
750,
193,
10061,
10337,
1838,
2675,
10707,
1766,
26,
2466,
2580,
1783,
6594,
46,
958,
2585,
2019,
1297,
37,
501,
4335,
4642,
2880,
]
 
def load_agglo(c: int):
    n = np.load("../cluster/agglomorative/{}_clus.npz".format(c), allow_pickle=True)
    return n['X'], n['Y'] 

def load_gmm(c: int):
    n = np.load("{}_clus.npz".format(c), allow_pickle=True)
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
        
        self.fig = plt.figure(figsize=(12, 10))
        # self.ax = self.fig.add_subplot(111, projection='3d')        
                
                
        self.trans_X = None
        self.trans_Y = None
        
    def feature_plot(self):
        f = self.fig.add_subplot(111)
        for fidx in range(len(self.X[0])):
            for i in range(len(self.X)):
                f.scatter(self.X[i][fidx],self.Y[i])
            plt.show()
        
        
    def add_plot(self, n_row, n_col,iteration,max_img):
        axis_num = 100*n_row+10*n_col+iteration
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
            axis_num = 100*nrows+10*2+i+1
            ax = self.fig.add_subplot(axis_num, projection='3d')
            for n in range(self.number_of_classes):
                ax.scatter([row[idxs[0]] for row in self.classes[n]], [row[idxs[1]] for row in self.classes[n]], [row[idxs[2]] for row in self.classes[n]], color=self.color_list[n])
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
        
        
        
    def RandomForest(self, write = False):
        Model = RandomForestClassifier(n_estimators=1)
        Model.fit(self.X, self.Y)
        print("Decision Tree number : {} \n Score of the tree : {}".format(len(Model.estimators_), Model.score(self.X, self.Y)))
        estimator = Model.estimators_[0]
        
        # y = Model.predict(self.X[0])
        # print("true : {}, predict : {}".format(self.Y[0], y))
        # print("weight : {}".format(Model.decision_path(self.X[0])))

        dot_data = export_graphviz(estimator, out_file='tree.dot', 
                        feature_names = key_indexs,
                        precision = 3, # 소수점 표기 자릿수
                        filled = True, # class별 color 채우기
                        rounded=True, # 박스의 모양을 둥글게
                    )

        # graph = graphviz.Source(dot_data)
        # graph.render(filename='tree.png', directory='./', format='png')

        pass
    
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
        
    def K_NN_for_user(self):
        temp_x = []
        temp_y = []
        
        for idx in star_list:
            temp_x.append(self.X[idx])
            temp_y.append(self.Y[idx])

        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(temp_x)
        distances , indexs = nbrs.kneighbors([[0,0,0,0,0,0,0,0,0,0,0,0,0]])
        
        print(indexs)
        
        return indexs
        
        pass
            
    def means(self):
        mean_list = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        max_list = [None for _ in range(13)]
        min_list = [None for _ in range(13)]
        length = len(self.X)
        for elem in self.X:
            for i, feat in enumerate(elem):
                mean_list[i] += feat/length
                if max_list[i] == None:
                    max_list[i] = feat
                else:
                    if max_list[i] <= feat:
                        max_list[i] = feat
                if min_list[i] == None:
                    min_list[i] = feat
                else:
                    if min_list[i] >= feat:
                        min_list[i] = feat

        print(mean_list)
        print(max_list)
        print(min_list)
            
        
t = dataset(class_num= 5,cluster="gaussian")
t.K_NN_for_user()
t.means()
for idx in star_list:
    print(t.Y[idx])

# temp = [4,
# 2,
# 3,
# 3,
# 3,
# 3,
# 2,
# 2,
# 4,
# 4,
# 2,
# 2,
# 1,
# 1,
# 3,
# 3,
# 4,
# 0,
# 2,
# 2,
# 1,
# 2,
# 3,
# 2,
# 2,
# 2,
# 1,
# 3,
# 2,
# 3,
# 3,
# 2,
# 2,
# 2,
# 1,
# 4,
# 3,
# 2,
# 4,
# 3,
# ]

# dic = {
#     0 : 0,
#     1 : 0,
#     2 : 0,
#     3 : 0,
#     4 : 0
# }

# for elem in temp:
#     dic[elem] += 1

# print(dic)