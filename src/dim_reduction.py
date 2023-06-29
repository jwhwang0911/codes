import numpy as np
from clustering import *
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

number_of_classes = 3

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# 예제 데이터 생성
X, y = load_agglo(3)

print("Manifold learning start!")

# t-SNE 모델 인스턴스화 및 매니폴드 학습
tsne = TSNE(n_components=3, random_state=42)
X_tsne = tsne.fit_transform(X)
print(X_tsne.shape)
print("params : {}".format(tsne.get_params()))

classes = []

for j in range(number_of_classes):
    classes.append([])

for i in range(len(y)):
    match y[i]:
        case 0:
            classes[0].append(X_tsne[i])
        
        case 1:
            classes[1].append(X_tsne[i])

        case 2:
            classes[2].append(X_tsne[i])
            

# 시각화
ax.scatter(classes[0][:][0], classes[0][ : ][ 1], classes[0][ :][ 2], color="red")
ax.scatter(classes[1][:][0], classes[1][ : ][ 1], classes[1][ :][ 2], color="red")
ax.scatter(classes[2][:][0], classes[2][ : ][ 1], classes[2][ :][ 2], color="red")


plt.show()