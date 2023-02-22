import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

import os

print("=================================")
print("|            실습 #1-1           |")
print("=================================")


main_dir = 'C:\\Users\\parks\\Desktop\\대학\\4학년 1학기\\ml\\7week'

# DataFrame 가져오기
dataFrame = pd.read_csv(os.path.join(main_dir, "Mall_Customers.csv"))

# 특성 데이터 
data = dataFrame[['Annual Income (k$)', 'Spending Score (1-100)']]

k = 3 # Clustering 갯수

"""_summary_
    n_clusters : 그룹 수 (k)
    k-means++  : 방식, 랜덤값 지정
    random_state : ( 난수 고정)    
"""
model = KMeans(n_clusters = k, init='k-means++', random_state= 10)

# 데이터 프레임에 결과 값 추가
dataFrame['cluster'] = model.fit_predict(data)

# 마지막 중심 값
# final_centroid type : numpy. ndrray
final_centroid = model.cluster_centers_
print(final_centroid)

# 그래프 출력
plt.figure(figsize=(8,8))
for i in range(k):
    plt.scatter(dataFrame.loc[dataFrame['cluster'] == i, 'Annual Income (k$)'], dataFrame.loc[dataFrame['cluster'] == i, 'Spending Score (1-100)'], label="cluster_"+str(i))
plt.scatter(final_centroid[:,0], final_centroid[:,1], s=100, c='violet', marker='x', label="Centroids")
plt.legend()
plt.title(f'K={k} results', size=15)
plt.xlabel('Annual Income', size=12)
plt.ylabel('Spending Score', size=12)
plt.show()

print("=================================")
print("|            실습 #1-2           |")
print("=================================")

# elbow method 
def elbow(x):
    """_summary_
        x: data
    """
    sse = []
    for i in range(1, 11):
        km = KMeans(n_clusters = i, init = 'k-means++', random_state=0)
        km.fit(x)
        sse.append(km.inertia_)
        print(km.inertia_)
    plt.plot(range(1,11), sse, marker = 'o')
    plt.xlabel("# of clusters")
    plt.ylabel("Inertia")
    plt.show()
elbow(data)


print("=================================")
print("|            실습 #2            |")
print("=================================")

# Income, Spending Score(1~100) 그래프
plt.figure(figsize=(8,8))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], color='black', label='unclustered data')
plt.legend(loc = 'upper right')
plt.xlabel('Income', size=12)
plt.ylabel('Spending Score(1-100)', size=12)
plt.show()


print("=================================")
print("|            실습 #3            |")
print("=================================")

K=3 # 클러스터 갯수
iteration = 100 # 반복 횟수
data_set = data.to_numpy() # Income, Spending Score 데이터셋 리스트로 변환

# 랜덤 클러스터 초기값 가져오는 함수
def get_random_clusters(k):
    """_summary_
        k : 클러스터 갯수
        Returns: 클래스터 갯수와 특성에 맞게 반환
    """

    data_length = len(data) # 데이터 갯수 저장
     
    # 데이터 갯수만큼의 범위로 k개의 인덱스 랜덤으로 가져온다.
    rand_indexes = [random.randrange(0, data_length) for _ in range(k)]
    print("rand indexes : ",rand_indexes)
    return data_set[rand_indexes] # 데이터 셋에서 랜덤 인덱스에 맞게 가져온다.

# 초기 클러스터 중심점 저장
cluster_centroid =  get_random_clusters(k)
print(cluster_centroid)

# p-norm 함수
# 유클리디안 거리
def euclidean_dist(data, centroid):
    result = list(map(lambda x: sum(x), (centroid - data) ** 2))
    return result

# 맨해튼 거리
def manhattan_dist(data, centroid):
    result = list(map(lambda x: sum(x), abs(centroid - data)))
    return result

# 클러스터 중심정들 중 가장 가까운 지점 찾아내는 함수
def find_cluster_index(result):
    return np.argmin(result)

# 다음 클러스터 중심정르 위해 평균을 구하는 함수
def find_mean(cluster_list, features):
    result = np.array([[0 for _ in range(features)] for _ in range(len(cluster_list))])
    
    # 클러스트 리스트 : 클러스터 벡터    
    for i in range(len(cluster_list)):
        length = len(cluster_list[i])
        for d in cluster_list[i]: # 클러스터의 벡터 값들의 합
            result[i] = [result[i][f] + d[f] for f in range(features)]
        # 클러스타 벡터의 평균을 구함
        result[i] = [result[i][f] / length for f in range(features)]

    return result

# 클러스터링 동작하는 메인 함수
def clustering(data_set, cluster_centroid, k):
    # 클러스터의 벡터가 변화했는지 감지하기 위한 변수
    old_list = [[] for _ in range(k)]
    
    # 데이터 셋의 특성 갯수
    features = data_set.shape[1]
    
    # 100회 반복
    for i in range(iteration):
        cluster_list = [[] for _ in range(k)]
        
        # 데이터 셋 전체
        for d in data_set:
            # 각 데이터와 중심간 거리
            result = euclidean_dist(d,cluster_centroid)
            # 거리를 통한 가장 가까운 곳의 중심값 인덱스 
            cluster_index = find_cluster_index(result)
            # 중심값을 중심으로한 벡터에 포함
            cluster_list[cluster_index].append(d)

        # 기존 클러스터 벡터와 변경된 클러스터 벡터가 같으면 종료
        if np.array_equal(old_list,cluster_list):
            return cluster_list, cluster_centroid, i
        
        # 평균 구하는 함수
        cluster_centroid = find_mean(cluster_list, features)
        old_list = cluster_list.copy() 
        
    return cluster_list, cluster_centroid, i
f_cluster_list, f_cluster_centroid, final_index = clustering(data_set, cluster_centroid, k)

print(f_cluster_centroid)

# 구한 값들을 통해 결과 출력
plt.figure(figsize=(8,8))
for i, cluster in enumerate(f_cluster_list):
    cluster = np.array(cluster).T
    plt.scatter(cluster[0], cluster[1], label="cluster_"+str(i))
    
final_centroid = np.array(f_cluster_centroid).T
plt.scatter(final_centroid[0], final_centroid[1], s=100, c='violet', marker='x', label="Centroids")
plt.legend()
plt.title(f'K={k} results', size=15)
plt.xlabel('Annual Income', size=12)
plt.ylabel('Spending Score', size=12)
plt.show()