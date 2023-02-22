import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

current_path = os.path.dirname(os.path.abspath(__file__))

print("==============================")
print("|           실습 #1          |")
print("==============================")

# 데이터 셋 읽어들이비다.
data = pd.read_csv(current_path + "/multiple_linear_regression_data.csv")
print("데이터셋:\n",data)

# 데이터 특성에 따른 값으로 변수에 저장
height = np.array(data["height"])
weight = np.array(data["weight"])
label = np.array(data["label"])

# 실습 1 출력
ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(height, weight, label)
ax.set_xlabel("height"); ax.set_ylabel("weight"); ax.set_zlabel("age", rotation=0)
plt.show()

# csv파일 데이터 크기 80 x 3
data_size = data.shape

# height과 weight을 하나의 데이터 x로 묶고 결과 값인 label을 따로 묶었습니다. 
x= np.c_[height, weight]
y = label

print("==============================")
print("|           실습 #2          |")
print("==============================")

# 기존 데이터에 bias를 추가했습니다.
# bias, height, weight
x_bias = np.c_[np.ones(data_size[0]), x]
#print("bias, height, weight\n",x_bias)


# 유사역행렬 (pseudoinverse inverse matrix) = Moore-Penrose inverse mastrx
#  square matrix가 아니거나 어떤 특성이 중복되어 행렬 X^T X의 역행렬이 없다면 정규방정식 동작안한다.
x_bias_t = x_bias.T
#행렬 계산을 통한 theta값 : bias, height, weight
theta = np.linalg.pinv(x_bias_t.dot(x_bias)).dot(x_bias_t).dot(y)
print("최적 세타 값 : ", theta)

# x축(height)과 y축(weight)의 좌표 구간을 설정하기 위한 최대 최소 설정
# 최소 구간 : 1의 자리에서 내림
# 최대 구간 : 1의 자리에서 올림
# 모든 구간 포함할 수 있게 설정
height_min = (min(height))
height_max = (max(height))
height_lim = [(height_min//10) * 10 ,(height_max // 10 + 1) * 10]

weight_min = (min(weight))
weight_max = (max(weight))
weight_lim = [(weight_min//10)*10, (weight_max // 10 +1) * 10]
print("설정한 좌표 범위:\n", height_lim, weight_lim)

# 각 자리의 시작부터 끝지점까지 동일한 간격으로 1000개의 데이터 생성
height_space = np.linspace(height_lim[0], height_lim[1], 1000)
weight_space = np.linspace(weight_lim[0], weight_lim[1], 1000)

# 데이터 1차원에서 2차원으로 확장
height_mesh, weight_mesh = np.meshgrid(height_space, weight_space)
print("height 2d :\n {}\nweight 2d : {}".format(height_mesh,weight_mesh))

# 가중치를 사용하여 평면을 만든다.
y_space = theta[2]*weight_mesh + theta[1]*height_mesh + theta[0]
print(y_space)

def origin_data_space():
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(height,weight,label)
    ax.axis(height_lim + weight_lim)
    ax.set_xlabel('height'); ax.set_ylabel('weight'); ax.set_zlabel('age', rotation=0)
    ax.plot_surface(height_mesh, weight_mesh, y_space, cmap="plasma")
    plt.title("Analytic Solution")
    plt.grid(True)
    plt.show()
    
origin_data_space()

print("==============================")
print("|           실습 #3          |")
print("==============================")

# y의 예측값을 구하기 위해 비교 대상인 y와 대응하는 동일한 데이터 x사용함
# 따라서 height, weight을 사용함
y_pred_eq = theta[2]*weight + theta[1]*height + theta[0] # 수식 사용
y_pred_matrix = x_bias @ theta # 행렬 곱 사용

# mse값을 만드는 함수 y의 예측값과 y 원본 값, 데이터의 갯수를 받는다.
def calc_mse(y_pred, y_origin, size):
    return sum((y_pred - y_origin) ** 2) / size

# mse, data_size는 x의 모양 80 X 3이다. 따라서 data_size[0]은 80
mse_eq = calc_mse(y_pred_eq, y, data_size[0])
mse_matrix = calc_mse(y_pred_matrix, y, data_size[0])
print("mse eq값 : ", mse_eq)
print("mse matrix값 : ", mse_matrix)


print("==============================")
print("|           실습 #4          |")
print("==============================")
# 경사하강법

# 기본
learning_rate = 0.000055
n_iter = 200000

np.random.seed(85) # 이후 모든 출력 값을 동일하게 하기 위해 rand값 고정
gd_theta = np.random.randn(3,) # 평균 0 표준편차 1인 정규분포따르는 -1~1사이값 가져옴
gd_theta *= 3  # -1~1의 값 증폭을 위한 값 (시그마값)
print("임의의 초기 theta값 : bias GD = {}, height GD = {}, weight GD = {}) : ".format(gd_theta[0], gd_theta[1], gd_theta[2]))
print("초기 learning rate : ",learning_rate)
print("초기 반복 값 : ", n_iter)

# theata의 매개변수 learning rate에 따른 값의 변화를 보기 위한 함수
def y_space_graph(ax, y_pred=None, rate=learning_rate, epoch=n_iter, show="origin"):
    if ax == None:
        return

    title = ""
    if show == "origin":
        title = "Analytic Solution"
    elif show=="rate":
        title = "Gradient Decent Method, rate:{}".format(rate)
    elif show=="epoch":
        title = "Gradient Decent Method, epoch:{}".format(epoch)
     # Analytic solution인지 Gradient descent인지 구분
        
    # 가장 먼저 기본 데이터를 출력
    ax.scatter(height,weight,label)
    
    if show == "origin":
        # Analytic
        ax.plot_surface(height_mesh, weight_mesh, y_space,  alpha=0.7, cmap="plasma")
    else:
        # Gradient Descent
        ax.plot_surface(height_mesh, weight_mesh, y_pred, alpha=0.7,cmap="plasma", rstride=100,cstride=100, edgecolor="black")
    ax.axis(height_lim + weight_lim)
    ax.set_title(title)
    ax.set_xlabel('height'); ax.set_ylabel('weight'); ax.set_zlabel('age', rotation=0)

tolerance = 0.000001
# 오차 계산 함수
def isStop(now_th, before_th):
    # 현재 theta와 befoe theta값에 절대값 사용하여
    # 각각의 차이 중 큰 값이 오차 허용 값보다 작을 때 참을 반환
    th_abs = np.abs(now_th) 
    before_th_abs = np.abs(before_th)
    result = (abs(th_abs - before_th_abs).max() < tolerance)
    return result

# iter과 가중치 theta값 고정 후 learning rate에 따른 값 변화
def theta_graph(rate=learning_rate, th=gd_theta, epoch=n_iter, ax=None, show="origin"):
    # 최종적으로 몇번 반복했는지 나타내는 변수
    stop_iter = 0

    # 반복 횟수를 고정한 채 진행
    for i in range(epoch):
        before_th = th

        # 행렬을 통한 기울기 계산
        gradient = 2/data_size[0] * x_bias.T.dot(x_bias.dot(th) - y)
        th = th - rate * gradient
        
        # MSE계산을 위한 y 예측값 계산 후 mse 값 계싼
        y_pred = x_bias.dot(th)
        mse = calc_mse(y_pred, y, data_size[0])

        # 가중치 값 비교를 하여 오차 범위보다 적은지 확인하는 함수 호출
        if isStop(th, before_th):
            # 오차 범위보다 적다면 멈추고 해당 횟수 저장
            stop_iter = i+1
            break

    if stop_iter == 0:
        stop_iter =  epoch
    
    # 최종 가중치(세타)값을 활용하여 y예측값 만든다.
    y_pred_graph = th[2]*weight_mesh + th[1]*height_mesh + th[0]
    y_space_graph(ax, y_pred_graph, rate, epoch, show)
    
    # 최종 가중치(세타)값을 활용하여 y예측값 만든다
    y_pred = x_bias.dot(th)
    mse = calc_mse(y_pred, y, data_size[0])
    print("[Epoch]:{}, [최종반복 횟수]:{}, [Learning rate]:{} ===> [W초기]:{}, [gradient]:{}, MSE값:{:.6f}".format(epoch, stop_iter, rate, gd_theta, th, mse))
    return th

# 그래프 출력 함수 반복횟수 고정일 때
def print_rate(func):
    fig = plt.figure(figsize=(15, 30))
    ax = fig.add_subplot(321, projection='3d')
    th = y_space_graph(ax)
    
    ax = fig.add_subplot(322, projection='3d')
    th2 = func(rate=0.0001, ax=ax, show="rate")
    
    ax = fig.add_subplot(323, projection='3d')
    th = y_space_graph(ax)
    
    ax = fig.add_subplot(324, projection='3d')
    th4 = func(rate=0.00005, ax=ax, show="rate")

    ax = fig.add_subplot(325, projection='3d')
    th = y_space_graph(ax)
    
    ax = fig.add_subplot(326, projection='3d')
    th6 = func(rate=0.000001, ax=ax, show="rate")
    plt.show()
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(321, projection='3d')
    th = y_space_graph(ax)
    
    ax = fig.add_subplot(322, projection='3d')
    th2 = func(rate=0.000057, ax=ax, show="rate")
    
    ax = fig.add_subplot(323, projection='3d')
    th = y_space_graph(ax)
    
    ax = fig.add_subplot(324, projection='3d')
    th4 = func(rate=0.000055, ax=ax, show="rate")

    ax = fig.add_subplot(325, projection='3d')
    th = y_space_graph(ax)
    
    ax = fig.add_subplot(326, projection='3d')
    th6 = func(rate=0.000053, ax=ax, show="rate")
    plt.show()

print("\nLearning Rate에 따른 변화")  
print_rate(theta_graph)

# learning rate 고정 후 반복 횟수 조정
def print_iter(func):
    fig = plt.figure(figsize=(15, 30))
    ax = fig.add_subplot(321, projection='3d')
    th = y_space_graph(ax)
    
    ax = fig.add_subplot(321, projection='3d')
    th1 = func(epoch=20000, ax=ax, show="epoch")

    ax = fig.add_subplot(322, projection='3d')
    th2 = func(epoch=50000, ax=ax, show="epoch")
    
    ax = fig.add_subplot(323, projection='3d')
    th3 = func(epoch=100000, ax=ax, show="epoch")

    ax = fig.add_subplot(324, projection='3d')
    th4 = func(epoch=200000, ax=ax, show="epoch")

    ax = fig.add_subplot(325, projection='3d')
    th5 = func(epoch=500000, ax=ax, show="epoch")

    ax = fig.add_subplot(326, projection='3d')
    th6 = func(epoch=800000, ax=ax, show="epoch")
    plt.show()

print("\nEpoch에 따른 변화")
print_iter(theta_graph)


print("==============================")
print("|           실스 #5          |")
print("==============================")

data1 = pd.read_csv("linear_regression_data01.csv", names=["age", "tall"])
print("지난 데이터:\n",data1)

# 나이를 x1, 결과인 키를 y1로 분리하여 저장
x1 = data1['age']
y1 = data1['tall']

# x1의 최대값과 최소값 저장
x1_max = max(x1)
x1_min = min(x1)
x1_size = x1.shape

# mu값 계산
# K값에 대해 배열을 반환, 각각의 값 미리 계산
def calc_mu(K):
    return [x1_min + (x1_max - x1_min) / (K - 1) * k for k in range(K)]

# sigma값 계산
# 고정된 값으로 하나의 상수
def calc_sigma(K):
    return (x1_max - x1_min) / (K-1)

# 가우스 계산
# x값 자체가 하나의 특성 데이터 전체로 들어옴
# mu는 각 배열의 원소들을 하나씩 넣어준다.
# x는 array, mu자체가 현재 k(새로운 특성)에 해당하는 값중 하나(기존 리스트에서 한개씩 함수로 전달받음) 
def calc_gauss(x, mu, sigma):
    return np.exp(-1/2 * ((x - mu)/sigma)**2)

# 각 특성마다 해당하는 mu값과 함께 가우스를 계산하여 배열에 추가
# result[i]는 각 특성 데이터 값이므로 np.array로 변환하여 T시켜
# 데이터마다 여러개의 특성을 뽑아올 수 있도록 변경
# result[0] = [특성1의 데이터1, 특성1의 데이터2] => [특성1, 특성2, 특성3 ...]  
def make_pi(x, K, mu, sigma):
    result = []
    for i in range(K):
        # 계산한 mu값은 리스트
        # mu0, mu1 mu2 ... 값을 하나하나 넣어준다.
        result.append(calc_gauss(x, mu[i], sigma))
    return np.array(result).T
    
# 가우시안 값을 구하기 위해 처음으로 호출하는 함수
def gaussian_func(K, x_data):
    # u_k = 평균, sigma = 표준편차
    mu = calc_mu(K) # mu값 구하는 함수 호출
    sigma = calc_sigma(K) # sigma값 구하는 함수 호출
    
    # 기본 데이터값을 변경하지 않기 위해 깊은 복사로 데이터 가져옴
    x_temp = x_data.copy()
    print("[K = {}] : mu={}, sigma={}".format(K,mu, sigma))

    # pi 행렬을 구하는 함수
    pi = make_pi(x_temp,K, mu,sigma)
    
    # y_hat = bias + pi_0(x) + pi_1(x) + pi_2(x) 
    pi_b = np.c_[np.ones(x1_size[0]), pi]
    return pi_b

k_list = [3,5,8,10]
pi_list = []
# k 갯수 많큼 for문 돌린다.
for i in range(len(k_list)):
    # 구한 pi 행렬을 리스트에 넣어 저장해둔다.
    # 각 값은 k_list의 K 값과 대응 된다.
    pi_list.append(gaussian_func(k_list[i], x1))

# 가중치(theta)값 구하는 함수
def calc_pi_theta(pi):
    # pinv를 사용하여 해석해를 구했습니다.
    pi_t = pi.T
    return np.linalg.pinv(pi_t.dot(pi)).dot(pi_t).dot(y1)

# theta값
theta_list = []
for i in range(len(k_list)):
    # 각각의 pi값에 대한 theta(가중치를) 리스트에 저장
    theta_list.append(calc_pi_theta(pi_list[i]))
    print("K = {}:[GD,(bias, 0,1,...)]:{}\n".format(k_list[i], theta_list[i]))

print("==============================")
print("|           실스 #6          |")
print("==============================")

# y 예측 값과 mse 값을 구합니다.
y1_list= []
mse_pi_list = []

for i in range(len(k_list)):
    # 위에서 구한 pi값과 theta값을 사용하여 y 예측값을 구하고 리스트에 저장
    y1_list.append(pi_list[i].dot(theta_list[i]))
    # 저장한 y예측값과 실제 y값을 통해 mse를 구하기 위해
    # 위에서 선언한 mse구하는 함수 호출하여 mse값을 저장합니다.
    mse_pi_list.append(calc_mse(y1_list[i], y1, x1_size[0]))

# 실습 6 출력 함수
def show_graph6():
    # 그래프에서 y예측값 출력은 직선으로 되어있어서 데이터를 정렬할 필요가 있습니다.
    # 따라서 x와 y예측값을 같이 묶어주기 위해 show_data 선언
    show_data = np.array(x1)
    
    # 출력할 예측값의 갯수는 k값의 갯수입니다.
    # y1_list는 4 X 25의 형태이므로 한개씩 넣어줬습니다.
    for i in range(len(k_list)):
        # 매번 옆에 하나의 y예측값들을 넣습니다.
        show_data = np.c_[show_data, y1_list[i]]    
    print(show_data)

    # 해당 데이터 집합은 x데이터의 순서대로 정렬해야 하므로
    # show_data[0]을 기준으로 정렬합니다.
    show_data = sorted(show_data, key = lambda x : x[0])
    # 정렬 후 하나의 데이터는 [x값, y1예측값, y2예측값, y3예측값, y4예측값]
    # 형태이므로 transpose하여 show_data[0] = [x값들], 
    #                         show_data[1~4] = [y예측값들] 이 나오게 한다.
    show_data = np.array(show_data).T
    print(show_data)

    # 그래프 출력
    plt.figure(figsize=(20, 20))
    for i in range(4):
        plt.subplot(2,2,1 + i); 
        # 원본 데이터
        plt.plot(x1, y1, 'b.', label="original")
        # y예측 값
        plt.plot(show_data[0], show_data[i+1], 'r-', label="prediect, k={}, MSE={}".format(k_list[i],mse_pi_list[i]))
        plt.xlabel("age")
        plt.ylabel("height", rotation=0)
        plt.title("Regression with gqussian basis function")
        plt.legend(loc="upper left")
        plt.grid(True)
    plt.show()
show_graph6()

print("==============================")
print("|           실스 #7          |")
print("==============================")

# 이미 위에서 한번 구현했습니다.
#mse_pi_list = []
for i in range(len(k_list)):
    # 저장한 y예측값과 실제 y값을 통해 mse를 구하기 위해
    # 위에서 선언한 mse구하는 함수 호출하여 mse값을 저장합니다.
    #mse_pi_list.append(calc_mse(y1_list[i], y1, x1_size[0]))
    print("K가 {}일때 MSE : {}".format(i,mse_pi_list[i]))

plt.figure()
plt.stem(k_list,mse_pi_list, 'b.', label="MSE")
plt.xlabel("age")
plt.ylabel("MSE")
plt.grid()
plt.legend(loc = "upper right")
plt.show()