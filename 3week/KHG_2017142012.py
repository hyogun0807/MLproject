import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

print("==============================")
print("|           실습 #1          |")
print("==============================")

# 데이터 셋 읽어들입니다.
data = pd.read_csv("lin_regression_data_03.csv", names=["age","height"])
print("데이터셋:\n",data)

plt.plot(data["age"],data["height"], 'b.', markersize=10, label="infant's age and height data")
plt.xlabel("age[months]")
plt.ylabel("height[cm]")
plt.legend(loc="upper left")
plt.grid()
plt.show()

print("==============================")
print("|           실습 #2          |")
print("==============================")

# 훈련 세트와 테스트 세트 분리
train_set, test_set = data[:20], data[20:]

plt.plot(train_set["age"], train_set["height"], 'b.',markersize=10, label="Training data")
plt.plot(test_set["age"], test_set["height"], '.', color='orange',markersize=10, label="Test data")
plt.xlabel("age[months]")
plt.ylabel("height[cm]")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()


print("==============================")
print("|           실습 #3          |")
print("==============================")

# 최대최소의 인덱스 나타내기 위해 선언
MINMAX_CONST = {
    "MIN" : 0,
    "MAX" : 1
}

# mu값 계산
# K값에 대해 배열을 반환, 각각의 값 미리 계산
# Train 데이터 마다 가지고 있는 min, max가 다르기에 각각의 값을 가져온다
def calc_mu(K, minmax):
    return [minmax[MINMAX_CONST["MIN"]] + (minmax[MINMAX_CONST["MAX"]] - minmax[MINMAX_CONST["MIN"]]) / (K - 1) * k for k in range(K)]

# sigma값 계산
# 고정된 값으로 하나의 상수
def calc_sigma(K, minmax):
    return (minmax[MINMAX_CONST["MAX"]] - minmax[MINMAX_CONST["MIN"]]) / (K-1)

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
def gaussian_func(K, x_data, minmax):
    # u_k = 평균, sigma = 표준편차
    mu = calc_mu(K, minmax) # mu값 구하는 함수 호출
    sigma = calc_sigma(K, minmax) # sigma값 구하는 함수 호출
    
    # 기본 데이터값을 변경하지 않기 위해 깊은 복사로 데이터 가져옴
    x_temp = x_data.copy()

    # pi 행렬을 구하는 함수
    pi = make_pi(x_temp,K, mu,sigma)
    
    # y_hat = bias + pi_0(x) + pi_1(x) + pi_2(x) 
    pi_b = np.c_[np.ones(len(x_data)), pi]
    return pi_b

# 구해야 하는 K 값
k_list = [6,7,8,9,10,11,12,13]

# 훈련 데이터의 최소 최대값 : 시그마와 평균값을 구하기 위해 존재
train_minmax = [min(train_set["age"]), max(train_set["age"])]

train_pi_k_list = []
test_pi_k_list = []
for i in range(len(k_list)):
    train_pi_k_list.append(gaussian_func(k_list[i], train_set["age"], train_minmax))
    test_pi_k_list.append(gaussian_func(k_list[i], test_set["age"], train_minmax))    
    
# 가중치(theta)값 구하는 함수
def calc_pi_theta(pi, y):
    # pinv를 사용하여 해석해를 구했습니다.
    pi_t = pi.T
    return np.linalg.pinv(pi_t.dot(pi)).dot(pi_t).dot(y)

# train에 대한 theta값
theta_list = []
for i in range(len(k_list)):
    # 각각의 pi값에 대한 theta(가중치를) 리스트에 저장
    theta_list.append(calc_pi_theta(train_pi_k_list[i], train_set["height"]))
    print("K = {}:[GD,(bias, 0,1,...)]:{}\n".format(k_list[i], theta_list[i]))
    
    
print("==============================")
print("|           실습 #4          |")
print("==============================")
# mse값을 만드는 함수 y의 예측값과 y 원본 값, 데이터의 갯수를 받는다.
# 현재 RMSE사용
def calc_mse(y_pred, y_origin, size):
    return np.sqrt(sum((y_pred - y_origin) ** 2) / size)


# 구한 mse 값을 저장하기 위한 리스트.
train_mse_list = []
test_mse_list = []

# RMSE 구현을 위한 train과 test 데이터 갯수
train_y_size = len(train_set["height"])
test_y_size = len(test_set["height"])
for i in range(len(k_list)):
    # 위에서 구한 pi값과 theta값을 사용하여 y 예측값을 구한다.
    train_y = train_pi_k_list[i].dot(theta_list[i])
    test_y = test_pi_k_list[i].dot(theta_list[i])
    
    # 저장한 y예측값과 실제 y값을 통해 mse를 구하기 위해
    # 위에서 선언한 mse구하는 함수 호출하여 mse값을 저장합니다.
    train_mse_list.append(calc_mse(train_y, train_set["height"], train_y_size))
    test_mse_list.append(calc_mse(test_y, test_set["height"],test_y_size))
    
plt.plot(k_list, train_mse_list, "b-", label="training MSE")
plt.plot(k_list, test_mse_list, "-", color="orange", label="test MSE")
plt.xlabel("K"); plt.ylabel("MSE")
plt.legend(loc="lower left")
plt.grid(True)
plt.show()

print("==============================")
print("|           실습 #5          |")
print("==============================")

# 임의의 숫자 생성르 위해
import random

split_num = 5
COLOR_DATA = "0123456789ABCDEF"

def make_color():
    # '#' + COLOR_DATA에서 6개를 랜덤으로 선택하여 값을 가져온다.
    return '#'+ ''.join([random.choice(COLOR_DATA) for _ in range(6)])
    
color_list = [make_color() for i in range(split_num)]
# st, nd, rd, th .... 
word_list = ["th" for _ in range(split_num)]
word_list[0] = "st"
word_list[1] = "nd"
word_list[2] = "rd"

# 분할된 데이터를 넣는 리스트
data_split_list = []

plt.figure(figsize=(10,10))
for i in range(split_num):
    # 필요한 데이터 숫자만 뽑아오기
    data_split_list.append(data[split_num * i : split_num * (i+1)])
    
    plt.plot(data_split_list[i]["age"], data_split_list[i]["height"], '.', color=color_list[i], markersize=15, label="{}{} set".format(i+1, word_list[i]))
plt.xlabel("age[months]")
plt.ylabel("height[cm]")
plt.legend(loc="upper left")
plt.grid(True)
plt.show()

print("==============================")
print("|           실습 #6          |")
print("==============================")

# age와 height을 각각 0과 1로 매칭 
# 나중에 인덱스로 접근할때 보기 쉽게하기 위해 사용
XY_CONST = {
    "age" : 0,
    "height" : 1
}

# 기본 설정 K = 9
K = 9

# 분리된 데이터 다시 모으는 리스트 : 각 폴드마다 훈련 4개, 검증 1개로 만든다.
train_data_list = []
valid_data_list = []

# 가중치 : theta
theta_list = []

# mse
train_mse_list = []
valid_mse_list = []

def make_prac6():
    for i in range(split_num):
        # 특성은 age와 라벨링값 height 2개
        # reshape로 특성만 2개를 고정 나머지는 알아서 맞게 하기 위해 -1,2
        # 각각이 0번째까 age, 1번째가 height을 나타낼 수 있게 transpose
        # 데이터를 한개씩 검증 데이터로 변환
        v_data = np.array(data_split_list[i]).reshape(-1,2).T
        # 훈련 데이터는 검증 데이터를 제외한 나머지 4개이다.
        t_data = np.array(data_split_list[:i] + data_split_list[i+1:]).reshape(-1,2).T

        # 완성된 훈련데이터와 검증데이터를 각각의 리스트에 저장 : 나중에 그래프 출력에 사용
        valid_data_list.append(v_data)
        train_data_list.append(t_data)

        # 훈련 데이터에 대한 최소 최대를 저장
        # 가우시안의 평균과 시그마를 구하기 위해 선언
        minmax = [min(t_data[XY_CONST["age"]]), max(t_data[XY_CONST["age"]])]

        # 각각의 데이터 pi값 계산후 저장
        train_pi = gaussian_func(K, train_data_list[i][XY_CONST["age"]], minmax)
        valid_pi = gaussian_func(K, valid_data_list[i][XY_CONST["age"]], minmax)

        # train에 대한 가중치(theta)를 구한다. 
        theta_list.append(calc_pi_theta(train_pi, train_data_list[i][XY_CONST["height"]]))

        # 위에서 구한 pi값과 theta값을 사용하여 y 예측값을 구한다.
        train_y = train_pi.dot(theta_list[i])
        valid_y = valid_pi.dot(theta_list[i])

        # 저장한 y예측값과 실제 y값을 통해 mse를 구하기 위해
        # 위에서 선언한 mse구하는 함수 호출하여 mse값을 구한 후 저장합니다.
        train_mse_list.append(calc_mse(train_y, t_data[XY_CONST["height"]], len(train_y)))
        valid_mse_list.append(calc_mse(valid_y, v_data[XY_CONST["height"]], len(valid_y)))
        print("K(폴드)={}, 매개변수 : {}, 일반화 오차 : {}".format(i+1, theta_list[i], valid_mse_list[i]))
    
make_prac6()

def make_prac7():
    plt.figure(figsize=(25,25))
    for i in range(split_num):
        plt.subplot(321+ i)
        
        # 전체 데이터에 대한 minmax값
        minmax_all = [min(data["age"]), max(data["age"])]
        
        # 해당 폴드에서의 훈련데이터 값으로만 만든 minmax값
        minmax = [min(train_data_list[i][XY_CONST["age"]]), max(train_data_list[i][XY_CONST["age"]])]
        
        # 전체 데이터를 포함하는 그래프 그리기 위해 해당 범위를 포함하는 값 생성.
        x_lin = np.linspace(min(data["age"]), max(data["age"]), 1000)
        
        # 전 범위의 데이터셋에 대해 가우시안 적용
        x_pi = gaussian_func(K, x_lin, minmax_all)
        
        # 가우시안을 적용한 데이터 셋과 각 폴드마다의 가중치값을 곱하여 y예측값 도출
        y_hat = x_pi.dot(theta_list[i])
        plt.plot(x_lin, y_hat, 'b-', label="predict, k={}-fold, MSE={}".format(i+1, round(valid_mse_list[i], 5)))
        
        # 실습 6에서의 저장한 기본 데이터 값 출력
        plt.plot(train_data_list[i][XY_CONST["age"]], train_data_list[i][XY_CONST["height"]], 'b.', label="training set")
        plt.plot(valid_data_list[i][XY_CONST["age"]], valid_data_list[i][XY_CONST["height"]], '.', color="orange", label="validation set")
        
        plt.xlabel("age[months]")
        plt.xlabel("height[cm]")
        plt.grid(True)
        plt.legend(loc="upper left")
    plt.show()
    
make_prac7()