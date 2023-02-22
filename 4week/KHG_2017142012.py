import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from matplotlib import font_manager, rc

print("==========================")
print("=        실습 #1-1       =")
print("==========================")

# 시그모이드 함수 적용, 0~1사이의 값으로 변환
def sigmoid(linear_equation):
    # Args   : linear_equation (np.array): 데이터 값과 가중치의 선형조합
    # Returns:  np.array
    return 1 / (1 + np.exp(-linear_equation))

#  sigmoid function을 그리기 위해 x좌표 설정
x_lin = np.linspace(-10, 10, 1000)
x_lin_sigmoid = sigmoid(x_lin)

plt.title("(Sigmoid Function)")
plt.plot(x_lin, x_lin_sigmoid)
plt.grid(True)
plt.show()

# 한글 폰트 설정
#malgunbd.ttf
font_path = "C:\Windows\Fonts\malgunbd.ttf"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
rc('axes', unicode_minus=False)

print("==========================")
print("=        실습 #1-2       =")
print("==========================")

# 데이터 셋 읽어들입니다.
dataFrame = pd.read_csv("binary_data_insect.csv")
print("데이터셋:\n",dataFrame)

# Dataframe을 편의상 numpy로 변경
data = dataFrame.to_numpy()

# 그래프 출력을 위한 여성 남성 분리
female = np.array([x for x in data if x[1] == 0]).T
male = np.array([x for x in data if x[1] == 1]).T

plt.plot(female[0], female[1], 'b.', mfc='none', markersize=15, label="암컷")
plt.plot(male[0], male[1], 'r^', mfc='none', markersize=15, label="수컷")
plt.legend(loc="center right")
plt.yticks([-0.2,0,1, 1.2],[None,0,1,None])
plt.xlabel("무게 [g]")
plt.ylabel("성별")
plt.grid()
plt.show()

print("==========================")
print("=        실습 #1-3       =")
print("==========================")

gender_N = len(data)        # 데이터 갯수
gender_M = len(data[0]) - 1 # 특성 갯수 = 레이블 (1) + 특성(1) - 레이블(1)
gender_M_b = gender_M + 1   # bias 추가한 값 , 총 2개

gender_THRESHOLD = 0.5      # Gedner 경계 값

log_error = 1e-8            # Log 계산시 Log(0)으로 인한 계산 오류 문제를 위해 사용

# 각 경계값을 기준으로 높은지 아닌지를 결정하여 Bool값이 나온다.
# 이를 int로 변경하여 0과 1로 변경
def probability_func(pb):
    # Args: pb (np.array): 로지스틱 함수를 거쳐 0~1사이로 변경된 값
    # Return : list
    return list(map(int, pb > gender_THRESHOLD))

# 손실함수 Cross Entropy Loss 계산
def cross_entropy_loss(pb, y):
    """   
    Args:
        pb (np.array): 로지스틱 함수를 거쳐 0~1사이로 변경된 값
        y (np.array): 데이터 셋의 라벨 값, (실제 값)
    Return : float
    """
    N = len(y)
    # Log에 넣는 인자 값이 0이 안되게 만든다.
    for i in range(N):
        if pb[i] == 1: # np.log(1-pb) 예외처리
            pb[i] -= log_error
        elif pb[i] == 0: # np.log(pb) 예외처리, 혹시모를 예외처리
            pb[i] += log_error
    cee = sum(y * np.log(pb) + (1-y)*np.log(1-pb)) 
    return -cee / N

# 경사하강법

# 초기 값
LEARNING_RATE = 0.004 # Learning Rate
N_ITER = 200000       # 반복 횟수

# 특성과 레이블 분리
gender_x = dataFrame["Weight"]
gender_y = dataFrame["Gender"]

np.random.seed(85) # 이후 모든 출력 값을 동일하게 하기 위해 rand값 고정
# 평균 0 표준편차 1인 정규분포따르는 -1~1사이값 가져옴
# 2 X 1로 만듬
gd_theta = np.random.randn(gender_M_b,) 

# w0 * x0 + W1 * 1(bias) 형식으로 만들기 위해 [특성, bias]
gender_x_b = np.c_[gender_x, np.ones(gender_N)]

# 그래프 색 : Red, Blue, Black
color_list = ["#FF0000", "#0000FF", "#000000"]

# theta_gradient_descent 함수의 결과 값 인덱싱 매핑
GD_RESULT_MAP = {
    "weight" : 0,
    "cee" : 1,
    "epoch" : 2
}

# 그래프 라벨 출력을 위한 변수
PLT_LABEL = ["weight", "bias", "sepal_length", "petal_length", "bias"]


# 오차 허용 범위
tolerance = 1e-12

# 오차 계산 함수, 허용범위 이내인지 확인 후, [작다:참], [크다:거짓] 반환
def isStop(now_th, before_th):
    """
    Args:
        now_th (np.array)   : 현재 가중치 값
        before_th (np.array): 이전 가중치 값
    Returns: Bool
    """
    th_abs = np.abs(now_th) 
    before_th_abs = np.abs(before_th)
    result = (abs(th_abs - before_th_abs).max() < tolerance)
    return result

# 경사 하강법 함수
def theta_gradient_descent(x_b, y, M_b, lr=LEARNING_RATE, epoch=N_ITER, th=gd_theta):
    """
    Args:
        x_b   (np.array): 특성 데이터 + bias
        y     (np.array): 데이터에 따른 라벨값
        M_b   (int): _description_
        lr    (float, optional): 학습률. Defaults to LEARNING_RATE
        epoch (int, optional): 반복 횟수. Defaults to N_ITER.
        th    (np.array, optional): 초기 가중치 값. Defaults to gd_theta.
    Returns:
        List: accuracy(정확도), th(가중치), cee(손실 값), epoch(반복 횟수)
    """
    N = len(y)

    th = th.copy()
    stop_iter = 0

    th_list = []
    cee_list = []

    for i in range(epoch):
        before_th = th.copy() # 현재 가중치 변경 전에 저장

        zn = x_b.dot(th) # linear equation
        probability = sigmoid(zn) # 0~1사이로 변경
        cee = cross_entropy_loss(probability, y) # Corss Entropy Loss
        
        # w0, w1 ... bias 순으로 경사 하강 시행
        for j in range(M_b):
            discent = lr * 1/ N *sum((probability-y)*x_b.T[j])
            th[j] = th[j] - discent
        
        th_list.append(th.copy())
        cee_list.append(cee)
        
        # 가중치 값 비교를 하여 오차 범위보다 적은지 확인하는 함수 호출
        if isStop(th, before_th):
            # 오차 범위보다 적다면 멈추고 해당 횟수 저장
            stop_iter = i+1
            break

    if stop_iter == 0:
        stop_iter =  epoch

    epoch_list  = [i for i in range(stop_iter)]
    
    # 출력을 위해 [데이터 갯수 X 가중치 갯수(특성의 갯수)] =>
    #  [가중치 갯수(특성의 갯수) X 데이터 갯수]
    th_list = np.array(th_list).T

    plt.figure(figsize=(15,15))
    plt.subplot(121)
    for i in range(len(th_list)):
        label_index = 0 if len(th_list) == 2 else 2
        plt.plot(epoch_list, th_list[i], '-', color=color_list[i], label=PLT_LABEL[label_index+i])
    plt.xlabel("Epoch")
    plt.ylabel("weight")
    plt.xlim(0, epoch)
    plt.legend(loc="upper left")
    plt.grid()
    
    plt.subplot(122)
    plt.plot(epoch_list, cee_list, 'g-')
    plt.xlabel("Epoch")
    plt.ylabel("CEE")
    y_lim = [0,1] if N==10 else [0,6]
    plt.ylim(y_lim)
    plt.grid()
    plt.show()
    return th, cee_list[-1], epoch

# epoch에 따른 gd결과 값과 출력
gd_result1 = theta_gradient_descent(gender_x_b, gender_y, gender_M_b, epoch=230000)
gd_result2 = theta_gradient_descent(gender_x_b, gender_y, gender_M_b, epoch=240000)
print("epoch : {} ===========> W : [{} {}], cee : {}".format(gd_result1[GD_RESULT_MAP["epoch"]], gd_result1[GD_RESULT_MAP["weight"]][0], gd_result1[GD_RESULT_MAP["weight"]][1], gd_result1[GD_RESULT_MAP["cee"]]))
print("epoch : {} ===========> W : [{} {}], cee : {}".format(gd_result2[GD_RESULT_MAP["epoch"]], gd_result2[GD_RESULT_MAP["weight"]][0], gd_result2[GD_RESULT_MAP["weight"]][1], gd_result2[GD_RESULT_MAP["cee"]]))

# 두 결과 값중 높은 결과 값 출력
most_gd_result = gd_result1 \
                if gd_result1[GD_RESULT_MAP["cee"]] < gd_result2[GD_RESULT_MAP["cee"]] else \
                 gd_result2
print("GD 종료\nw0(feature) = {}, w1(bias) = {}".format(most_gd_result[GD_RESULT_MAP["weight"]][0], most_gd_result[GD_RESULT_MAP["weight"]][1]))

print("==========================")
print("=        실습 #1-4       =")
print("==========================")

# 정확도 구하는 함수, 예측값과 라벨값이 같은 갯수를 파악하여 전체갯수로 나눠 반환
def find_accuracy(y_hat, y):
    """
    Args:
        y_hat (np.array): Threshold를 통해 구분된 예측 값
        y (np.array): 실제 라벨 값
    Returns:
        flaot
    """
    count=0
    n = len(y)
    for i in range(n):
        if y[i] == y_hat[i]:
            count += 1
    return count / n

# 가장 좋은 가중치 값을 통한 정확도 측정
def best_th_accuracy(x_b, y, th):
    """
    Args:
        x_b (np.array): 특성 값 + bias
        y (np.array): 라벨 값
        th (np.array): 최적의 가중치 값
    Returns: float
    """
    zn = x_b.dot(th) # linear equation
    probability = sigmoid(zn) # 0~1사이로 변경
    y_hat = probability_func(probability) # Corss Entropy Loss
    return find_accuracy(y_hat, y) 

accuracy = best_th_accuracy(gender_x_b, gender_y, most_gd_result[GD_RESULT_MAP["weight"]])

print("훈련결과, 정확도 {}%\nw0(feature) = {}, w1(bias) = {}".format( \
                                                             accuracy * 100,\
                                                             most_gd_result[GD_RESULT_MAP["weight"]][0], \
                                                             most_gd_result[GD_RESULT_MAP["weight"]][1]))

# 결정 경계 그래프 출력을 위한 X값 설정
x_lin = np.linspace(min(gender_x), max(gender_x),1000)
# weight(x) * w0 + bias(1) * w1
dicision_boundary = most_gd_result[GD_RESULT_MAP["weight"]][0] * x_lin +  most_gd_result[GD_RESULT_MAP["weight"]][1]

plt.scatter(gender_x, gender_y)
plt.plot(x_lin, dicision_boundary, 'b-')
plt.ylim(-0.2,1.22)
plt.grid()
plt.xlabel("weight")
plt.ylabel("gender")
plt.show()

# 예측된 결과 값에 따라 1이면 남자 0이면 여자를 출력하기 위한 리스트
RESULT_GENDER = ["female", "male"]

# 예측 함수
def predict(data, weight):
    """
    Args:
        data (List or number(int or float)): 예측을 원하는 값
        weight (np.array): 최적화된 가중치
    """
    # weight = bias + 특성(1) | 2 X 1 matrix
    w = np.array(weight)
    # data = 데이터 갯수 X 1(feature) martrix
    d = np.array([data]).T
    
    # d = 데이터 갯수 X 2(feature + bias) matrix  
    d_b = np.c_[d, np.ones(len(d))]

    zn = d_b @ w # linear equation
    probability = sigmoid(zn) # sigmoid
    y_pred = probability_func(probability) # Threshold = 0.5

    for i in range(len(y_pred)):
        print("데이터 {}의 예측 결과 : Result={}".format(d[i], RESULT_GENDER[y_pred[i]]))
predict(20, most_gd_result[GD_RESULT_MAP["weight"]])


print("==========================")
print("=        실습 9주차       =")
print("==========================")
print("=        실습 #2-1       =")
print("==========================")

iris_data = pd.read_csv("iris.csv")
print(iris_data)
print(iris_data.dtypes)

# 특성과 라벨 분리
iris_df_feature = iris_data.iloc[:,:2].to_numpy() # sepal, petal특성을 가져온다.
iris_df_label = iris_data.iloc[:,-1].to_numpy()   # variety 라벨 값을 가져온다.

# 각각의 특성을 인덱스로 접근하기 위해 Transpose
# 100 X 3 => 3 X 100
iris_feature = iris_df_feature.T
iris_label = iris_df_label.T

# Object로 되어 있으므로 각 값을 0과 1로 변경하기위해 매핑
LABEL_NAME = {
    "Setosa" : 0,
    "Versicolor" : 1
}

# x : Setosa, Versicolor
# LABEL_NAME["Setosa"] = 0, lABEL_NAME["Versicolor"] = 1
iris_label = np.array(list((map(lambda x : LABEL_NAME[x], iris_label))))

# 인덱싱 접근의 가독성을 위해 각 특성이름을 매핑
FEATURE_NAME = {
    "sepal_length" : 0,
    "petal_length" : 1
}

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(iris_feature[FEATURE_NAME["sepal_length"]], iris_feature[FEATURE_NAME["petal_length"]], iris_label)
ax.set_xlabel("sepal_length"); ax.set_ylabel("petal_length"); ax.set_zlabel("variety", rotation=0)
plt.show()

print("==========================")
print("=        실습 #2-2       =")
print("==========================")

# 경사하강법

# 초기 값
IRIS_LEARNING_RATE = 0.0005
IRIS_N_ITER = 200000

iris_M = len(iris_feature) # 특성의 갯수
iris_M_b = iris_M+1        # 특성의 갯수 + bias

# -3~3의 임의의 값, (특성+bias)의 갯수만큼 리스트로 만듬
iris_gd_theta = np.array([random.uniform(-3,3) for _ in range(iris_M_b)])

# w0 * x0 + w1*x1 +  w2 * 1(bias)를 만들기 위해 끝에 추가
# [sepal_length, petal_length, bias(1)]   [[w0, w1, w2]] <= 이런 형식
iris_x_b = np.c_[iris_df_feature, np.ones(len(iris_df_label))]

# (특성+bias)의 갯수에 따라 변경되는 로직 작성하였기에
# 동일한 함수로 해당 함수 적용
iris_result = theta_gradient_descent(iris_x_b, iris_label, iris_M_b, th=iris_gd_theta, lr=IRIS_LEARNING_RATE, epoch=IRIS_N_ITER)


print("==========================")
print("=        실습 #2-3       =")
print("==========================")
accuracy = best_th_accuracy(iris_x_b, iris_label, iris_result[GD_RESULT_MAP["weight"]])

print("훈련결과, 정확도 {}%\nw0(sepal) = {}, w1(petal) = {}, w2(bias) = {}".format( \
                                                             accuracy * 100,\
                                                             iris_result[GD_RESULT_MAP["weight"]][0], \
                                                             iris_result[GD_RESULT_MAP["weight"]][1], \
                                                             iris_result[GD_RESULT_MAP["weight"]][2]))

def origin_data_space(weight):
    petal_min = (min(iris_feature[FEATURE_NAME["petal_length"]]))
    petal_max = (max(iris_feature[FEATURE_NAME["petal_length"]]))
    petal_lim = [petal_min-1 ,petal_max +1]

    sepal_min = (min(iris_feature[FEATURE_NAME["sepal_length"]]))
    sepal_max = (max(iris_feature[FEATURE_NAME["sepal_length"]]))
    sepal_lim = [sepal_min-1, sepal_max +1]
    
    petal_space = np.linspace(petal_lim[0], petal_lim[1], 1000)
    sepal_space = np.linspace(sepal_lim[0], sepal_lim[1], 1000)

    petal_mesh, sepal_mesh = np.meshgrid(petal_space, sepal_space)
    y_space = sepal_mesh * weight[0] + petal_mesh * weight[1] + weight[2]
    p_y_space = sigmoid(y_space)
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(iris_feature[FEATURE_NAME["petal_length"]], iris_feature[FEATURE_NAME["sepal_length"]], iris_label)
    ax.set_ylabel("sepal_length"); ax.set_xlabel("petal_length"); ax.set_zlabel("variety", rotation=0)
    ax.plot_surface(petal_mesh, sepal_mesh, p_y_space, cmap="plasma")
    plt.title("Emperical Solution")
    plt.grid(True)
    plt.show()
    
origin_data_space(iris_result[GD_RESULT_MAP["weight"]])

# 예측된 결과 값에 따라 1이면 남자 0이면 여자를 출력하기 위한 리스트 
RESULT_IRIS = ["Setosa", "Versicolor"]

def predict_iris(data, weight):
    """
    Args:
        data (List or number(int or float)): 예측을 원하는 값
        weight (np.array): 최적화된 가중치
    """
    # weight = bias + 특성(1) | 3 X 1 matrix
    w = np.array(weight)
    # data = 데이터 갯수 X 1(feature) martrix
    d = np.asarray(data)

    # d = 데이터 갯수 X 2(feature + bias) matrix  
    d_b = np.c_[d, np.ones(len(d))]
    zn = d_b @ w # linear equation
    probability = sigmoid(zn) # sigmoid
    y_pred = probability_func(probability) # Threshold = 0.5

    for i in range(len(y_pred)):
        print("데이터 {}의 예측 결과 : Result={}".format(d[i], RESULT_IRIS[y_pred[i]]))

predict_iris([[5.7,0.2],[6.4,4]], iris_result[GD_RESULT_MAP["weight"]])