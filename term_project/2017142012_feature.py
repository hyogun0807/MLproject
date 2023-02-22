import numpy as np
import math as ma


############################################################################
def feature_1(input_data):
    # 특징 후보 2번: 가로축 Projection => 확률밀도함수로 변환 => 분산
    result = []
    for i in input_data.T:
        result.append(sum(i))

    # 확률밀도함수 변환
    n_sum = sum(result)
    pdf =  np.array(result) / n_sum

    # 기대치 구함
    length = len(pdf)
    expect = 0
    for i in range(length):
        expect += result[i] * pdf[i]
    
    # 분산 구함    
    length = len(input_data)
    var = 0
    for i in range(length):
        var += (result[i] - expect)**2 * pdf[i]
    return var
    
############################################################################

############################################################################
def feature_2(input_data):
    # 특징 후보 3번: 세로축 Projection => 확률밀도함수로 변환 => 기댓값
    result = []
    for i in input_data:
        result.append(sum(i))

    # 확률밀도함수 변환
    n_sum = sum(result)
    pdf = np.array(result) / n_sum
    
    # 기대치
    length = len(pdf)
    expect = 0
    for i in range(length):
        expect += result[i] * pdf[i]
    return expect
############################################################################

############################################################################
def feature_3(input_data):
    # 특징 후보 4번: 세로축 Projection => 확률밀도함수로 변환 => 분산
    result = []
    for i in input_data:
        result.append(sum(i))

    # 확률밀도함수 변환
    n_sum = sum(result)
    pdf = np.array(result) / n_sum
    
    # 기대치
    length = len(pdf)
    expect = 0
    for i in range(length):
        expect += result[i] * pdf[i]
        
    # 분산
    length = len(pdf)
    var = 0
    for i in range(length):
        var += (result[i] - expect)**2 * pdf[i]
    return var
############################################################################

############################################################################
def feature_4(input_data):
    # 특징 후보 7번: Diagonal 원소 배열 추출 => 0의 개수 
    
    # 원소 추출
    length = len(input_data)
    result = []
    for i in range(0,length):
        result.append(input_data[i][i])
    result = np.array(result)
    
    # 0인 갯수 세기
    cnt = 0
    for data in result:
        if data == 0:
            cnt += 1
    return cnt
############################################################################

############################################################################
def feature_5(input_data):
    # 특징 후보 10번: Anti-Diagonal 원소 배열 추출 => 0의 개수
    
    # 원소 추출
    length = len(input_data)
    result = []
    for i in range(0,length):
        result.append(input_data[i][length-1-i])
    result = np.array(result)
    
    # 0인 갯수 세기
    cnt = 0
    for data in result:
        if data == 0:
            cnt += 1
    return cnt
############################################################################
