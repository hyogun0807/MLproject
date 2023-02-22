import tensorflow_datasets
import numpy as np
import os
import json

class MultiPerceptron:
    def __init__(self, last_layer=None, node=0, activation_param="none", name=None):
        """_summary_
        Args:
            last_layer: 이전 층, 이전층의 데이터 값과 가중치를 통해 입력값을 계산하기 위해 가져옴
            node: 사용자 설정 값, 원하는 노드 수 설정
            activation_param: 활성화 함수 string 변수
            name: 각 층의 이름 설정
        """
        self.tolerance = 1e-15    # 허용 오차
        
        # 각 활성화 함수 string변수와 함수 매핑
        self.activate_func = {
            "none": self.none,
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "step": self.step,
            "tanh": self.tanh,
            "identity": self.identity,
            "softmax": self.softmax
        }
        
        self.last_layer = last_layer # 이전 층
        # 현재 층의 활성 함수 종류 저장
        self.activation_param = activation_param
        # 활성화 함수 매핑
        self.activate = self.activate_func[activation_param]  
        self.hidden_node = node # 은닉층의 노드의 갯수
        self.in_data = None     # 이전 층의 데이터와 가중치의 행렬 계산한 값이 들어온다.
        self.w = None           # 현재 층의 가중치
        self.out_data = None    # 입력으로 들어온 데이터를 활성화 함수를 거치게 한 값 다음 층이나 결과가 된다.
        
        self.name=name # Debug용, 각 은닉층의 이름 지정
    
    # Forward propagation
    def neuron(self):
        """_summary_
            퍼셉트론의 계산을 진행하는 주요 함수
            이전층의 출력값과 가중치 값의 행렬 계산 => 입력 => 활성화 함수 => 출력 => (다음 층)
        """
        # 이전 레이어와 가중치의 곱으로 퍼셉트론 계산
        tmp = self.layer_calc()
        
        # 활성화 전 곱으로 들어온 값, 입력 데이터에 저장
        self.in_data = tmp
        
        # 입력 데이터 활성화해서 넘겨줌
        tmp = self.activate(self.in_data)
        
        # 활성화를 거친 값을 출력 값으로 지정
        self.out_data = tmp
    
    
    def layer_calc(self):
        """_summary_
        이전 층의 출력값과 가중치 값의 행렬 계산
        Returns: 행렬 계산 값
        """
        
        x_b = np.append(self.last_layer.out_data,1)
        tmp = x_b.dot(self.last_layer.w)
   
        return tmp
    
    # Input later            
    def none(self, x):
        return x
    
    # 아래는 활성화 함수
    # 시그모이드 함수
    def sigmoid(self, x): 
        x = 1 / ( 1 + np.exp(-x) - 0.00000001)
        return x
    
    # 시그모이드 미분 값
    def sigmoid_prime(self, x): 
        return x * (1-x)

    # 계단 함수
    def step(self, x):
        return x > 0

    # 하이퍼볼릭 탄젠트 함수
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))\
                / (np.exp(x) + np.exp(-x))
           
    # 탄젠트 미분 함수     
    def tanh_prime(self, x):
        return 1 - self.tanh(x) ** 2

    # 항등 함수
    def identity(self, x):
        return x
    
    # Softmax 함수
    def softmax(self, x):
        e = np.exp(x - np.max(x))
        s = np.sum(e)
        return e / s
    
    # ReLU함수
    def relu(self, x):
        return np.array(list(map(lambda d: d if d>0 else 0, x)))
    
    def relu_prime(self, x):
        return np.array(list(map(lambda d: 1 if d>0 else 0, x)))
    
# 모델
class customModel:
    """_summary_
        모든 퍼셉트론 층을 담아두는 함수
    """
    def __init__(self,x=[], y=[], lr=0.0001, epoch=10000, w=[], test_x=[], test_y=[], mean=0, mu=0.5):
        """_summary
        Args:
            x: 입력 데이터, 입력데이터는 항상 numpy array로 준다.
            y: 레이블 데이터
            mean mu, 랜덤값을 위한 평균과 분산
            hidden_layer: 은닉층의 수
        """
        # 모델 전체 layer와 가중치 담아두는 리스트
        self.layer= []
        self.weight = w
        self.best_weight = w
        self.node_prime_model = []
        
        # Epoch에 따른 정확도, MSE 리스트
        self.accuracy_list = []
        self.mse_list = []
        
        self.best_acc = 0
        
        # 초기화
        # 랜덤하게 가중치 생성을 위한 평균과 분산
        self.mean = mean
        self.mu = mu
        
        # lr, epoch 초기화
        self.lr = lr
        self.epoch = epoch
        
        self.label=y        # 라벨 값
        self.in_data = x    # 처음 데이터 셋
        self.out_data = []  # 신경망 결과
        
        # 그래프용 & valid 테스트
        self.test_accuracy_list = []
        self.test_mse_list = []
        self.test_x = test_x
        self.test_y = test_y
        
        self.input_feature_cnt = 0 # 입력 데이터의 특성 수
        self.output_class_cnt = 0  # 출력 결과의 클래스 수
        self.hidden_layer_cnt = 0  # 은닉층의 수
        
        self.check_feature_class_cnt() # 입력 데이터의 특성 수와 출력 결과 클래스 수 파악
    
    # 모델 클래스 생성시 입력데이터를 통해 입력 특성 수 지정하고 
    # 라벨데이터를 통해 출력 클래스 개수 정하는 함수
    
    def check_feature_class_cnt(self):
        # numpy인 경우에만 사용
        if type(self.in_data) is np.ndarray:
            self.input_feature_cnt = len(self.in_data[0])
            
        if type(self.label) is np.ndarray:
            self.output_class_cnt = len(self.label[0])
    
    # 매 에폭마다 데이터를 셔플하여 데이터 순서를 연관하게 하지 않는다.
    def dataShuffle(self):
        # 데이터 갯수 만큼 랜덤 인덱스를 만든다.
        index = np.random.permutation(len(self.train_x_data))
        length = len(self.train_x_data)
        # 각 인덱스에 맞게 라벨과 값을 섞어준다.
        temp_x = self.train_x_data.copy()
        temp_y = self.train_label.copy()
        self.train_x_data = np.array([temp_x[index[i]] for i in range(length)] )
        self.train_label = np.array([temp_y[index[i]] for i in range(length)])
    
    # valid 함수, 나중에 텀프할때 쓸거같아 만들었습니다.
    def valid(self):
        # 현재 가중치를 통해 valid 데이터를 순전파하여
        # 새로운 데이터로 overfitting한지 확인
        accuracy = 0
        mse = 0
        length = len(self.test_x)
        # 한 에폭마다 전체 데이터 셋을 전부 돌아야한다.
        for iter in range(len(self.test_x)):    
            # 하나의 train데이터가 입력 데이터로 된다.
            self.layer[0].in_data = self.test_x[iter]
            self.layer[0].out_data = self.test_x[iter]
            
            # 순전파 시작
            self.foward_propagation()
            
            # 하나의 데이터가 정확한지와 mse를 구한다.
            acc, one_mse, _, _ = self.one_mse_acc(self.layer[-1], self.test_y[iter])
            accuracy += acc
            mse += one_mse
        
        self.test_accuracy_list.append(accuracy / length)
        self.test_mse_list.append(mse / length)
    
    # 설계한 은닉층과 출력층을 실질적으로 데이터를 넣어 실행시키는 함수 
    def fit(self):
        """
            입력 데이터 : row, col
        """
       
        # 모델로 들어온 데이터 입력 데이터에 넣는다.
        # layer0 = 입력층, 입력층은 아무 것도 설정하지 않으므로 입력 데이터 -> 출력 데이터
        length = len(self.in_data)
        
        # 시작시 train셋에 입력 데이터를 넣느다.
        self.train_x_data = self.in_data.copy()
        self.train_label = self.label.copy()

        # 가중치 업데이트 반복횟수
        for i in range(self.epoch):
            # 매 에폭마다 셔플
            self.dataShuffle()
            
            # 평균과 정확도 계산을 위한 변수
            accuracy = 0
            mse = 0
            
            self.valid()
            # 한 에폭마다 전체 데이터 셋을 전부 돌아야한다.
            for iter in range(len(self.train_x_data)):    
                # 하나의 train데이터가 입력 데이터로 된다.
                self.layer[0].in_data = self.train_x_data[iter]
                self.layer[0].out_data = self.train_x_data[iter]
                
                # 순전파 시작
                self.foward_propagation()
                
                # 하나의 데이터가 정확한지와 mse를 구한다.
                acc, one_mse, _, _ = self.one_mse_acc(self.layer[-1], self.train_label[iter])
                accuracy += acc
                mse += one_mse
                
                # 역전파 시작
                self.back_propagation(iter)
            
            if accuracy/ length > self.best_acc:
                self.best_acc = accuracy / length
                for i in range(len(self.weight)):
                    self.best_weight[i]= self.layer[i].w
                
            
            self.accuracy_list.append(accuracy / length)
            self.mse_list.append(mse / length)
            
            if i % 100 == 0:
                print("[Epoch] : {} ===> ( [MSE] : {}, [Accuracy] : {} ), ( [test_MSE] : {}, [test_acc] : {} )"\
                    .format(i, self.mse_list[i], self.accuracy_list[i], self.test_mse_list[i], self.test_accuracy_list[i]))
        
        # 학습 끝난 최종 가중치 전체 weight에 저장
        # 최종 가중치로 테스트에 사용하기 위해 저장, file로 넣어 둔다
        for i in range(len(self.weight)):
            self.weight[i]= self.layer[i].w
        
        # 마지막 back propagation을 통해 업데이트 된 가중치를
        # 순전파를 한번더 진행한다.
        accuracy = 0
        mse = 0
        for iter in range(len(self.train_x_data)):
            self.layer[0].in_data = self.train_x_data[iter]
            self.layer[0].out_data = self.train_x_data[iter]
            
            self.foward_propagation()
                
            acc, one_mse, _ , _= self.one_mse_acc(self.layer[-1], self.train_label[iter])
            if acc == 1:
                self.out_data.append("True")
            else:
                self.out_data.append("False")
            accuracy += acc
            mse += one_mse
        self.accuracy_list.append(accuracy / length)
        self.mse_list.append(mse / length)
    
    def foward_propagation(self):
        # 설계한 은닉층에서 출력층까지 돌린다.
        # MultiPerceptron 클래스에서 내부적으로 이전 layer를 가지고 있으므로 돌려주기만 하면 된다.
        for i in range(1,self.hidden_layer_cnt+2):
            self.layer[i].neuron()
     
     # 역전파 함수       
    def back_propagation(self, iter):    
            # Chain
            # 손실함수 미분
            cmp = self.cost_mse_prime(self.train_label[iter], self.layer[-1].out_data)

            # 활성화 함수 미분
            sig_prime = self.find_active_prime(self.layer[-1])
            
            # 활성화 함수까지의 Chain 결과
            node_prime = self.node_prime_model.copy()
            node_prime[-1] = [cmp[temp_i] * sig_prime[temp_i] for temp_i in range(len(cmp))]
            
            weight_prime = []
            # 은닉층 뒤에서 부터 새로운 가중치 저장 
            for layer_i in range(self.hidden_layer_cnt+1, 0,-1):            
                # 역전파의 다음 노드(입력층과 가까운 노드)의 활성화 함수 미분값 
                sig_prime = self.find_active_prime(self.layer[layer_i-1])
                
                # 가중치 미분값 구하는 파트
                out_d = np.r_[self.layer[layer_i].last_layer.out_data, [1]].reshape(-1,1)
                node_p = np.array(node_prime[layer_i]).reshape(1,-1)
                weight_prime.insert(0, out_d.dot(node_p))
                
                new_node = self.layer[layer_i].last_layer.w.dot(node_prime[layer_i])
                node_prime[layer_i-1] = new_node[:-1] * sig_prime
            
            # 가중치 갱신
            for i in range(len(weight_prime)):
                self.layer[i].w = self.layer[i].w - self.lr * weight_prime[i]
    

           
    # 하나의 데이터에 대해 정화도와 MSE구한다.
    def one_mse_acc(self, layer, label):
        accuracy = 0
        mse = 0
        
        # 순전파 결과에서 가장 높은 값과 라벨의 가장 높은 값이
        # 같으면 정확도를 1 높인다.
        data_index = np.argmax(layer.out_data)
        label_index = np.argmax(label)
        if label_index == -1:
            print("[Error] : label에 1 존재하지 X")
        elif data_index == label_index:
            accuracy = 1
        
        # MSE 구하는 함수
        mse = self.cost_one_mse(label, layer.out_data)
        return accuracy, mse, data_index, label_index
    
    # 하나의 데이터 MSE를 구하는 함수
    def cost_one_mse(self, label, y_pred):
        er = (label - y_pred) ** 2
        return sum(er)
    
    # 손실함수 미분 함수
    def cost_mse_prime(self, label, out_data):
        return (-2)*(label - out_data) 
    
    # 활성 함수를 미분한 값을 가져오는 함수
    def find_active_prime(self, layer):
        # Input ddata
        if layer.activation_param == 'none':
            return layer.none(layer.out_data)
        # Rule data
        if layer.activation_param == 'relu':
            return layer.relu_prime(layer.out_data)
        if layer.activation_param != "sigmoid":
            #print("현재 구현 안함, 자동으로 sigmoid 변경")
            pass
        return layer.sigmoid_prime(layer.out_data)
    
    # 테스트 데이터를 넣고 돌리는 함수 
    def test(self):
        # 넣은 데이터를 최적의 가중치로 순전파하여
        # 결과값을 확인 후 정확도와 mse를 반환한다.
        length = len(self.in_data)
        accuracy = 0
        mse = 0
        for iter in range(len(self.in_data)):
            self.layer[0].in_data = self.in_data[iter]
            self.layer[0].out_data = self.in_data[iter]
            # 순전파
            self.foward_propagation()
            # 정확도, mse, 예측과 라벨의 인덱스 값    
            acc, one_mse, pred_index, label_index = self.one_mse_acc(self.layer[-1], self.label[iter])
            if acc == 1:
                self.out_data.append("True")
            else:
                self.out_data.append("False")
            accuracy += acc
            mse += one_mse
            print("데이터: {},\t라벨: {},  예측: {}\t결과: {}".format(self.in_data[iter], label_index+1, pred_index+1, self.out_data[iter]))
        return accuracy / length, mse / length
        
    
    # 가중치 세팅
    def set_weight(self):
        # Input
        # 입력층 - 은닉1층 
        # matrix = 입력 특성수 x 은닉1층의 노드 수
        if len(self.weight) != 0:
            for i in range(self.hidden_layer_cnt+1):
                self.weight[i] = np.array(self.weight[i])
                self.layer[i].w = self.weight[i] # 자신의 클래스에 값 저장
            
            return
        
        w =np.random.normal(self.mean,self.mu, size=(self.input_feature_cnt + 1,self.layer[1].hidden_node))
        self.weight.append(w) # 설정한 가중치 값 model에 list로 저장
        self.layer[0].w = w   # 각 layer(MultiPerceptron 클래스)에 자신의 가중치 저장
        
        # Hidden - Output
        # matrix = 은닉층의 노드 x 출력층 클래스 수
        # 은닉층만 적용
        for i in range(1, self.hidden_layer_cnt+1):
            w =np.random.normal(self.mean,self.mu, size=(self.layer[i].hidden_node+1,self.layer[i+1].hidden_node))
            self.layer[i].w = w # 자신의 클래스에 값 저장
            self.weight.append(w) # 설정한 가중치 값 model에 list로 저장 
        
            
    def set_layer(self):
        # 입력 층 설정
        # 입력층의 노드 = 입력 데이터의 특성 수
        input_data = MultiPerceptron(node=self.input_feature_cnt, name="Input_Layer")
        self.layer.append(input_data)
        
        # 은닉층1: {활성화함수:tanh, node=7}으로 지정
        layer1= MultiPerceptron(input_data, node=8, activation_param="sigmoid", name="Layer1")
        self.layer.append(layer1)
        
        # 출력층: {활성화함수:softmax, node=출력 클래스 갯수}으로 지정
        output = MultiPerceptron(layer1, node=self.output_class_cnt, activation_param="sigmoid", name="Output_Layer")
        self.layer.append(output)
        
        # 모든 레이어 - 입력 - 출력3
        self.hidden_layer_cnt = len(self.layer) - 2
        
        # 모델 설계후 각 지정한 노드를 통해 가중치 설정
        self.set_weight()
        
        for layer in self.layer:
            self.node_prime_model.append([0 for _ in range(layer.hidden_node)])
            
mnist = tensorflow_datasets.load('mnist')
train_set = mnist["train"]
# (train_images, train_labels), (test_images, test_labels) = mnist

data_list = []
label_list = []
one = []
one_label = []
two = []
two_label = []
zero=[]
zero_label = []

t = [[1,0,0], [0,1,0], [0,0,1]]

for d in train_set:
    if d["label"] == 0:
      zero.append(d["image"])
      zero_label.append([1,0,0])
    elif d["label"] == 1:
      one.append(d["image"])
      one_label.append([0,1,0])
    elif d["label"] == 2:
      two.append(d["image"])
      two_label.append([0,0,1])
      
data_list = zero + one + two
label_list = zero_label + one_label + two_label
data_list = np.array(data_list)
label_list = np.array(label_list)
one_index = len(one)
two_index = len(two)
zero_index = len(zero)    

# 확률밀도값 구하는 함수
def calc_pdf(result):
    n_sum = sum(result)
    return np.array(result) / n_sum
# 가로축과 세로축의 projection 함수
def calc_pro(temp):
    result = []
    for i in temp:
        result.append(sum(i))
    return result, calc_pdf(result)

# 가로축 Projection축 구하는 함수
def col_pdf(input_data):
    return calc_pro(input_data.T[0])

# 세로축 Projection축 구하는 함수
def row_pdf(input_data):
    input_data = input_data.T[0].T
    return calc_pro(input_data)

# 평균 구하는 함수
def expect(sun_r, input_data):
    length = len(input_data)
    val = 0
    for i in range(length):
        val += sun_r[i] * input_data[i]
    return val

# 분산 구하는 함수
def variance(sun_r, input_data, m):
    length = len(input_data)
    var = 0
    for i in range(length):
        var += (sun_r[i] - m)**2 * input_data[i]
    return var

# 대각선 축 구하는 함수
def diagonal(input_data):
    input_data = input_data.T[0].T
    length = len(input_data)
    result = []
    for i in range(0,length):
        result.append(input_data[i][i])
    return np.array(result)

# 역 대각선 축 구하는 함수
def reverse_diagonal(input_data):
    input_data = input_data.T[0].T
    length = len(input_data)
    result = []
    for i in range(0,length):
        result.append(input_data[i][length-1-i])
    return np.array(result)

# 대각선에서 갯수 구하는 함수
def calc_count(input_data):
    result = 0
    for data in input_data:
        if data == 0:
            result += 1
    return result

# 특성
# 특성1 : 가로축 Projection 평균
def col_pdf_mean(input_data):
    sun_r, temp = col_pdf(input_data)
    return expect(sun_r, temp)
# 특성2 : 가로축 Projection 분산
def col_pdf_var(input_data):
    sun_r, temp = col_pdf(input_data)
    m = expect(sun_r, temp)
    return variance(sun_r, temp, m)

# 특성3 : 세로축 Projection 평균
def row_pdf_mean(input_data):
    sum_r, temp = row_pdf(input_data)
    return expect(sum_r, temp)
# 특성4 : 세로축 Projection 분산
def row_pdf_var(input_data):
    sum_r, temp = row_pdf(input_data)
    m = expect(sum_r, temp)
    return variance(sum_r, temp, m)

# 특성5 : 대각선 확률밀도 평균
def dig_pdf_mean(input_data):
    temp = diagonal(input_data)
    temp_pdf = calc_pdf(temp)
    return expect(temp, temp_pdf)
# 특성6 : 대각선 확률밀도 분산
def dig_pdf_var(input_data):
    temp = diagonal(input_data)
    temp_pdf = calc_pdf(temp)
    m = expect(temp, temp_pdf)
    return variance(temp, temp_pdf, m)

# 특성7 : 대각선 확률밀도 0 갯수
def dig_cnt(input_data):
    temp = diagonal(input_data)
    return calc_count(temp)

# 특성8 : 역 대각선 확률밀도 평균
def re_dig_pdf_mean(input_data):
    temp = reverse_diagonal(input_data)
    temp_pdf = calc_pdf(temp)
    return expect(temp, temp_pdf)
# 특성9 : 역 대각선 확률밀도 분산
def re_dig_pdf_var(input_data):
    temp = reverse_diagonal(input_data)
    temp_pdf = calc_pdf(temp)
    m = expect(temp, temp_pdf)
    return variance(temp, temp_pdf, m)

# 특성10 : 역 대각선 확률밀도 0 갯수
def re_dig_cnt(input_data):
    temp = reverse_diagonal(input_data)
    return calc_count(temp)

# col_var, row_val, re_dig_var, dig_var, dig_count
def feature_1(input_data):
    return col_pdf_var(input_data)
def feature_2(input_data):
    return row_pdf_mean(input_data)
def feature_3(input_data):
    return row_pdf_var(input_data)
def feature_4(input_data):
    return dig_cnt(input_data)
def feature_5(input_data):
    return re_dig_cnt(input_data)



def get_feature():
    x_set= np.array([], dtype="float32")
    x_set = np.resize(x_set, (0,5))
    for d in data_list:
        x0= feature_1(d)
        x1= feature_2(d)
        x2= feature_3(d)
        x3= feature_4(d)
        x4= feature_5(d)
        
        features = np.array([x0,x1,x2,x3,x4], dtype='float32')
        # print(features.shape)
        features= np.resize(features, (1,5))
        x_set=np.concatenate((x_set, features), axis=0)
    
    return x_set

data_x = get_feature()


def show_feature():
    feature = [[] for i in range(10)]
    for d in data_x:
        
        feature[0].append(col_pdf_mean(d))
        feature[1].append(col_pdf_var(d))
        feature[2].append(row_pdf_mean(d))
        feature[3].append(row_pdf_var(d))
        feature[4].append(dig_pdf_mean(d))
        feature[5].append(dig_pdf_var(d))
        feature[6].append(dig_cnt(d))
        feature[7].append(re_dig_pdf_mean(d))
        feature[8].append(re_dig_pdf_var(d))
        feature[9].append(re_dig_cnt(d))
    print(feature[0][0], feature[1][0])
    print(feature[2][0],feature[3][0])
    print(feature[4][0],feature[5][0],feature[6][0])
    print(feature[7][0],feature[8][0],feature[9][0])
    feature = np.array(feature)
    lab = ["가로축 expect", "가로축 분산", "세로축 expect", "세로축 분산", "대각선 expect", "대각선 분산", "대각선 0갯수","역대각선 expect", "역대각선 분산", "역대각선 0갯수"]
    for i in range(len(feature)):
        zero, one, two = feature[i][:500], feature[i][500:1000], feature[i][1000:]
        print("="*30)
        print("[{}]".format(lab[i]))
        print("0: 최대={}\t최소={}\t 값 평균={}".format(max(zero), min(zero), sum(zero)/500))
        print("1: 최대={}\t최소={}\t 값 평균={}".format(max(one), min(one), sum(one)/500))
        print("2: 최대={}\t최소={}\t 값 평균={}".format(max(two), min(two), sum(two)/500))

# show_feature()


print(data_x.shape)
main_dir = 'C:\\Users\\parks\\Desktop\\대학\\4학년 1학기\\ml\\term_project'
def test_data():
    # 최종 lr = 0.005
    lr = 0.0011
    # 값 저장한 파일에서 데이터 읽어온다.
    save_path = os.path.join(main_dir, 'save_result')
    with open(os.path.join(save_path,"node_9.json".format(lr)), 'r') as f:
        json_data = json.load(f)
        acc_list = json_data["accuracy"]
        max_acc_index = np.argmax(acc_list)
        weight = json_data["weight"]
        best_weight = json_data["best_w"]
    
    # 테스트 데이터 넣고 반복
    model = customModel(data_x, label_list, w=weight,epoch=20000, lr=lr)
    model.set_layer() # 계층 설계
    acc, mse = model.test() # 데이터 적용 및 실행
    
    print("최종 결과 : Acc={}, MSE={}".format(acc,mse))
    
test_data()