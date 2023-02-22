import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

print("==========================")
print("|        실습 # 1        |")
print("==========================")

# 제공해준 퍼셉트론 클래스
class perceptron:
    def __init__(self, w):
        self.w =w
        
    def output(self, x):
        # 데이터의 수식 행렬 연산
        tmp = self.w.T.dot(np.append(1,x)) # 3x1 3x1
        
        # 결과 값을 1과 0으로 출력
        result = 1.0 * (tmp > 0)
        return result   
  
print("\n[AND 논리 연산 결과]")  
# AND
w_and = np.array([-1.2, 1, 1])
and_gate = perceptron(w_and)
x_list = [[0,0],[1,0],[0,1],[1,1]]
for x in x_list:
    print("x = {} ====> {}".format(x, and_gate.output(x)))
    
print("\n[OR 논리 연산 결과]")
# OR
w_or = np.array([-0.5, 1, 1])
or_gate = perceptron(w_or)
x_list = [[0,0],[1,0],[0,1],[1,1]]
for x in x_list:
    print("x = {} ====> {}".format(x, or_gate.output(x)))
    
print("\n[XOR 논리 연산 결과]")
# XOR
x_list = [[0,0],[1,0],[0,1],[1,1]]
for x in x_list:
    new_x = np.c_[1 - and_gate.output(x), or_gate.output(x)]
    print("x = {} ====> {}".format(x, and_gate.output(new_x)))

print("==========================")
print("|        실습 # 2        |")
print("==========================")

# 각 층을 클래스로 설정
class MultiPerceptron:
    def __init__(self, last_layer=None, node=0, activation_param="relu", name=None):
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
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "step": self.step,
            "tanh": self.tanh,
            "identity": self.identity,
            "softmax": self.softmax
        }
        
        self.last_layer = last_layer # 이전 층
        # activation_param을 통해 함수 매핑하여 넣어준다.
        self.activate = self.activate_func[activation_param]  
        self.hidden_node = node # 은닉층의 노드의 갯수
        self.in_data = None     # 이전 층의 데이터와 가중치의 행렬 계산한 값이 들어온다.
        self.w = None           # 현재 층의 가중치
        self.out_data = None    # 입력으로 들어온 데이터를 활성화 함수를 거치게 한 값 다음 층이나 결과가 된다.
        
        self.name=name # Debug용, 각 은닉층의 이름 지정
    
    
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
        tmp = self.activate_loop(tmp)
        
        # 활성화를 거친 값을 출력 값으로 지정
        self.out_data = tmp
    
    
    def layer_calc(self):
        """_summary_
        이전 층의 출력값과 가중치 값의 행렬 계산
        Returns: 행렬 계산 값
        """
        # 계산 후
        x_b = np.c_[self.last_layer.out_data, np.ones(len(self.last_layer.out_data))]
        tmp = x_b.dot(self.last_layer.w)
        
        return tmp
    
    def activate_loop(self, tmp):
        """_summary_
        각 데이터 셋에 대해서 적용 시키기위한 함수
        Args: tmp: 행렬 계산된 값, 
        Returns:  활성화 함수를 거친 값
        """
        for i, data in enumerate(tmp):
            tmp[i] = self.activate(data)
        return tmp
                
    # 아래는 활성화 함수
    # 시그모이드 함수
    def sigmoid(self, x): 
        x = 1 / ( 1 + np.exp(-x))
        return x

    # 계단 함수
    def step(self, x):
        return x > 0

    # 하이퍼볼릭 탄젠트 함수
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x))\
                / (np.exp(x) + np.exp(-x))

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
        return list(map(lambda d: d if d>0 else 0, x))
    
# 모델
class customModel:
    """_summary_
        모든 퍼셉트론 층을 담아두는 함수
    """
    def __init__(self,x=[], y=[], mean=0, mu=2):
        """_summary
        Args:
            x: 입력 데이터, 입력데이터는 항상 numpy array로 준다.
            y: 레이블 데이터
            mean mu, 랜덤값을 위한 평균과 분산
            hidden_layer: 은닉층의 수
        """
        # 모델 전체 layer와 가중치 담아두는 리스트
        self.layer= []
        self.weight = []
        
        # 랜덤하게 가중치 생성을 위한 평균과 분산
        self.mean = mean
        self.mu = mu
        
        self.label=y        # 라벨 값
        self.in_data = x    # 처음 데이터 셋
        self.out_data = []  # 출력을 위한 값
        self.y_pred = []    # 출력을 통해 예측한 값
        
        self.input_feature_cnt = 0 # 입력 데이터의 특성 수
        self.output_class_cnt = 0  # 출력 결과의 클래스 수
        self.hidden_layer_cnt = 0  # 은닉층의 수
        
        self.check_feature_class_cnt() # 입력 데이터의 특성 수와 출력 결과 클래스 수 파악
      
    # 퍼셉트론을 통해 도출한 결과 반환하는 함수          
    def return_output(self):
        return self.out_data
    
    # softmax함수를 나온 값을 통해 결과 값 도출
    def softmax_predict(self):
        for d in self.out_data:
            # 입력 데이터마다 각각의 최대 값의 인덱스를 통해 결과 값 도출
            self.y_pred.append(np.argmax(d)+1)
    
        return self.y_pred
    
    # 설계한 은닉층과 출력층을 실질적으로 데이터를 넣어 실행시키는 함수 
    def fit(self):
        """
            입력 데이터 : row, col
        """
        
        # 모델로 들어온 데이터 입력 데이터에 넣는다.
        # layer0 = 입력층, 입력층은 아무 것도 설정하지 않으므로 입력 데이터 -> 출력 데이터
        self.layer[0].in_data = self.in_data
        self.layer[0].out_data = self.in_data
        
        # 설계한 은닉층에서 출력층까지 돌린다.
        # MultiPerceptron 클래스에서 내부적으로 이전 layer를 가지고 있으므로 돌려주기만 하면 된다.
        for i in range(1,self.hidden_layer_cnt+2):
            self.layer[i].neuron()
        
        # 마지막 출력층의 활성화 함수까지 거친 데이터를 최종 출력 데이터에 넣는다. 
        self.out_data = self.layer[-1].out_data
        
    
    # 모델 클래스 생성시 입력데이터를 통해 입력 특성 수 지정하고 
    # 라벨데이터를 통해 출력 클래스 개수 정하는 함수
    def check_feature_class_cnt(self):
        # numpy인 경우에만 사용
        if type(self.in_data) is np.ndarray:
            self.input_feature_cnt = len(self.in_data[0])
            
        if type(self.label) is np.ndarray:
            self.output_class_cnt = len(self.label[0])
    
    # 가중치 세팅
    def set_weight(self, auto=True):
        """_summary_
        Args:
            auto: True=자동으로 가중치값을 설정해 준다. False=실습2번
        """
        # 실습 2번을 위한 자동 설정 값
        if not auto:
            index = 0
            # Layer 1
            self.weight.append(np.array([
                [0.1,0.2,0.3],
                [0.1,0.3,0.5],
                [0.2,0.4,0.6]
            ]))
            self.layer[index].w = self.weight[index]

            index += 1
            # Layer 2
            self.weight.append(np.array([
                [0.1,0.2],
                [0.1,0.4],
                [0.2,0.5],
                [0.3,0.6]
            ]))
            self.layer[index].w = self.weight[index]
            return

        # AUTO
        # Input
        # 입력층 - 은닉1층 
        # matrix = 입력 특성수 x 은닉1층의 노드 수
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
        
    # 실습#2 Case 1
    def set_layer_case1(self):
        # input = 이전 레이어
        # node = 원하는 입력 층 수
        # name = 층의 이름 설정
        # activation_param = 활성화 함수, 자신한테 들어온 데이터를 활성화 함수를 통해 비선형적이게 만들어 다음 층으로 넘긴다. 
        
        # 입력 층 설정
        # 입력층의 노드 = 입력 데이터의 특성 수
        input = MultiPerceptron(node=self.input_feature_cnt, name="Input_Layer")
        self.layer.append(input)
        
        # 은닉층1: {활성화함수:시그모이드, node=3}으로 지정
        layer1 = MultiPerceptron(input, node=3, activation_param="sigmoid", name="Layer1")
        self.layer.append(layer1)
        
        # 출력층: {활성화함수:identity(항등), node=출력 클래스 갯수}으로 지정
        output = MultiPerceptron(layer1, node=self.output_class_cnt, activation_param="identity", name="Output_Layer")
        self.layer.append(output)
        
        # 모든 레이어 - 입력 - 출력
        self.hidden_layer_cnt = len(self.layer) - 2
        
        # 모델 설계후 각 지정한 노드를 통해 가중치 설정
        self.set_weight(auto=False)
        
     # 실습#2 Case 2
    def set_layer_case2(self):
         # 입력 층 설정
        # 입력층의 노드 = 입력 데이터의 특성 수
        input_data = MultiPerceptron(node=self.input_feature_cnt, name="Input_Layer")
        self.layer.append(input_data)
        
        # 은닉층1: {활성화함수:시그모이드, node=3}으로 지정
        layer1 = MultiPerceptron(input_data, node=3, activation_param="sigmoid", name="Layer1")
        self.layer.append(layer1)
        
        # 출력층: {활성화함수:softmax, node=출력 클래스 갯수}으로 지정
        output = MultiPerceptron(layer1, node=self.output_class_cnt, activation_param="softmax", name="Output_Layer")
        self.layer.append(output)
        
        # 모든 레이어 - 입력 - 출력
        self.hidden_layer_cnt = len(self.layer) - 2
        
        # 모델 설계후 각 지정한 노드를 통해 가중치 설정
        self.set_weight(auto=False)
        
    # 실습#2 Case 3  
    def set_layer_case3(self):
        # 입력 층 설정
        # 입력층의 노드 = 입력 데이터의 특성 수
        input_data = MultiPerceptron(node=self.input_feature_cnt, name="Input_Layer")
        self.layer.append(input_data)
        
        # 은닉층1: {활성화함수:relu, node=3}으로 지정
        layer1 = MultiPerceptron(input_data, node=3, activation_param="relu", name="Layer1")
        self.layer.append(layer1)
        
        # 출력층: {활성화함수:identity, node=출력 클래스 갯수}으로 지정
        output = MultiPerceptron(layer1, node=self.output_class_cnt, activation_param="identity", name="Output_Layer")
        self.layer.append(output)
        
        # 모든 레이어 - 입력 - 출력
        self.hidden_layer_cnt = len(self.layer) - 2
        # 모델 설계후 각 지정한 노드를 통해 가중치 설정
        self.set_weight(auto=False)
        
    # 실습#2 Case 4
    def set_layer_case4(self):
        # 입력 층 설정
        # 입력층의 노드 = 입력 데이터의 특성 수
        input_data = MultiPerceptron(node=self.input_feature_cnt, name="Input_Layer")
        self.layer.append(input_data)
        
        # 은닉층1: {활성화함수:relu, node=3}으로 지정
        layer1= MultiPerceptron(input_data, node=3, activation_param="relu", name="Layer1")
        self.layer.append(layer1)
        
        # 출력층: {활성화함수:softmax, node=출력 클래스 갯수}으로 지정
        output = MultiPerceptron(layer1, node=self.output_class_cnt, activation_param="softmax", name="Output_Layer")
        self.layer.append(output)
        
        # 모든 레이어 - 입력 - 출력
        self.hidden_layer_cnt = len(self.layer) - 2
        # 모델 설계후 각 지정한 노드를 통해 가중치 설정
        self.set_weight(auto=False)
      
    # 실습 #3 
    def set_layer(self):
        # 입력 층 설정
        # 입력층의 노드 = 입력 데이터의 특성 수
        input_data = MultiPerceptron(node=self.input_feature_cnt, name="Input_Layer")
        self.layer.append(input_data)
        
        # 은닉층1: {활성화함수:tanh, node=7}으로 지정
        layer1= MultiPerceptron(input_data, node=10, activation_param="tanh", name="Layer1")
        self.layer.append(layer1)
        
        # 출력층: {활성화함수:softmax, node=출력 클래스 갯수}으로 지정
        output = MultiPerceptron(layer1, node=self.output_class_cnt, activation_param="softmax", name="Output_Layer")
        self.layer.append(output)
        
        # 모든 레이어 - 입력 - 출력
        self.hidden_layer_cnt = len(self.layer) - 2
        
        # 모델 설계후 각 지정한 노드를 통해 가중치 설정
        self.set_weight()

x = [1.0, 0.5]

case1 = customModel(np.array([x])) # case1 모델 생성
case1.set_layer_case1()            # 모델 만들기
case1.fit()                        # 모델 학습
print("[Case1] x:{}, y:{}".format(x, case1.return_output()))

case2 = customModel(np.array([x])) # case2 모델 생성
case2.set_layer_case2()            # 모델 만들기
case2.fit()                        # 모델 학습
print("[Case2] x:{}, y:{}".format(x, case2.return_output()))

case3 = customModel(np.array([x])) # case3 모델 생성
case3.set_layer_case3()            # 모델 만들기
case3.fit()                        # 모델 학습
print("[Case3] x:{}, y:{}".format(x, case3.return_output()))

case4 = customModel(np.array([x])) # case4 모델 생성
case4.set_layer_case4()            # 모델 만들기
case4.fit()                        # 모델 학습
print("[Case4] x:{}, y:{}".format(x, case4.return_output()))


print("==========================")
print("|        실습 # 3        |")
print("==========================")

# main_dir
main_dir = 'C:\\Users\\parks\\Desktop\\대학\\4학년 1학기\\ml\\5week'
# 데이터 셋 읽어들입니다.
dataFrame = pd.read_csv(os.path.join(main_dir, "NN_data.csv"))
data = dataFrame.iloc[:,1:] # 첫번째 열의 경우 필요 없어서 제외
data_x = data.iloc[:,:3].to_numpy() # 0~2까지 3개의 특성에 따른 데이터들 data_x로 변환
data_y = data.iloc[:,3].to_numpy()  # 마지막 라벨 열 data_y로 변환

# noise 추가
noise =np.random.randn(len(data_x),len(data_x[0])) # -1 ~ 1까자의 임의의 랜덤값 데이터 숫자 x 특성의 수 만큼 생성
data_x_noise = data_x + noise # x데이터에 noise값 추가

# 클래스 분리
class1 = data_x_noise[:300]
class2 = data_x_noise[300:600]
class3 = data_x_noise[600:]

# 특성 라벨 매핑
FEATURE_LABEL = {
    "x0": 0,
    "x1": 1,
    "x2": 2
}

ax = plt.figure().add_subplot(111, projection='3d')
ax.scatter(class1.T[FEATURE_LABEL["x0"]], class1.T[FEATURE_LABEL["x1"]], class1.T[FEATURE_LABEL["x2"]],  facecolors='none', edgecolors='blue', label="1")
ax.scatter(class2.T[FEATURE_LABEL["x0"]], class2.T[FEATURE_LABEL["x1"]], class2.T[FEATURE_LABEL["x2"]],  facecolors='none', edgecolors='red', label="2")
ax.scatter(class3.T[FEATURE_LABEL["x0"]], class3.T[FEATURE_LABEL["x1"]], class3.T[FEATURE_LABEL["x2"]],  facecolors='none', edgecolors='orange', label="3")
plt.legend(loc="upper right")
plt.show()

# 데이터의 클래스 갯수 구하는 함수
data_encode_label = []
def find_class_num(y):
    """_summary_
    Args:
        y (numpy.ndarray): 데이터 셋의 Y값

    Returns:
        list : 서로다른 클래스 리스트 반환
    """
    class_list = {}
    # y값에서 같은 값만 넣기 위해 남겨둔다.
    index = 0
    for d in y:
        # 클래스 리스트에 기존에 넣은 클래스가 없으면 넣는다.
        if d not in class_list:
            class_list[d] = index
            index += 1
    return class_list

def one_hot_encode(y):
    """_summary_
    Args:
        y (numpy.ndarray): 데이터 셋의 Y값

    Returns:
        _type_: _description_
    """
    class_list = find_class_num(y)
    
    # 클래스의 갯수
    cnt_class = len(class_list)
    print("Class Summary: Total={}".format(cnt_class))
    
    # 클래스에 따라 리스트를 만들어 준다.
    # 라벨이 숫자가 아닐수도 있기에 dictionary로 받습니다.
    one_hot_label = {}
    for i, key in enumerate(class_list):
        print("\tClass{}번: {}".format(i+1, i))
        one_hot_label[key] = [1 if class_list[key]==j else 0 for j in range(cnt_class)]
    print("[One-Hot Mapping] ",one_hot_label)
    
    for i in range(len(data_y)):
        data_encode_label.append(one_hot_label[y[i]])
    return cnt_class

cnt_class = one_hot_encode(data_y)

# NN_data.csv에 적용시키기
# NN_data_one_encode라는 새로운 csv파일로 write 
with open(os.path.join(main_dir,'NN_data_one_encode.csv'), 'w') as f:
    write = csv.writer(f)
    
    # data의 열 이름은 x0 x1 x2 y이므로 마지막 빼고 추출 후 리스트 변환
    feature_name = data.columns[:-1].to_list()
    
    # 데이터 특성과 라벨 인코드 값 만드는 부분
    # 클래스 갯수에 따라 열이름도 y0, y1, y2로 만든다.
    for i in range(cnt_class):
        feature_name.append("y{}".format(i))
    
    # 데이터에 노이즈를 더한 값과 y값을 
    # 각각 one-hot encoding하여 변환한 값을 노이즈 데이터에 연결
    csv_save_data = np.c_[data_x_noise, np.array(data_encode_label)]
    
    # 데이터 특성 이름 저장
    write.writerow(feature_name)
    write.writerows(csv_save_data)

# 실습 #3 입력 데이터
# 각각 numpy array로 변환
x_noise = data_x_noise
y_encode = np.array(data_encode_label)

# 모델 생성
model = customModel(x_noise, y_encode)
model.set_layer() # 계층 설계
model.fit() # 데이터 적용 및 실행
print("클래스 예측 값 :\n")

# y값 예측
y_pred = model.softmax_predict()
correct = 0
# 정확도 측정
for i in range(len(y_pred)):
    print("실제값:{}, 예측값:{}".format(data_y[i], y_pred[i]))
    if data_y[i] == y_pred[i]:
        correct += 1
print("정확도: ",correct / len(y_pred))

with open(os.path.join(main_dir,'NN_data_pred.csv'), 'w') as f:
    write = csv.writer(f)
    
    # data의 열 이름은 x0 x1 x2 y이므로 마지막 빼고 추출 후 리스트 변환
    feature_name = data.columns.to_list()
    
    # 데이터 특성과 라벨 인코드 값 만드는 부분
    # 클래스 갯수에 따라 열이름도 y0, y1, y2로 만든다.
    feature_name.append("y_pred")
    
    # 데이터에 노이즈를 더한 값과 y값을 
    # 각각 one-hot encoding하여 변환한 값을 노이즈 데이터에 연결
    csv_save_data = np.c_[data_x_noise, data_y, np.array(y_pred)]
    
    # 데이터 특성 이름 저장
    write.writerow(feature_name)
    write.writerows(csv_save_data)
