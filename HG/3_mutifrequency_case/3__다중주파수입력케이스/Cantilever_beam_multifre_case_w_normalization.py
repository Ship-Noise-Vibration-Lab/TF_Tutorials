# Cantilever beam test
from xmlrpc.client import Fault
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import(concatenate)
import pandas as pd
import numpy as np

# 입력 데이터
x1 = pd.read_csv('100sin_5Hz_input.csv') # 힘
x2 = pd.read_csv('100sin_15Hz_input.csv') # 힘
x3 = pd.read_csv('100sin_20Hz_input.csv') # 힘
x4 = pd.read_csv('100sin_30Hz_input.csv') # 힘
x5 = pd.read_csv('100sin_40Hz_input.csv') # 힘
x6 = pd.read_csv('100sin_60Hz_input.csv') # 힘
x7 = pd.read_csv('100sin_80Hz_input.csv') # 힘
x8 = pd.read_csv('100sin_100Hz_input.csv') # 힘

b1 = pd.read_csv('sin9Hz_bending.csv') # 1st bend frequency
b1_data = b1.iloc[:,1] # 1st bending frequency

amp_data = np.ones(b1_data.shape)

input_data_num = 5
fre_data_num = 8
multi_data_num = 10
data_num = fre_data_num * multi_data_num

# 주파수케이스 데이터 정리
for i in range(8):
    name1 = "x"
    name2 = "t_data"
    name3 = "f_data"
    name4 = "a_data"
    name5 = "X"
    globals()[name2 + str(i+1)] = np.array([globals()[name1 + str(i+1)].iloc[:,0]]) # time
    globals()[name3 + str(i+1)] = np.array([globals()[name1 + str(i+1)].iloc[:,1]]) # force
    globals()[name4 + str(i+1)] = np.array([globals()[name1 + str(i+1)].iloc[:,3]]) # acceleraion
    globals()[name5 + str(i+1)] = np.zeros((input_data_num,len(x1)))
    globals()[name5 + str(i+1)][0,:] = globals()[name2 + str(i+1)]
    globals()[name5 + str(i+1)][1,:] = globals()[name3 + str(i+1)]
    globals()[name5 + str(i+1)][2,:] = globals()[name4 + str(i+1)]
    globals()[name5 + str(i+1)][3,:] = b1_data
    globals()[name5 + str(i+1)][4,:] = amp_data
    globals()[name5 + str(i+1)] = np.transpose(globals()[name5 + str(i+1)])

XX = np.zeros((data_num,len(X1),X1.shape[1]))
  
# 진폭케이스 데이터 추가 전처리
for i in range(fre_data_num):
    name5 = "X"
    XX[i,:,:] = globals()[name5 + str(i+1)]
    XX[1*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[2*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[3*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[4*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[5*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[6*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[7*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[8*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    XX[9*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]

# 진폭케이스 데이터 추가
const1 = np.linspace(1,30,multi_data_num) # amplitude data

for i in range(fre_data_num):
    XX[i,:,1:3] = XX[i,:,1:2] * const1[0]
    XX[1*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[1]
    XX[2*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[2]
    XX[3*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[3]
    XX[4*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[4]
    XX[5*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[5]
    XX[6*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[6]
    XX[7*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[7]
    XX[8*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[8]
    XX[9*fre_data_num + i,:,1:3] = XX[i,:,1:2] * const1[9]
    XX[i,:,4] = XX[i,:,4] * const1[0]
    XX[1*fre_data_num + i,:,4] = XX[i,:,4] * const1[1]
    XX[2*fre_data_num + i,:,4] = XX[i,:,4] * const1[2]
    XX[3*fre_data_num + i,:,4] = XX[i,:,4] * const1[3]
    XX[4*fre_data_num + i,:,4] = XX[i,:,4] * const1[4]
    XX[5*fre_data_num + i,:,4] = XX[i,:,4] * const1[5]
    XX[6*fre_data_num + i,:,4] = XX[i,:,4] * const1[6]
    XX[7*fre_data_num + i,:,4] = XX[i,:,4] * const1[7]
    XX[8*fre_data_num + i,:,4] = XX[i,:,4] * const1[8]
    XX[9*fre_data_num + i,:,4] = XX[i,:,4] * const1[9]
    
# 데이터 정규화 작업
for i in range(data_num):
    name1 = "Temp_X"
    name2 = "Xscaler_"
    name3 = "XScaled_"
    globals()[name1 + str(i+1)] = XX[i,:,:]
    globals()[name2 + str(i+1)] = StandardScaler()
    globals()[name3 + str(i+1)] = locals()[name2 + str(i+1)].fit_transform(locals()[name1 + str(i+1)])
    globals()[name3 + str(i+1)][:,4] = XX[i,:,4]
    XX[i,:,:] = globals()[name3 + str(i+1)]

# 역정규화
for i in range(data_num):
    name1 = "XRescaled_"
    name2 = "Xscaler_"
    globals()[name1 + str(i+1)] = globals()[name2 + str(i+1)].inverse_transform(globals()[name3 + str(i+1)])


'''
# 동일 데이터 확장
data_len = len(X1)
LL = 10
XX = np.zeros((LL*data_len,X1.shape[1]))
for i in range(LL):
    XX[3*i*data_len:(3*i+1)*data_len,:] = X1
    XX[(3*i+1)*data_len:(3*i+2)*data_len,:] = X2
    XX[(3*i+2)*data_len:(3*i+3)*data_len,:] = X3

'''
# 출력 데이터
y1 = pd.read_csv('100sin5Hz_output.csv') # 변위
y2 = pd.read_csv('100sin15Hz_output.csv') # 변위
y3 = pd.read_csv('100sin20Hz_output.csv') # 변위
y4 = pd.read_csv('100sin30Hz_output.csv') # 변위
y5 = pd.read_csv('100sin40Hz_output.csv') # 변위
y6 = pd.read_csv('100sin60Hz_output.csv') # 변위
y7 = pd.read_csv('100sin80Hz_output.csv') # 변위
y8 = pd.read_csv('100sin100Hz_output.csv') # 변위

for i in range(fre_data_num):
    name1 = "y"
    name2 = "u_data"
    globals()[name2 + str(i+1)] = np.array([globals()[name1 + str(i+1)].iloc[:,1]])# output
    globals()[name2 + str(i+1)] = np.transpose(globals()[name2 + str(i+1)])

YY = np.zeros((data_num,len(X1),1))
  
# 진폭케이스 데이터 추가 전처리
for i in range(fre_data_num):
    name5 = "u_data"
    YY[i,:,:] = globals()[name5 + str(i+1)]
    YY[1*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[2*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[3*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[4*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[5*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[6*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[7*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[8*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]
    YY[9*fre_data_num + i,:,:] = globals()[name5 + str(i+1)]

# 진폭케이스 데이터 추가
for i in range(fre_data_num):
    YY[i,:,1:3] = YY[i,:,1:2] * const1[0]
    YY[1*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[1]
    YY[2*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[2]
    YY[3*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[3]
    YY[4*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[4]
    YY[5*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[5]
    YY[6*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[6]
    YY[7*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[7]
    YY[8*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[8]
    YY[9*fre_data_num + i,:,1:3] = YY[i,:,1:2] * const1[9]

# train, test 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1, test_size=0.1, shuffle=False, random_state=1) 

# 모델 생성 ( 1-input 2-neurons 1-output)
hidden1_num = 2
output_data_num = 1
neurons_num1 = 8
neurons_num2 = 16

#X = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]), name = 'input(case_num,dt,force,mode_fre)')
X = tf.keras.layers.Input(shape=(XX.shape[1], XX.shape[2]), name = 'input(case_num,dt,force,mode_fre)')
layer1 = tf.keras.layers.Dense(neurons_num1 ,name = 'layer1') (X) # hidden layer 1
layer2 = tf.keras.layers.Dense(neurons_num2 ,name = 'layer2') (layer1) # hidden layer 1
layer3 = tf.keras.layers.Dense(neurons_num2 ,name = 'layer3') (layer2) # hidden layer 1
layer4 = tf.keras.layers.Dense(neurons_num1 ,name = 'layer4') (layer3) # hidden layer 1
Y = tf.keras.layers.Dense(output_data_num, name = 'predictions')(layer4)

model = tf.keras.models.Model(inputs = [X], outputs =[Y], name = 'model')
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True


# 내장 루프 모델 학습 
model.compile(optimizer = Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])
#history = model.fit(X_train,Y_train, epochs=10, batch_size= 1, verbose=1, validation_split=0.2)
history = model.fit(XX,YY, epochs=100, batch_size= 1, verbose=1, validation_split=0.2)
#plt.plot(history.history['loss'])
#plt.show()

#%%
# 모델 예측
#x_test_data = x1.iloc[:,1] *900
#X_test = np.array([x1_data, x_test_data, a1_data, b1_data], dtype=np.float32)
#X_test = np.transpose(X_test)
x_test = np.array([XX[3,:,:]])
#print('x_test shape is', x_test.shape)
y_hat_model = model.predict(x_test)
#print("predictions will be",y_hat_model)
#print(y_hat_model.shape)
yy =  y_hat_model[0]
#print(yy.shape)
plt.plot(yy)
plt.show()

'''
# Export
df = pd.DataFrame(y_hat_model)
df.to_csv('./y_expected_test.csv',header=False, index=False)
'''

# %%
