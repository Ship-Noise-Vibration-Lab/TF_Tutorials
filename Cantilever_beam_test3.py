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
b1 = pd.read_csv('sin9Hz_bending.csv') # 1st bend frequency

x1_data = x1.iloc[:,0] # time
x2_data = x1.iloc[:,1] # force 100N
a1_data = x1.iloc[:,3] # acceleration
b1_data = b1.iloc[:,1] # 1st bending frequency
amp_data = np.ones(b1_data.shape)

plt.plot(a1_data)
plt.show()

X1 = np.array([x1_data, x2_data, a1_data, b1_data, amp_data], dtype=np.float32)

data_num = 10
const1 = 3
const2 = 1
X1 = np.zeros((data_num,len(X1),X1.shape[1]))

for i in range(data_num):
    X1[i,:,:] = np.array([x1_data, const2*x2_data*(const1*i+1), const2*a1_data*(const1*i+1), b1_data, amp_data*(const1*i+1)], dtype=np.float32)
X1 = X1.transpose(0,2,1)

for i in range(data_num):
    name1 = "X_"
    name2 = "Xscaler_"
    name3 = "XScaled_"
    globals()[name1 + str(i+1)] = X1[i,:,:]
    globals()[name2 + str(i+1)] = StandardScaler()
    globals()[name3 + str(i+1)] = locals()[name2 + str(i+1)].fit_transform(locals()[name1 + str(i+1)])
    globals()[name3 + str(i+1)][:,4] = X1[i,:,4]
    X1[i,:,:] = globals()[name3 + str(i+1)]

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
y2_data = y1.iloc[:,1] # output

Y1 = np.array([y2_data], dtype=np.float32)


Y = np.zeros((data_num,len(Y1),Y1.shape[1]))
for i in range(data_num):
    Y[i,:,:] = np.array([const2*y2_data*(const1*i-1)], dtype=np.float32)
Y1 = Y.transpose(0,2,1)

# train, test 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1, test_size=0.1, shuffle=False, random_state=1) 

# 모델 생성 ( 1-input 2-neurons 1-output)
input_data_num = 4 
hidden1_num = 2
output_data_num = 1
neurons_num1 = 8
neurons_num2 = 16

X = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]), name = 'input(case_num,dt,force,mode_fre)')
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
history = model.fit(X1,Y1, epochs=100, batch_size= 1, verbose=1, validation_split=0.2)
#plt.plot(history.history['loss'])
#plt.show()

#%%
# 모델 예측
#x_test_data = x1.iloc[:,1] *900
#X_test = np.array([x1_data, x_test_data, a1_data, b1_data], dtype=np.float32)
#X_test = np.transpose(X_test)
x_test = np.array([X1[3,:,:]])
#print('x_test shape is', x_test.shape)
y_hat_model = model.predict(x_test)
#print("predictions will be",y_hat_model)
#print(y_hat_model.shape)
yy =  y_hat_model[0]
#print(yy.shape)
plt.plot(x1_data,yy)
plt.show()

'''
# Export
df = pd.DataFrame(y_hat_model)
df.to_csv('./y_expected_test.csv',header=False, index=False)
'''

# %%
