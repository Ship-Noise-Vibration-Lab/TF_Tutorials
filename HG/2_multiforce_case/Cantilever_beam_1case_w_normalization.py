# Cantilever beam test

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import(concatenate)
import pandas as pd
import numpy as np

# 입력 데이터
x1 = pd.read_csv('100sin_5Hz_input.csv') # 힘
b1 = pd.read_csv('sin9Hz_bending.csv') # 힘

x1_data = x1.iloc[:,0] # 시간 input
x2_data = x1.iloc[:,1] # 힘 input
b1_data = b1.iloc[:,1] # 1st bending frequency input

#plt.plot(x2_data)
#plt.show()

X1 = np.array([x1_data, x2_data, b1_data], dtype=np.float32)
X1 = np.transpose(X1)

# 출력 데이터
y1 = pd.read_csv('100sin5Hz_output.csv') # 변위

y2_data = y1.iloc[:,1] # output

Y1 = np.array([y2_data], dtype=np.float32)
Y1 = np.transpose(Y1)

print('Y1 shape is', Y1.shape)

# train, test 데이터 분할
#_train, X_test, Y_train, Y_test = train_test_split(X1,Y1, test_size=0.1, shuffle=False, random_state=512) 
X_train, X_test, Y_train, Y_test = train_test_split(X1,Y1, test_size=0.1, shuffle=False) 



# 데이터 전처리
scaler_x = StandardScaler()
Scaled_X_data = scaler_x.fit_transform(X_train)
scaler_y = StandardScaler()
Scaled_Y_data = scaler_y.fit_transform(Y_train)

# 모델 생성 ( 1-input 2-neurons 1-output)
input_data_num = 3 # weight function 개수와 같은
output_data_num = 1
neurons_num1 = 10
neurons_num2 = 5

X = tf.keras.layers.Input(input_data_num, name = 'input(dt,force,mode_Fre)')
layer1 = tf.keras.layers.Dense(neurons_num1 ,name = 'layer1') (X) # hidden layer 1
Y = tf.keras.layers.Dense(output_data_num, name = 'predictions')(layer1)

model = tf.keras.models.Model(inputs = [X], outputs =[Y], name = 'model')
print(model.summary())


# 내장 루프 모델 학습 
model.compile(optimizer = Adam(learning_rate=0.0005), loss='mse', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=600)
plt.plot(history.history['loss'])
plt.show()

# 모델 예측
y_hat_model = model.predict(X1)
#label = labels[1 if y_hat_model[0][0] > 0.5 else 0]
print("predictions will be",y_hat_model)
yy = scaler_y.inverse_transform(y_hat_model)
plt.plot(yy)
plt.show()
