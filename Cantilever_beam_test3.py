# Cantilever beam test
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import(concatenate)
import pandas as pd
import numpy as np

# 입력 데이터
x1 = pd.read_csv('100sin_5Hz_input.csv') # 힘
x2 = pd.read_csv('100sin_20Hz_input.csv') # 힘
x3 = pd.read_csv('100sin_40Hz_input.csv') # 힘

x1_data = x1.iloc[:,0] # 시간 input
x2_data = x1.iloc[:,1] # 힘 input
x3_data = x2.iloc[:,1] # 힘 input
x4_data = x3.iloc[:,1] # 힘 input

X1 = np.array([x1_data, x2_data], dtype=np.float32)
X2 = np.array([x1_data, x3_data], dtype=np.float32)
X3 = np.array([x1_data, x4_data], dtype=np.float32)

X1 = np.transpose(X1)
X2 = np.transpose(X2)
X3 = np.transpose(X3)
XX1 = np.array([X1, X2, X3], dtype=np.float32)
#XX1 = np.array([[x1_data, x2_data], [x1_data, x3_data], [x1_data, x4_data]], dtype=np.float32)
print('XN shape is', X1.shape, X2.shape, X3.shape)

#x1_test = np.array([X1], dtype=np.float32)
x1_list = []
x1_list.append(np.array(X1))
x_test = np.asarray(x1_list)
#print('x1_test shape is', x1_test.shape)
print('x_test shape is', x_test.shape)


# 출력 데이터
y1 = pd.read_csv('100sin5Hz_output.csv') # 변위
y2 = pd.read_csv('100sin20Hz_output.csv') # 변위
y3 = pd.read_csv('100sin40Hz_output.csv') # 변위
y2_data = y1.iloc[:,1] # output
y3_data = y2.iloc[:,1] # output
y4_data = y3.iloc[:,1] # output


Y1 = np.array([y2_data], dtype=np.float32)
Y2 = np.array([y3_data], dtype=np.float32)
Y3 = np.array([y4_data], dtype=np.float32)
Y1 = np.transpose(Y1)
Y2 = np.transpose(Y2)
Y3 = np.transpose(Y3)
print('Y1 shape is', Y1.shape)

YY1 = np.array([Y1, Y2, Y3], dtype=np.float32)
#YY1 = np.array([[y2_data], [y3_data], [y4_data]], dtype=np.float32)
#y1_data = np.array(y1_data)

print(XX1.shape,YY1.shape)
print(XX1.ndim,YY1.ndim)



# 모델 생성 ( 1-input 2-neurons 1-output)
input_data_num = 2 # weight function 개수와 같은
hidden1_num = 2
output_data_num = 1
neurons_num1 = 5
neurons_num2 = 5

X = tf.keras.layers.Input(shape=(XX1.shape[1], XX1.shape[2]),name = 'input(dt,disp)')

#concat1 = concatenate([X1, X2, X3], name = 'concat1') # 여러개 input 합치기
layer1 = tf.keras.layers.Dense(neurons_num1, activation = 'relu',name = 'layer1') (X) # hidden layer 1
#layer2 = tf.keras.layers.Dense(neurons_num2, activation = 'relu',name = 'layer2') (layer1) # hidden layer 1
#layer2 = tf.keras.layers.Dense(1, name = 'layer2')(X1)

Y = tf.keras.layers.Dense(output_data_num, activation = 'sigmoid',name = 'predictions')(layer1)
#Y2 = tf.keras.layers.Dense(output_data_num, activation = 'sigmoid',name = 'output2')(layer2)

#Y = tf.keras.layers.Activation(activation = 'sigmoid')(Y)
#Y = tf.keras.layers.Activation(activation = 'softmax')(Y)

model = tf.keras.models.Model(inputs = [X], outputs = [Y],name = 'model')
print(model.summary())
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
'''

# 사용자 루프 모델 학습 
def binary_crossentropy(YY1, Y): # 실제값(y1_data), 예측값(Y)의 함수
    cross_entropy = -YY1 * tf.math.log(Y) - (1 - YY1) * tf.math.log(1 - Y) # Cost function
    loss = tf.math.reduce_mean(cross_entropy) # cost(=loss)
    return loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
epochs = 100
for epoch_index in range(epochs):
    with tf.GradientTape() as tape:
        Y = model(XX1, training=True)
        loss_value = binary_crossentropy(YY1, Y)
    gradients = tape.gradient(loss_value, model.trainable_variables) 
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epochs % 500 == 0:
       print('epoch: {}/{}: loss: {:.4f}'.format(epoch_index + 1, epochs, loss_value.numpy()))
       

'''

# 내장 루프 모델 학습 
'''
def binary_crossentropy(YY1, Y): # 실제값(y1_data), 예측값(Y)의 함수
    cross_entropy = -YY1 * tf.math.log(Y) - (1 - YY1) * tf.math.log(1 - Y) # Cost function
    loss = tf.math.reduce_mean(cross_entropy) # cost(=loss)
    return loss
'''
#categorical_crossentropy
model.compile(optimizer = Adam(learning_rate=0.005), loss='mse', metrics=['accuracy'])
#history = model.fit(inputs = [XX1], outputs = [YY1], epochs=100)
history = model.fit(XX1, YY1, epochs=3000)



'''
#history = model.fit(x1_data, y1_data, epochs=2000, batch_size=10, verbose=200)
_, accuracy = model.evaluate(x1_data, y1_data)

print('Accuracy: %.2f' % (accuracy*100))

# history plot
def Drawing_Scalar(history_name)
history_DF = pd.DataFrame(history_name.history)
history_DF.plot(figsize = (12,8), linewidth=3)
plt.grid(True)
plt.show()

'''

# 모델 예측
y_hat_model = model.predict(x_test)
#label = labels[1 if y_hat_model[0][0] > 0.5 else 0]
#y_hat_model = model.predict(x_data)
print("predictions will be",y_hat_model)
#print(y_hat_model,tf.math.argmax(y_hat_model,1))

'''
df = pd.DataFrame(y_hat_model)
df.to_csv('./y_expected_test.csv',header=False, index=False)
'''