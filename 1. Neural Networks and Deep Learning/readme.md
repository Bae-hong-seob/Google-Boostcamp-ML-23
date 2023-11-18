# Python_Basics_with_Numpy

## 1. we learn about sigmoid function using numpy
<img width="1011" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/assets/49437396/4c4cfef6-d721-4b03-81ea-8fee94d7d235">

~~~
sigmoid = 1 / (1+np.exp(-x))
gradient = sigmoid * (1-sigmoid)
~~~

## 2. normalize (row : axis=1, column : axis=0)
- if you use /= operation, which is not supported broadcast.
~~~
x_norm = np.linalg.norm(x, axis=1, keepdims=True) # axis=1이므로 행 별로 noramlize 값 구하기.
x = np.divide(x,x_norm) # boradcasting으로 각 row별로 x_norm 곱하기.
~~~

## 3. softmax
<img width="969" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/assets/49437396/8822ccc7-411d-4be2-9095-3927ef64fb29">

~~~
softmax = np.divide(np.exp(x), np.sum(np.exp(x), axis=1, keepdims=True))
~~~

## 4. important of vectorization
- look vectorization.ipynb
- revolution of deep learning dealing time complexity

## 5. L1, L2 loss functions
- L1 : manhattan distance
- L2 : euclidean distance
~~~
L1 = sum(np.abs(y_hat - y))
L2 = sum((y_hat - y)**2)
~~~

# Logistic_Regression_with_a_Neural_Network_mindset

## 1. understanding about vector dimension
- train x,y shape : (m : number of train sets, height, width, channels) , (1,m)
- test x shape (n : number of test sets, height, width, channels) , (1,n)

## 2. image flatten using numpy reshape
- train_flatten = train.reshape(heights x width x chaneels, m)
- test_flatten = test.reshape(height x width x channels, n)
**the point is using numpy reshape for purpose**

## 3. build learning process
~~~
# initalize parameter with 0
w = np.zeros((dim,1))
b = 0.0
~~~

<img width="1003" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/assets/49437396/52ac2f89-d124-4ced-af3d-8c4a1f51917c">

**loss function : Logistic Regression**
~~~
# forward propagation
Y_hat = sigmoid(np.dot(w.T, x) + b) 
cost = -1/m * np.sum(Y*np.log(Y_hat) + (1-Y)*np.log((1-Y_hat)))

# backward propagation
dw = 1/m * np.dot(X, (Y_hat-Y).T)
db = 1/m * np.sum(Y_hat-Y)
~~~

## 4. optimize loss (=train)
~~~
for _ in range(iterations):
    gradient, loss = propagate(w,b,X,Y)
    dw = grads['dw']
    db = grads['db']

    w = w - learning_rate * dw # update
    b = b - learning_rate * db # update
~~~

## 5. what is the best learning rate?
<img width="466" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/assets/49437396/64682bbd-ccbe-442f-9413-96fc7fe1f746">

- too large : the cost may up and down
- too small : cause overfitting

Then, what is the best learning rate? we talk about that later..

# Planar_data_classification_with_one_hidden_layer

Planar_dataset:

<img width="100" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/3-fig1.png">

## 1. parameter initalize
- random initialize is better than zeros. (proven probabilistically in many case)
~~~
W1 = np.random.randn(n_h,n_x)*0.01
b1 = np.zeros((n_h,1))
~~~

## 2. forward propagation
<p align="center"><img width="300" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/3-fig2.png"></p>
- in this case, we use tanh and sigmoid
~~~
Z1 = np.dot(W1, X) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
~~~

## 3. comput the cost (=total loss)
- in this case, we use cross entropy loss
<img width="100" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/3-fig2.png">
~~~
'''
Arguments:
A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
Y -- "true" labels vector of shape (1, number of examples)
'''

logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
cost = -1/m * np.sum(logprobs)
~~~
