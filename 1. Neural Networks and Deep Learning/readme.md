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

<p align="center"><img width="400" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/3-fig1.png"></p>

## 1. parameter initalize
- random initialize is better than zeros. (proven probabilistically in many case)
~~~
W1 = np.random.randn(n_h,n_x)*0.01
b1 = np.zeros((n_h,1))
~~~

## 2. forward propagation
<p align="center"><img width="200" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/3-fig2.png"></p>
- in this case, we use tanh and sigmoid
~~~
Z1 = np.dot(W1, X) + b1
A1 = np.tanh(Z1)
Z2 = np.dot(W2, A1) + b2
A2 = sigmoid(Z2)
~~~

## 3. comput the cost (=total loss)
- in this case, we use cross entropy loss
<p align="center"><img width="300" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/3-fig3.png"></p>

~~~
'''
Arguments:
A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
Y -- "true" labels vector of shape (1, number of examples)
'''

logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),(1-Y))
cost = -1/m * np.sum(logprobs)
~~~

## 4. backward propagation
- hardest part(=most mathematical part) in deep learning.
<p align="center"><img width="500" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/3-fig4.png"></p>

~~~
dZ2 = A2 - Y
dW2 = 1/m * np.dot(dZ2, A1.T)
db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
dW1 = 1/m * np.dot(dZ1, X.T)
db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
~~~

## 5. update parameters
~~~
W1 -= learning_rate * dW1
b1 -= learning_rate * db1
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
~~~
- if derivative is positive, we have to move left(=loss decrease direction) so W1 will be decrease
- if derivative is negative, we have to move right(=loss decreses direction) so W1 will be decreas

# Building_your_Deep_Neural_Network_Step_by_Step
In this case, we generalize deep learning model about one-layer to n-layers
- the model's structure is input -> layer(linear-ReLU) -> layer(linear-sigmoid) -> output

## 1. initialize parameters deep
~~~
for l in range(1, L): # L = number of layers
    parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
    parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
~~~

## 2. forward propagation
linear layer

~~~
def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)

    return Z, cache
~~~

activation function

~~~
def linear_activation_forward(A_prev, W, b, activation):
   
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
            
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        
    cache = (linear_cache, activation_cache)

    return A, cache
~~~

forward propagation

~~~
def L_model_forward(X, parameters):
    caches = [] # for backward propagation

    for l in range(1, L):
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation='relu')
        caches.append(cache)

    # last layer : sigmoid
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches
~~~

## 3. compute cost
- In this case, we use cross-entropy loss function

~~~
def compute_cost(AL, Y):
    m = Y.shape[1] # the number of examples(=training set)

    cost = -1/m * np.sum((Y*np.log(AL)) + (1-Y)*np.log(1-AL))

    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost
~~~

## 4. backward propagation
linear(WX+b) backward

<p align="center"><img width="300" alt="image" src="https://github.com/Bae-hong-seob/Google-Boostcamp-ML-23/blob/main/1.%20Neural%20Networks%20and%20Deep%20Learning/figs/4-fig1.png"></p>

~~~
dW = 1/m * np.dot(dZ,A_prev.T)
db = 1/m * np.sum(dZ, axis=1, keepdims=True)
dA_prev = np.dot(W.T, dZ)
~~~

activation backward
- It's hard to calculate. so we use pre-defined helper function.
- sigmoid_backward() and relu_backward()

~~~
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
~~~

backward propagation

~~~
def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    current_cache = caches[-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation='sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, activation='relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        
    return grads
~~~

## 5. update parameters
~~~
# GRADED FUNCTION: update_parameters

def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads['dW'+str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads['db'+str(l+1)]
        
    return parameters
~~~

# Deep Neural Network - Application
- In 2. Logistic_Regression_with_a_Neural_Network_mindset.ipynb, we build 2 layers model
- now, we build more deeper model and compare performance.

Now, we make sure you familiar process of deep learning model.
- first, initialize parameters
- second, forward propagation
- third, compute loss
- fourth, backward propagation
- fifth, update parameters

As a result, 2 layer model accuracy is 72% and L layer model accuracy is 80% on the same test set
- Thus we can feel the power of deep layer model.
- Next, we can be able to obtain even higher accuracy bu systematically searching for better hyperparameters(e.g. learning rate, layers dims, number of iterations)

Good work! I hope you have a nice day.
읽어주셔서 감사합니다. 좋은 하루 되세요!
