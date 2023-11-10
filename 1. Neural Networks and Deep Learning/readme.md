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
