# 1. Initializawtion.ipynb
- we can see different about weight initialization zeros, random, random * regularization
- 'He' Initialization is the best
~~~
for l in range(1,L):
    W = np.random.randn(layer_dims[l-1], layers_dims[l]) * np.sqrt(2/layers_dims[l-1])
    b = np.zeros((layers_dims[l],1))
~~~
