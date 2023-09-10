# 1. Initializawtion.ipynb
- we can see different about weight initialization zeros, random, random * regularization
- 'He' Initialization is the best
~~~
for l in range(1,L):
    W = np.random.randn(layer_dims[l-1], layers_dims[l]) * np.sqrt(2/layers_dims[l-1])
    b = np.zeros((layers_dims[l],1))
~~~

# 2. Regularization.ipynb
- we can see different about Regularization None, L2, Dropout
- Dropout is the best
~~~
# forward_propagation
D1 = np.random.rand(A1.shape[0],A1.shape[1])    
D1 = (D1 < keep_prob).astype(int)
A1 = A1 * D1
A1 /= keep_prob

cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) # we have to cache 

# backward_propagation
dA1 *= D1 
dA1 /= keep_prob
~~~
