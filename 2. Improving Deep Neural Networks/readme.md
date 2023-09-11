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

# 4. Optimization_methods.ipynb
- we can see different about Gradient Descent, Momentum, Adam
- without learning rate decay, Adam is the best
- However, with learning rate decay, they are all similar.
~~~
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(1, L + 1):
        v["dW" + str(l)] = (beta1 * v['dW'+str(l)]) + ((1-beta1)*grads['dW'+str(l)])
        v["db" + str(l)] = (beta1 * v['db'+str(l)]) + ((1-beta1)*grads['db'+str(l)])
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1-(beta1**t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1-(beta1**t))

        s["dW" + str(l)] = (beta2 * s['dW'+str(l)]) + ((1-beta2)*(grads['dW'+str(l)]**2))
        s["db" + str(l)] = (beta2 * s['db'+str(l)]) + ((1-beta2)*(grads['db'+str(l)]**2))
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1-(beta2**t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1-(beta2**t))

        #update parameters
        parameters["W" + str(l)] -= ((learning_rate * (v_corrected["dW" + str(l)]) / (np.sqrt(s_corrected["dW" + str(l)])+epsilon)))
        parameters["b" + str(l)] -= ((learning_rate * (v_corrected["db" + str(l)]) / (np.sqrt(s_corrected["db" + str(l)])+epsilon)))
        
        # YOUR CODE ENDS HERE

    return parameters, v, s, v_corrected, s_corrected
~~~
~~~
def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):

    learning_rate = (1 / (1+decay_rate*int(epoch_num / time_interval))) * learning_rate0 
    return learning_rate
~~~
