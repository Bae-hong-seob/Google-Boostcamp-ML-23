# 1. Convolution_model_Step_by_Step_v1.ipynb
convolution operation : image * filter element wise.
- if filter.shape(3,3,3), create 27 values.
- sum of 27 values is the output 1 pixel.
- Thus, one filter make 2D image(called feature map)

padding, pooling
- forward , bacward propagation
- calculate gradient : sum all of W(m,i,j,c)

# 4. Transfer_learning_with_MobileNet_v1.ipynb
pretrained model을 불러올 때 include_top=False
- 가장 마지막 layer를 떼고 불러옴, 그리고 모델 output으로부터 직접 softmax 혹은 sigmoid. task에 맞는 마지막 layer를 설정
- fine-tuning시, 데이터 양에 따라 적절한 layer수만큼 layer.trainable = True로 설정
We can Transfer Learning and Fine-tuning!!
~~~
base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False, # <== Important!!!!
                                               weights='imagenet') # From imageNet
# set training to False to avoid keeping track of statistics in the batch norm layer
x = base_model(input, training=base_model.trainable) 

# add the new Binary classification layers
# use global avg pooling to summarize the info in each channel
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# include dropout with probability of 0.2 to avoid overfitting
x = tf.keras.layers.Dropout(0.2)(x)
    
# use a prediction layer with one neuron (as a binary classifier only needs one)
outputs = tf.keras.layers.Dense(1)(x)

# UNQ_C3
base_model = model2.layers[4]
base_model.trainable = True
# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 120

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = True

base_learning_rate = 0.001
model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1 * base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
~~~
