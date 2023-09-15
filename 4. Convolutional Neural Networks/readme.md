# 1. Convolution_model_Step_by_Step_v1.ipynb
convolution operation : image * filter element wise.
- if filter.shape(3,3,3), create 27 values.
- sum of 27 values is the output 1 pixel.
- Thus, one filter make 2D image(called feature map)

padding, pooling
- forward , bacward propagation
- calculate gradient : sum all of W(m,i,j,c)
