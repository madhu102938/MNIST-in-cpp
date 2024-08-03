# MNIST-in-cpp
Handwritten digit recognition with c++  

`Matrix.cpp` contains Matrix operations and other utiliy functions like
  - generating normal distribution for initialization of weights
  - matrix multiplication, addition, scalar multiplication, dot product
  - softmax, accuracy_score
  - mini batch sampling

`Neural_network.cpp` contains `main` function which has
  - Foward pass and backward pass
  - determination test data accuracy 

<hr>

Achieved a test accuracy of **91.54%**

This is the loss and accuracy I achieved over 10000 epochs  

![loss](./logs/loss.png)

![accuracy](./logs/accuracy.png)
