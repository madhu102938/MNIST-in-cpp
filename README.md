# MNIST-in-cpp
Handwritten digit recognition with c++  

`Matrix.cpp` contains Matrix operations and other utiliy functions like
  - Generating normal distribution for initialization of weights
  - Matrix multiplication, addition, scalar multiplication, dot product
  - Softmax, accuracy_score
  - Tanh activation
  - Randomly sampling mini batch

`Neural_network.cpp` contains `main` function which has
  - Foward pass and backward pass
  - Determination of test data accuracy
  - Weights have been initialized according to **Kaiming normal distribution**

<hr>

Achieved a test accuracy of **91.54%**

This is the **training loss** and **training accuracy** I achieved over 10000 epochs  

![loss](./logs/loss.png)

![accuracy](./logs/accuracy.png)
