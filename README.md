# MNIST-in-cpp
Handwritten digit recognition with c++  

`Matrix.cpp` contains Matrix operations and other utiliy functions like
  - Generating normal distribution for initialization of weights
  - Matrix multiplication, addition, scalar multiplication, dot product
  - Softmax, accuracy_score
  - Tanh activation
  - Randomly sampling mini batch

`Neural_network.cpp` contains `main` function which has
  - Contains implementation for a simple 3 level neural network
    - Input, hidden and output layer
    - Images are flatten before feeding into neural net
    - Hidden layer has tanh activation
    - CrossEntropy loss is used
  - Foward pass and backward pass
  - Determination of test data accuracy
  - Weights have been initialized according to **Kaiming normal distribution**
    - Weights of the final layer have been factored down while initialization and bias has not been used for the final layer

<hr>

Achieved a test accuracy of **91.54%**

This is the **training loss** and **training accuracy** I achieved over 10000 epochs  

![loss](./logs/loss.png)

![accuracy](./logs/accuracy.png)
