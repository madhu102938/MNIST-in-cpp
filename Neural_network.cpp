#include "Matrices.h"
using namespace std;

/**
 * @brief Main function that trains a neural network model on the MNIST dataset.
 */
int main()
{
    // Path to the MNIST training images
    string x_train_path = "D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/archive/train-images.idx3-ubyte";
    // Read the MNIST training images
    vector<vector<vector<double>>> x_train = read_mnist_images(x_train_path);

    // Path to the MNIST training labels
    string y_train_path = "D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/archive/train-labels.idx1-ubyte";
    // Read the MNIST training labels
    vector<int> y_train = read_mnist_labels(y_train_path);

    // Path to the MNIST test images
    string x_test_path = "D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/archive/t10k-images.idx3-ubyte";
    // Read the MNIST test images
    vector<vector<vector<double>>> x_test = read_mnist_images(x_test_path);

    // Path to the MNIST test labels
    string y_test_path = "D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/archive/t10k-labels.idx1-ubyte";
    // Read the MNIST test labels
    vector<int> y_test = read_mnist_labels(y_test_path);

    // Flatten the images
    auto flat_x_train = flatten(x_train);
    auto flat_x_test = flatten(x_test);

    // Define the dimensions of the hidden layer and output layer
    int hidden_dim = 200;
    int out_dim = 10;

    // Initialize the weights and biases
    auto W1 = generateRandomMatrix(flat_x_train[0].size(), hidden_dim);
    auto b1 = generateRandomMatrix(1, hidden_dim);
    auto W2 = generateRandomMatrix(hidden_dim, out_dim);

    // Scale the weights
    W2 = scalar_multiply(W2, 0.01);
    W1 = scalar_multiply(W1, (5.0 / 3.0) / (sqrt(flat_x_train[0].size())));

    // Define the number of epochs, batch size, learning rate, and initial accuracy
    int epochs = 10000;
    int batch_size = 32;
    double learning_rate = 0.1;
    double accuracy_temp = 0.0;

    // Open a file to store the total number of epochs
    ofstream epoch_file("D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/logs/tot_epochs.txt");
    epoch_file << epochs << endl;
    epoch_file.close();

    // Vector to store the loss and accuracy for each epoch
    vector<pair<double, double>> loss_accuracy;

    // Training loop
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        // Get a mini-batch of training data
        vector<int> batch = mini_batch(0, flat_x_train.size() - 1, batch_size);
        auto X = extract_mini_batch(flat_x_train, batch);
        vector<int> y(batch_size);

        // Extract the labels for the mini-batch
        for (int i = 0; i < batch_size; i++)
        {
            y[i] = y_train[batch[i]];
        }

        // Forward pass
        auto m1 = matmul(X, W1); // X = batch * 784, W1 = 784 * 200, m1 = batch * 200
        auto Z1 = row_wise_add(m1, b1); // Z1 = batch * 200
        auto A1 = matrix_tanh(Z1); // A1 = batch * 200
        auto logits = matmul(A1, W2); // m2 = batch * 10
        auto out_preds = matrix_softmax(logits, 1); // out_preds = batch * 10

        // Calculate the loss
        int i = 0;
        double tot_loss = 0;
        for (auto &row : out_preds)
        {
            tot_loss += log(row[(int)y[i]]);
            i++;
        }
        double loss = -tot_loss / X.size();

        // Calculate the accuracy
        vector<int> y_preds_temp;
        for (int i = 0; i < out_preds.size(); i++)
        {
            y_preds_temp.push_back(max_element(out_preds[i].begin(), out_preds[i].end()) - out_preds[i].begin());
        }
        accuracy_temp = accuracy_score(y, y_preds_temp);

        // Print the loss and accuracy for every 100 epochs
        if (epoch == 1 || epoch % 100 == 0)
        {
            cout << "Epoch: " << epoch << "/" << epochs << " Loss: " << loss << " Accuracy : " << accuracy_temp << endl;

            // Adjust the learning rate based on the number of epochs
            if (epochs > 5000)
                learning_rate = 0.01;
            else if (epochs > 2500)
                learning_rate = 0.05;
        }

        // Store the loss and accuracy for the current epoch
        loss_accuracy.push_back({loss, accuracy_temp});

        // Backward pass
        vector<vector<double>> dlogits = out_preds;
        for (int i = 0; i < dlogits.size(); i++)
        {
            dlogits[i][y[i]] -= 1;
        }
        dlogits = scalar_multiply(dlogits, 1.0 / X.size());

        // dlogits = batch * 10, A1 = batch * 200, W2 = 200 * 10
        auto t_A1 = transpose(A1);
        vector<vector<double>> dW2 = matmul(t_A1, dlogits);

        auto t_W2 = transpose(W2);
        vector<vector<double>> dA1 = matmul(dlogits, t_W2);

        vector<vector<double>> dZ1 = zeros(Z1.size(), Z1[0].size());
        for (int i = 0; i < dZ1.size(); i++)
        {
            for (int j = 0; j < dZ1[0].size(); j++)
            {
                dZ1[i][j] = dA1[i][j] * (1 - A1[i][j] * A1[i][j]);
            }
        }

        vector<vector<double>> db1 = zeros(b1.size(), b1[0].size());
        for (int j = 0; j < dZ1[0].size(); j++)
        {
            for (int i = 0; i < dZ1.size(); i++)
            {
                db1[0][j] += dZ1[i][j];
            }
        }

        vector<vector<double>> dm1 = dZ1;

        auto t_X = transpose(X);
        vector<vector<double>> dW1 = matmul(t_X, dm1);

        auto s_m_1 = scalar_multiply(dW1, -1.0 * learning_rate);
        W1 = addMatrices(W1, s_m_1);

        auto s_m_2 = scalar_multiply(dW2, -1.0 * learning_rate);
        W2 = addMatrices(W2, s_m_2);

        auto s_m_3 = scalar_multiply(db1, -1.0 * learning_rate);
        b1 = addMatrices(b1, s_m_3);
    }

    // Write the running loss and accuracy to files
    ofstream loss_file("D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/logs/loss.txt", ios::app);
    ofstream accuracy_file("D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/logs/accuracy.txt", ios::app);
    for (pair<double, double> p : loss_accuracy)
    {
        loss_file << p.first << endl;
        accuracy_file << p.second << endl;
    }
    loss_file.close();
    accuracy_file.close();

    // Predict on the test set
    auto m1_t = matmul(flat_x_test, W1);
    auto Z1_t = row_wise_add(m1_t, b1);
    auto A1_t = matrix_tanh(Z1_t);
    auto logits_t = matmul(A1_t, W2);
    auto out_preds_t = matrix_softmax(logits_t, 1);
    vector<int> y_preds;
    for (int i = 0; i < out_preds_t.size(); i++)
    {
        y_preds.push_back(max_element(out_preds_t[i].begin(), out_preds_t[i].end()) - out_preds_t[i].begin());
    }

    // Calculate the accuracy on the test set
    double test_accuracy = accuracy_score(y_test, y_preds);
    cout << "Accuracy on test set: " << test_accuracy << endl;

    // Write the test accuracy to a file
    ofstream test_accuracy_file("D:/Machine Learning/machine_learning_with_cpp/cpp_ml/Project1/logs/test_acc.txt");
    test_accuracy_file << test_accuracy << endl;
    test_accuracy_file.close();

    return 0;
}