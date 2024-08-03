#pragma once
#include <vector>
#include <random>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <string>
#include <cstdint>
#include <sstream>
using namespace std;

static void assert(bool condition, string message);
vector<vector<double>> generateRandomMatrix(int row, int cols);
vector<vector<double>> addMatrices(vector<vector<double>> &mat1, vector<vector<double>> &mat2);
vector<vector<double>> subtractMatrices(vector<vector<double>> &mat1, vector<vector<double>> &mat2);
vector<vector<double>> element_wise_multiply(vector<vector<double>>& mat1, vector<vector<double>>& mat2);
vector<vector<double>> matmul(vector<vector<double>> &mat1, vector<vector<double>> &mat2);
double dot_product(vector<double> &vec1, vector<double> &vec2);
vector<vector<double>> transpose(vector<vector<double>>& mat);
vector<vector<double>> ones(int rows, int cols);
vector<vector<double>> zeros(int rows, int cols);
vector<vector<double>> row_wise_add(vector<vector<double>>& mat1, vector<vector<double>>& vec);
vector<vector<double>> matrix_tanh(vector<vector<double>>& mat1);
vector<vector<double>> matrix_softmax(vector<vector<double>>& mat1, int dim);
vector<vector<double>> scalar_multiply(vector<vector<double>>& mat1, double scalar);
vector<vector<vector<double>>> read_mnist_images(const string& file_path);
vector<int> read_mnist_labels(const string& file_path);
uint32_t swap_endian(uint32_t val);
vector<vector<double>> flatten(vector<vector<vector<double>>>& images);
vector<int> mini_batch(int start, int end, int mini_batch_size);
vector<vector<double>> extract_mini_batch(vector<vector<double>>& data, vector<int>& indices);
double accuracy_score(vector<int>& y_true, vector<int>& y_pred);