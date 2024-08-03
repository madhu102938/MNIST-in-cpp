#include "matrices.h"
using namespace std;

/**
 * @brief Asserts a condition and displays an error message if the condition is false.
 * @param condition The condition to be checked.
 * @param message The error message to be displayed.
 */
static void assert(bool condition, string message)
{
	if (!condition)
	{
		cout << message << endl;
		exit(1);
	}
}

/**
 * @brief Generates a random matrix with the specified number of rows and columns.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return The generated random matrix.
 */
vector<vector<double>> generateRandomMatrix(int rows, int cols)
{
    vector<vector<double>> matrix(rows, vector<double>(cols, 0));
    // normal distribution with mean = 0 and standard deviation = 1
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix[i][j] = dis(gen);
        }
    }
	return matrix;
}

/**
 * @brief Adds two matrices element-wise.
 * @param mat1 The first matrix.
 * @param mat2 The second matrix.
 * @return The result of adding the two matrices.
 */
vector<vector<double>> addMatrices(vector<vector<double>> &mat1, vector<vector<double>> &mat2)
{
	assert(mat1.size() == mat2.size() && mat1[0].size() == mat2[0].size(), "Matrices must have the same size");

	vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0));
	for (int i = 0; i < mat1.size(); ++i)
	{
		for (int j = 0; j < mat1[0].size(); ++j)
		{
			result[i][j] = mat1[i][j] + mat2[i][j];
		}
	}
	return result;
}

/**
 * @brief Subtracts two matrices element-wise.
 * @param mat1 The first matrix.
 * @param mat2 The second matrix.
 * @return The result of subtracting the two matrices.
 */
vector<vector<double>> subtractMatrices(vector<vector<double>> &mat1, vector<vector<double>> &mat2)
{
	assert(mat1.size() == mat2.size() && mat1[0].size() == mat2[0].size(), "Matrices must have the same size");

	vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0));
	for (int i = 0; i < mat1.size(); ++i)
	{
		for (int j = 0; j < mat1[0].size(); ++j)
		{
			result[i][j] = mat1[i][j] - mat2[i][j];
		}
	}
	return result;
}

/**
 * @brief Multiplies two matrices element-wise.
 * @param mat1 The first matrix.
 * @param mat2 The second matrix.
 * @return The result of multiplying the two matrices.
 */
vector<vector<double>> element_wise_multiply(vector<vector<double>> &mat1, vector<vector<double>> &mat2)
{
	assert(mat1.size() == mat2.size() && mat1[0].size() == mat2[0].size(), "Matrices must have the same size");

	vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0));
	for (int i = 0; i < mat1.size(); ++i)
	{
		for (int j = 0; j < mat1[0].size(); ++j)
		{
			result[i][j] = mat1[i][j] * mat2[i][j];
		}
	}
	return result;
}

/**
 * @brief Performs matrix multiplication between two matrices.
 * @param mat1 The first matrix.
 * @param mat2 The second matrix.
 * @return The result of matrix multiplication.
 */
vector<vector<double>> matmul(vector<vector<double>> &mat1, vector<vector<double>> &mat2)
{
	std::ostringstream stream;
	stream << mat1.size() << "x" << mat1[0].size() << " * " << mat2.size() << "x" << mat2[0].size();
	stream << " not compatible for matrix multiplication";

	std::string condition = stream.str();
	assert(mat1[0].size() == mat2.size(), condition);

	vector<vector<double>> result(mat1.size(), vector<double>(mat2[0].size(), 0));
	for (int i = 0; i < mat1.size(); ++i)
	{
		for (int j = 0; j < mat2[0].size(); ++j)
		{
			for (int k = 0; k < mat1[0].size(); ++k)
			{
				result[i][j] += mat1[i][k] * mat2[k][j];
			}
		}
	}
	return result;
}

/**
 * @brief Calculates the dot product of two vectors.
 * @param vec1 The first vector.
 * @param vec2 The second vector.
 * @return The dot product of the two vectors.
 */
double dot_product(vector<double> &vec1, vector<double> &vec2)
{
	assert(vec1.size() == vec2.size(), "Vectors must have the same size");

	int result = 0;
	for (int i = 0; i < vec1.size(); ++i)
	{
		result += vec1[i] * vec2[i];
	}
	return result;
}

/**
 * @brief Transposes a matrix.
 * @param mat The matrix to be transposed.
 * @return The transposed matrix.
 */
vector<vector<double>> transpose(vector<vector<double>> &mat)
{
	vector<vector<double>> result(mat[0].size(), vector<double>(mat.size(), 0));
	for (int i = 0; i < mat.size(); ++i)
	{
		for (int j = 0; j < mat[0].size(); ++j)
		{
			result[j][i] = mat[i][j];
		}
	}
	return result;
}

/**
 * @brief Creates a matrix filled with ones.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return The matrix filled with ones.
 */
vector<vector<double>> ones(int rows, int cols)
{
	vector<vector<double>> matrix(rows, vector<double>(cols, 1));
	return matrix;
}

/**
 * @brief Creates a matrix filled with zeros.
 * @param rows The number of rows in the matrix.
 * @param cols The number of columns in the matrix.
 * @return The matrix filled with zeros.
 */
vector<vector<double>> zeros(int rows, int cols)
{
	vector<vector<double>> matrix(rows, vector<double>(cols, 0));
	return matrix;
}

/**
 * @brief Adds a row vector to each row of a matrix.
 * @param mat1 The matrix.
 * @param mat2 The row vector.
 * @return The result of adding the row vector to each row of the matrix.
 */
vector<vector<double>> row_wise_add(vector<vector<double>> &mat1, vector<vector<double>> &mat2)
{
	assert(mat2.size() == 1, "Second matrix must have only one row");

	vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0));
	for (int i = 0; i < mat1.size(); ++i)
	{
		for (int j = 0; j < mat1[0].size(); ++j)
		{
			result[i][j] = mat1[i][j] + mat2[0][j];
		}
	}
	return result;
}

/**
 * @brief Applies the hyperbolic tangent function element-wise to a matrix.
 * @param mat1 The matrix.
 * @return The result of applying the hyperbolic tangent function to the matrix.
 */
vector<vector<double>> matrix_tanh(vector<vector<double>>& mat1)
{
	vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0));
	for (int i = 0; i < mat1.size(); ++i)
	{
		for (int j = 0; j < mat1[0].size(); ++j)
		{
			result[i][j] = tanh(mat1[i][j]);
		}
	}
	return result;
}

/**
 * @brief Applies the softmax function to a matrix along a specified dimension.
 * @param mat1 The matrix.
 * @param dim The dimension along which to apply the softmax function. Default is 1.
 * @return The result of applying the softmax function to the matrix.
 */
vector<vector<double>> matrix_softmax(vector<vector<double>>& mat1, int dim = 1)
{
	vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0));
	if (dim == 1)
	{
		for (int i = 0; i < mat1.size(); ++i)
		{
			double sum = 0;
			for (int j = 0; j < mat1[0].size(); ++j)
			{
				sum += exp(mat1[i][j]);
			}
			for (int j = 0; j < mat1[0].size(); ++j)
			{
				result[i][j] = exp(mat1[i][j]) / sum;
			}
		}
	}
	else if (dim == 0)
	{
		for (int j = 0; j < mat1[0].size(); j++)
		{
			for (int i = 0; i < mat1.size(); i++)
			{
				double sum = 0;
				for (int k = 0; k < mat1.size(); k++)
				{
					sum += exp(mat1[k][j]);
				}
				result[i][j] = exp(mat1[i][j]) / sum;
			}
		}
	}
	return result;
}

/**
 * @brief Multiplies a matrix by a scalar.
 * @param mat1 The matrix.
 * @param scalar The scalar value.
 * @return The result of multiplying the matrix by the scalar.
 */
vector<vector<double>> scalar_multiply(vector<vector<double>>& mat1, double scalar)
{
	vector<vector<double>> result(mat1.size(), vector<double>(mat1[0].size(), 0));
	for (int i = 0; i < mat1.size(); ++i)
	{
		for (int j = 0; j < mat1[0].size(); ++j)
		{
			result[i][j] = mat1[i][j] * scalar;
		}
	}
	return result;
}

/**
 * @brief Flattens a 3D matrix into a 2D matrix.
 * @param images The 3D matrix of images.
 * @return The flattened 2D matrix.
 */
vector<vector<double>> flatten(vector<vector<vector<double>>>& images)
{
	int B, H, W;
	B = images.size();
	H = images[0].size();
	W = images[0][0].size();

	vector<vector<double>> flattened_images;
	for (vector<vector<double>> &image : images)
	{
		vector<double> temp;
		for (vector<double> &row : image)
		{
			for (double &col : row)
			{
				temp.push_back(col);
			}
		}
		flattened_images.push_back(temp);
	}
	return flattened_images;
}

/**
 * @brief Generates a mini-batch of random indexes within a specified range.
 * @param start The start index.
 * @param end The end index.
 * @param mini_batch_size The size of the mini-batch.
 * @return The mini-batch of random indexes.
 */
vector<int> mini_batch(int start, int end, int mini_batch_size)
{
    vector<int> batch_indexes;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(start, end);

    for (int i = 0; i < mini_batch_size; ++i)
    {
        batch_indexes.push_back(dis(gen));
    }

    return batch_indexes;
}

/**
 * @brief Extracts a mini-batch of data from a given data set based on the specified indexes.
 * @param data The data set.
 * @param indices The indexes of the data to be extracted.
 * @return The mini-batch of data.
 */
vector<vector<double>> extract_mini_batch(vector<vector<double>>& data, vector<int>& indices)
{
	vector<vector<double>> mini_batch;
	for (int i : indices)
	{
		mini_batch.push_back(data[i]);
	}
	return mini_batch;
}

/**
 * @brief Calculates the accuracy score between two sets of labels.
 * @param y_true The true labels.
 * @param y_pred The predicted labels.
 * @return The accuracy score.
 */
double accuracy_score(vector<int> & y_true, vector<int>& y_pred)
{
	assert(y_true.size() == y_pred.size(), "y_true and y_pred must have the same size");

	int correct = 0;
	for (int i = 0; i < y_true.size(); i++)
	{
		if (y_true[i] == y_pred[i])
		{
			correct++;
		}
	}
	return (double)correct / y_true.size();
}