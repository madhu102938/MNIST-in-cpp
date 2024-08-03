#include "Matrices.h"
using namespace std;

uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0x000000ff) |
        ((val >> 8) & 0x0000ff00) |
        ((val << 8) & 0x00ff0000) |
        ((val << 24) & 0xff000000);
}


vector<vector<vector<double>>> read_mnist_images(const string& file_path)
{
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + file_path);
    }

    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    // Read the magic number
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    if (magic_number != 2051) {
        throw runtime_error("Invalid MNIST image file!");
    }

    // Read the number of images
    file.read(reinterpret_cast<char*>(&number_of_images), sizeof(number_of_images));
    number_of_images = swap_endian(number_of_images);

    // Read the number of rows
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    rows = swap_endian(rows);

    // Read the number of columns
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    cols = swap_endian(cols);

    // Prepare a vector to store images
    vector<vector<vector<double>>> images(number_of_images, vector<vector<double>>(rows, vector<double>(cols)));

    // Read image data
    for (int i = 0; i < number_of_images; ++i) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                images[i][r][c] = static_cast<double>(pixel);
            }
        }
    }

    return images;
}


vector<int> read_mnist_labels(const string& file_path) {
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + file_path);
    }

    uint32_t magic_number = 0;
    uint32_t number_of_labels = 0;

    // Read the magic number
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = swap_endian(magic_number);
    if (magic_number != 2049) {
        throw runtime_error("Invalid MNIST label file!");
    }

    // Read the number of labels
    file.read(reinterpret_cast<char*>(&number_of_labels), sizeof(number_of_labels));
    number_of_labels = swap_endian(number_of_labels);

    // Prepare a vector to store labels
    vector<int> labels(number_of_labels);

    // Read label data
    for (uint32_t i = 0; i < number_of_labels; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), sizeof(label));
        labels[i] = label;
    }

    return labels;
}