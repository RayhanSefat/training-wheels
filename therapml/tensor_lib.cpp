#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
using namespace std;

namespace py = pybind11;

vector<vector<vector<double>>> cpp_tensor_multiply(
    const vector<vector<vector<double>>>& arr1,
    const vector<vector<vector<double>>>& arr2) {

    size_t b = arr1.size();
    size_t x = arr1[0].size();
    size_t y = arr1[0][0].size();
    size_t z = arr2[0][0].size();

    vector<vector<vector<double>>> result(
        b, vector<vector<double>>(x, vector<double>(z, 0.0)));

    for (size_t i = 0; i < b; ++i) {
        for (size_t j = 0; j < x; ++j) {
            for (size_t k = 0; k < z; ++k) {
                double sum = 0;
                for (size_t l = 0; l < y; ++l) {
                    sum += arr1[i][j][l] * arr2[i][l][k];
                }
                result[i][j][k] = sum;
            }
        }
    }
    return result;
}

PYBIND11_MODULE(tensor_lib, m) {
    m.def("tensor_multiply", &cpp_tensor_multiply, "Batched Tensor Multiplication in C++");
}