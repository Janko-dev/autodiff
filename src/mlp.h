#ifndef _MLP_H
#define _MLP_H

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "autodiff.h"

// Vector and matrix structs that have a size_t ptr
// that points to a value in a tape structure.
//  
typedef struct {
    size_t ptr;  
    size_t rows;
} Vector;

typedef struct {
    size_t ptr;
    size_t rows;
    size_t cols;
} Matrix;

// Layer consists of weights, biases, and an activation function.
// The activation function can currently be one of the following:
// - ReLu (ad_relu())
// - tanh (ad_tanh())
typedef struct {
    Matrix weights;
    Vector biases;
    size_t (*activation)(Tape* tp, size_t a);
} Layer;

// Multi-Layer Perceptron struct
// It manages its own tape of parameters
// that gets copied into a new tape at every start of the fitness function
typedef struct {
    Tape params;
    Layer* layers;
    size_t num_layers;
    size_t max_layers;
    float learning_rate;
} MLP;

// MLP API
void mlp_init(MLP* nn, float learning_rate);
void mlp_destroy(MLP* nn);

void mlp_add_layer(MLP* nn, size_t num_inputs, size_t num_neurons, const char* activation_function);
float mlp_fit(MLP* nn, float* X, size_t X_size, float* Y, size_t Y_size);
void mlp_predict(MLP* nn, float* xs, size_t xs_size, float* out, size_t out_size);

void mlp_print(MLP* nn);

// Linear algebra API
Matrix mlp_create_matrix(Tape* tp, size_t rows, size_t cols);
Vector mlp_create_vector(Tape* tp, size_t rows);

void mlp_print_mat(Tape* tp, Matrix mat);
void mlp_print_vec(Tape* tp, Vector vec);

#endif //_NN_H
