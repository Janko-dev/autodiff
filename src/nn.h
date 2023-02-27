#ifndef _NN_H
#define _NN_H

#include <stdlib.h>
#include "autodiff.h"

typedef struct {
    Value** data;
    size_t rows;
} Vector;

typedef struct {
    Value** data;
    size_t rows;
    size_t cols;
} Matrix;

typedef struct {
    Matrix weights;
    Vector biases;
} Layer;

// Multi-Layer Perceptron struct
typedef struct {
    Layer* layers;
    size_t num_layers;
    size_t max_layers;
    float learning_rate;
} MLP;

// MLP API
void init_nn(MLP* nn, float learning_rate);
void destroy_nn(MLP* nn);
void add_layer(MLP* nn, size_t num_inputs, size_t num_neurons);
float fit(MLP* nn, float* X, size_t X_size, float* Y, size_t Y_size);
void print_nn(MLP* nn);

// Linear algebra API
Matrix create_matrix(size_t rows, size_t cols);
void destroy_matrix(Matrix mat);
Vector create_vector(size_t rows);
void destroy_vector(Vector vec);
Vector mat_vec_prod(Matrix mat, Vector vec);

void print_mat(Matrix mat);
void print_vec(Vector vec);

#endif //_NN_H
