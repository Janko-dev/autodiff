#ifndef _NN_H
#define _NN_H

#include <stdlib.h>
#include "autodiff.h"

typedef struct {
    size_t ptr;  
    size_t rows;
} Vector;

typedef struct {
    size_t ptr;
    size_t rows;
    size_t cols;
} Matrix;

typedef struct {
    Matrix weights;
    Vector biases;
} Layer;

// Multi-Layer Perceptron struct
typedef struct {
    Tape data_tp;
    Layer* layers;
    size_t num_layers;
    size_t max_layers;
    float learning_rate;
} MLP;

// MLP API
void init_nn(MLP* nn, float learning_rate);
void destroy_nn(MLP* nn);
void add_layer(MLP* nn, Tape* tp, size_t num_inputs, size_t num_neurons);
float fit(MLP* nn, Tape* tp, float* X, size_t X_size, float* Y, size_t Y_size);
void print_nn(MLP* nn);

// Linear algebra API
Matrix create_matrix(Tape* tp, size_t rows, size_t cols);
void destroy_matrix(Tape* tp, Matrix mat);
Vector create_vector(Tape* tp, size_t rows);
void destroy_vector(Tape* tp, Vector vec);
Vector mat_vec_prod(Tape* tp, Matrix mat, Vector vec);

void print_mat(Matrix mat);
void print_vec(Vector vec);

#endif //_NN_H
