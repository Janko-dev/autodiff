#ifndef _NN_H
#define _NN_H

#include <stdlib.h>
#include "autodiff.h"

#define Extend(n) (n == 0 ? 8 : (n*2))

typedef struct {
    size_t num_inputs;
    Value** w;
    Value* b;
} Neuron;

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
    Neuron* neurons;
    size_t num_neurons;
} Layer;

// Multi-Layer Perceptron
typedef struct {
    Layer* layers;
    size_t num_layers;
    size_t max_layers;
    float learning_rate;
} MLP;

// Neuron* create_neuron(size_t num_inputs);
// void destroy_neuron(Neuron* neuron);

MLP* create_nn();
void destroy_nn(MLP* nn);

void add_layer(MLP* nn, size_t num_inputs, size_t num_neurons);
void fit(MLP* nn, float* X, size_t X_size, float* Y, size_t Y_size);
// void predict(NeuralNetwork* nn, )

Matrix create_matrix(size_t rows, size_t cols, bool has_grad);
void destroy_matrix(Matrix mat);

Vector create_vector(size_t rows, bool has_grad);
void destroy_vector(Vector vec);

Vector mat_vec_prod(Matrix mat, Vector vec);

void print_mat(Matrix mat);
void print_vec(Vector vec);

#endif //_NN_H
