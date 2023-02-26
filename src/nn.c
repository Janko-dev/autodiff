#include "nn.h"

// Returns a floating point number between -1 and 1
float nn_rand(){
    return ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

Vector create_vector(size_t rows, bool has_grad) {
    Value** data = calloc(rows, sizeof(Value*));
    for (size_t i = 0; i < rows; ++i){
        data[i] = ad_create((float)i+1, has_grad);
    }
    return (Vector){
        .rows = rows,
        .data = data
    };
}

void destroy_vector(Vector vec){
    for (size_t i = 0; i < vec.rows; ++i){
        printf("Current: %f\n", vec.data[i]->data);
        ad_destroy(vec.data[i]);
    }
    free(vec.data);
}

Matrix create_matrix(size_t rows, size_t cols, bool has_grad) {
    Value** data = calloc(rows*cols, sizeof(Value*));
    for (size_t i = 0; i < rows*cols; ++i){
        data[i] = ad_create((float)i+1, has_grad);
    }
    return (Matrix){
        .rows = rows,
        .cols = cols,
        .data = data
    };
}

void destroy_matrix(Matrix mat){
    for (size_t i = 0; i < mat.cols*mat.rows; ++i)
        ad_destroy(mat.data[i]);
    free(mat.data);
}

Vector mat_vec_prod(Matrix mat, Vector vec){

    if (mat.cols != vec.rows) {
        fprintf(stderr, "Columns of matrix do not match rows of vector\n");
        exit(1);
    }

    Value** output = calloc(mat.rows, sizeof(Value*));
    for (size_t i = 0; i < mat.rows; ++i){
        Value* res = ad_create(0.0f, true);
        for (size_t j = 0; j < mat.cols; ++j){
            Value* mv = ad_mul(mat.data[j*mat.rows+i], vec.data[j]);
            res = ad_add(res, mv);
        }
        output[i] = res;
    }
    return (Vector){
        .data = output,
        .rows = mat.rows
    };
}

void print_mat(Matrix mat){
    printf("shape (%d, %d)\n", mat.rows, mat.cols);
    for (size_t i = 0; i < mat.rows; ++i){
        for (size_t j = 0; j < mat.cols; ++j){
            printf("[%f] ", mat.data[j*mat.rows + i]->data);
        }
        printf("\n");
    }
}

void print_vec(Vector vec){
    printf("shape (%d, 1)\n", vec.rows);
    for (size_t i = 0; i < vec.rows; ++i) 
        printf("[%f]\n", vec.data[i]->data);
}

MLP* create_nn(){
    MLP* result = calloc(1, sizeof(MLP));
    result->learning_rate = 0.01f;
    return result;
}

// void destroy_nn(MLP* nn){
//     for (size_t i = 0; i < nn->num_layers; ++i){
//         for (size_t j = 0; j < nn->layers[i].num_neurons; ++j){
//             ad_destroy(nn->layers[i].neurons[j].b);
//             for (size_t k = 0; k < nn->layers[i].neurons[j].num_inputs; ++k){
//                 ad_destroy(nn->layers[i].neurons[j].w[k]);
//             }
//             free(nn->layers[i].neurons[i].w);
//         }
//         free(nn->layers[i].neurons);
//         free(nn->layers + i);
//     }
//     free(nn);
// }

void init_layer(Layer* layer, size_t num_inputs, size_t num_neurons){
    layer->num_neurons = num_neurons;
    layer->neurons = calloc(num_neurons, sizeof(Neuron));
    for (size_t i = 0; i < num_neurons; ++i){
        layer->neurons[i].num_inputs = num_inputs;
        layer->neurons[i].b = ad_create(nn_rand(), true);
        layer->neurons[i].w = calloc(num_inputs, sizeof(Value*));
        for (size_t j = 0; j < num_inputs; ++j){
            layer->neurons[i].w[j] = ad_create(nn_rand(), true);
        }
    }
}

void add_layer(MLP* nn, size_t num_inputs, size_t num_neurons){
    if (nn->num_layers >= nn->max_layers){
        nn->max_layers = Extend(nn->max_layers);
        nn->layers = realloc(nn->layers, sizeof(Layer) * nn->max_layers);
    }
    init_layer(nn->layers + nn->num_layers, num_inputs, num_neurons);
    nn->num_layers++;
    // num_neurons = rows of matrix
    // num_inputs = columns of matrix
    // nn->layers[nn->num_layers] = *create_layer(num_inputs, num_neurons);
}

// Value* mat_mul(Value** mat, size_t rows, size_t cols, float* vec){
//     Value* result = calloc(rows, sizeof(Value));
//     for (size_t i = 0; i < rows; ++i){
//         for (size_t j = 0; j < cols; ++j){
//             // result[i] += mat[i][j] * vec[j];
//             Value wx = ad_mul(&mat[i][j], &VAL(vec[j]));
//             result[i] = ad_add(&result[i], &wx);
//         }   
//     }
//     return result;
// }

Value** forward(MLP* nn, Value** xs, size_t xs_size){
    
    Value** output = calloc(10, sizeof(Value*));
    output = xs;
    for (size_t i = 0; i < nn->num_layers; ++i){
        
        size_t out_index = 0;
        for (size_t j = 0; j < nn->layers[i].num_neurons; ++j){
            output[out_index] = ad_dot_product(
                output, 
                nn->layers[i].neurons[j].w, 
                nn->layers[i].neurons[j].num_inputs);
            output[out_index] = ad_add(output[out_index], 
                nn->layers[i].neurons[j].b);
            output[out_index] = ad_tanh(output[out_index]);
            out_index++;
        }
    }
    
    return output;
}

// void fit(NeuralNetwork* nn, float* X, size_t X_size, float* Y, size_t Y_size){

// }