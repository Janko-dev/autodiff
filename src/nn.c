#include "nn.h"

// Returns a floating point number between -1 and 1
float nn_rand(){
    return ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

Vector create_vector(size_t rows) {
    Value** data = calloc(rows, sizeof(Value*));
    for (size_t i = 0; i < rows; ++i){
        data[i] = ad_create(nn_rand());
    }
    return (Vector){
        .rows = rows,
        .data = data
    };
}

void destroy_vector(Vector vec){
    for (size_t i = 0; i < vec.rows; ++i){
        ad_destroy(vec.data[i]);
    }
    free(vec.data);
}

Matrix create_matrix(size_t rows, size_t cols) {
    Value** data = calloc(rows*cols, sizeof(Value*));
    for (size_t i = 0; i < rows*cols; ++i){
        data[i] = ad_create(nn_rand());
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
        Value* res = ad_create(0.0f);
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

void init_nn(MLP* nn, float learning_rate){
    nn->learning_rate = learning_rate;
    nn->num_layers = 0;
    nn->max_layers = 0;
    nn->layers = NULL;
}

void destroy_nn(MLP* nn){
    for (size_t i = 0; i < nn->num_layers; ++i){
        destroy_matrix(nn->layers[i].weights);
        destroy_vector(nn->layers[i].biases);
    }
    free(nn->layers);
}

void init_layer(Layer* layer, size_t num_inputs, size_t num_neurons){
    layer->weights = create_matrix(num_neurons, num_inputs);
    layer->biases  = create_vector(num_neurons);
}

void add_layer(MLP* nn, size_t num_inputs, size_t num_neurons){
    if (nn->num_layers >= nn->max_layers){
        nn->max_layers = Extend(nn->max_layers);
        nn->layers = realloc(nn->layers, sizeof(Layer) * nn->max_layers);
    }
    init_layer(nn->layers + nn->num_layers, num_inputs, num_neurons);
    nn->num_layers++;
}

void forward(MLP* nn, Vector* xs, Vector* out){
    *out = mat_vec_prod(nn->layers[0].weights, *xs);
    for (size_t i = 1; i < nn->num_layers; ++i){
        *out = mat_vec_prod(nn->layers[i].weights, *out);
        for (size_t j = 0; j < out->rows; ++j){
            out->data[j] = ad_tanh(
                ad_add(out->data[j], nn->layers[i].biases.data[j])
            );
        }
    }
}

float fit(MLP* nn, float* X, size_t X_size, float* Y, size_t Y_size){
    Vector xs = create_vector(X_size);
    for (size_t i = 0; i < X_size; ++i){
        xs.data[i]->data = X[i];
    }

    Vector ys = create_vector(Y_size);
    for (size_t i = 0; i < Y_size; ++i){
        ys.data[i]->data = Y[i];
    }

    // Forward pass
    Vector out = {0};
    forward(nn, &xs, &out);
    
    // Compute mean squared error
    Value* loss = ad_create(0.0f);
    for (size_t i = 0; i < out.rows; ++i){
        loss = ad_add(
            loss, 
            ad_pow(
                ad_sub(out.data[i], ys.data[i]), 
                ad_create(2.0f)
            )
        );
    }
    loss = ad_mul(loss, ad_create(1.0f/(float)out.rows));
    
    // backpropagation with autodiff
    ad_reverse(loss);

    // update rule
    for (size_t i = 0; i < nn->num_layers; ++i){
        Layer* layer = nn->layers + i;
        for (size_t j = 0; j < layer->biases.rows; ++j){
            Value* bias = layer->biases.data[j];
            // update bias by walking the negative gradient
            bias->data -= nn->learning_rate * bias->grad;
            // reset gradient
            bias->grad = 0;
        }
        size_t rows = layer->weights.rows;
        size_t cols = layer->weights.cols;        
        for (size_t j = 0; j < rows; ++j){
            for (size_t k = 0; k < cols; ++k){
                Value* weight = layer->weights.data[k*rows + j];
                // update weight by walking the negative gradient
                weight->data -= nn->learning_rate * weight->grad;
                // reset gradient
                weight->grad = 0;
            }
        }
    }

    destroy_vector(xs);
    destroy_vector(ys);
    free(loss);

    return loss->data;
}

void print_nn(MLP* nn){
    printf("------------- MLP model -------------\nlearning_rate = %g\n", nn->learning_rate);
    for (size_t i = 0; i < nn->num_layers; ++i){
        printf("Layer %d, shape (in: %3d, out: %3d):   ", i+1, 
            nn->layers[i].weights.cols, 
            nn->layers[i].weights.rows);
        for (size_t j = 0; j < nn->layers[i].weights.rows; ++j){
            printf("[n]  ");
        }
        printf("\n");
    }
    printf("-------------------------------------\n");
}