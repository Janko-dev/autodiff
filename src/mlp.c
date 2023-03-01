#include "mlp.h"

// Returns a floating point number between -1 and 1
float mlp_rand(){
    return ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

Vector mlp_create_vector(Tape* tp, size_t rows) {
    
    size_t ptr = ad_create(tp, mlp_rand());
    for (size_t i = 1; i < rows; ++i){
        ad_create(tp, mlp_rand());
    }

    return (Vector){
        .rows = rows,
        .ptr = ptr
    };
}

Matrix mlp_create_matrix(Tape* tp, size_t rows, size_t cols) {
    
    size_t ptr = ad_create(tp, mlp_rand());
    for (size_t i = 1; i < rows*cols; ++i){
        ad_create(tp, mlp_rand());
    }
    return (Matrix){
        .rows = rows,
        .cols = cols,
        .ptr = ptr
    };
}

Vector mlp_forward_pass_layer(
        Tape* tp, 
        Matrix mat, 
        Vector vec, 
        Vector bias, 
        size_t (*a_fun)(Tape*, size_t))
    {

    if (mat.cols != vec.rows || mat.rows != bias.rows) {
        fprintf(stderr, "Columns of matrix do not match rows of vector\n");
        exit(1);
    }

    Vector out = mlp_create_vector(tp, mat.rows);
    for (size_t i = 0; i < mat.rows; ++i){
        size_t res = ad_create(tp, 0.0f);
        for (size_t j = 0; j < mat.cols; ++j){
            res = ad_add(tp, 
                bias.ptr + j,
                ad_add(tp, 
                    res,
                    ad_mul(tp, 
                        mat.ptr + j*mat.rows + i,
                        vec.ptr + j)
                    )
            );
            res = a_fun(tp, res);
        }
        // printf("%f\n", GET(res).data);
        GET(out.ptr + i).data = GET(res).data;
        GET(out.ptr + i).left_child = GET(res).left_child;
        GET(out.ptr + i).right_child = GET(res).right_child;
        GET(out.ptr + i).op = GET(res).op;
    }

    return out;
}

void mlp_print_mat(Tape* tp, Matrix mat){
    printf("shape (%d, %d)\n", mat.rows, mat.cols);
    for (size_t i = 0; i < mat.rows; ++i){
        for (size_t j = 0; j < mat.cols; ++j){
            printf("[%f] ", GET(mat.ptr + j*mat.rows + i).data);
        }
        printf("\n");
    }
}

void mlp_print_vec(Tape* tp, Vector vec){
    printf("shape (%d, 1)\n", vec.rows);
    for (size_t i = 0; i < vec.rows; ++i) 
        printf("[%f]\n", GET(vec.ptr+i).data);
}

void mlp_init(MLP* nn, float learning_rate){
    srand(time(NULL));
    nn->learning_rate = learning_rate;
    ad_init_tape(&nn->params);
    nn->num_layers = 0;
    nn->max_layers = 0;
    nn->layers = NULL;
}

void mlp_destroy(MLP* nn){
    ad_destroy_tape(&nn->params);
    free(nn->layers);
}

void mlp_init_layer(Layer* layer, Tape* tp, size_t num_inputs, size_t num_neurons, const char* activation_function){
    
    if (strcmp("relu", activation_function) == 0) {
        layer->activation = ad_relu;
    } else if (strcmp("tanh", activation_function) == 0){
        layer->activation = ad_relu;
    } else if (strcmp("sigm", activation_function) == 0){
        layer->activation = ad_sigm;
    } else {
        fprintf(stderr, "The provided activation function is supported.\nChoose either 'relu' or 'tanh'\n");
        exit(1);
    }

    layer->weights = mlp_create_matrix(tp, num_neurons, num_inputs);
    layer->biases  = mlp_create_vector(tp, num_neurons);
}

void mlp_add_layer(MLP* nn, size_t num_inputs, size_t num_neurons, const char* activation_function){
    if (nn->num_layers >= nn->max_layers){
        nn->max_layers = Extend(nn->max_layers);
        nn->layers = realloc(nn->layers, sizeof(Layer) * nn->max_layers);
        if (!nn->layers) {
            fprintf(stderr, "Not enough memory, buy more ram!\n");
            exit(1);
        }
    }
    mlp_init_layer(nn->layers + nn->num_layers, &nn->params, num_inputs, num_neurons, activation_function);
    nn->num_layers++;
}

Vector mlp_forward_pass(MLP* nn, Tape* tp, Vector xs){
    
    Vector out = xs;
    for (size_t i = 0; i < nn->num_layers; ++i){
        out = mlp_forward_pass_layer(tp, 
            nn->layers[i].weights, 
            out,
            nn->layers[i].biases,
            nn->layers[i].activation
        );
    }

    return out;
}

float mlp_fit(MLP* nn, float* X, size_t X_size, float* Y, size_t Y_size){
    
    Tape tp = {0};
    ad_init_tape(&tp);

    // Copy over model params into new tape 
    for (size_t i = 1; i < nn->params.count; ++i){
        ad_create(&tp, nn->params.val_buf[i].data);
    }
    
    // Create and fill input vector
    Vector xs = mlp_create_vector(&tp, X_size);
    for (size_t i = 0; i < X_size; ++i){
        tp.val_buf[xs.ptr + i].data = X[i];
    }

    // Create and fill ground truth vector
    Vector ys = mlp_create_vector(&tp, Y_size);
    for (size_t i = 0; i < Y_size; ++i){
        tp.val_buf[ys.ptr + i].data = Y[i];
    }

    // Forward pass
    Vector out = mlp_forward_pass(nn, &tp, xs);

    // Compute mean squared error
    size_t loss = ad_create(&tp, 0.0f);
    for (size_t i = 0; i < out.rows; ++i){
        loss = ad_add(&tp,
            loss, 
            ad_pow(&tp,
                ad_sub(&tp, out.ptr + i, ys.ptr + i), 
                ad_create(&tp, 2.0f)
            )
        );
    }
    loss = ad_mul(&tp, 
        loss, 
        ad_create(&tp, 1.0f/(float)out.rows)
    );
    
    // Backpropagation with autodiff
    ad_reverse(&tp, loss);

    // Update rule
    for (size_t i = 1; i < nn->params.count; ++i){
        nn->params.val_buf[i].data -= nn->learning_rate * tp.val_buf[i].grad;
    }

    // Safe loss value
    float ret_loss = tp.val_buf[loss].data;
    
    // Clean tape
    ad_destroy_tape(&tp);

    return ret_loss;
}

void mlp_predict(MLP* nn, float* xs, size_t xs_size, float* out, size_t out_size){

    Tape tp = {0};
    ad_init_tape(&tp);

    // Copy over model params into new tape 
    for (size_t i = 1; i < nn->params.count; ++i){
        ad_create(&tp, nn->params.val_buf[i].data);
    }
    
    // Create and fill input vector
    Vector xs_vec = mlp_create_vector(&tp, xs_size);
    for (size_t i = 0; i < xs_size; ++i){
        tp.val_buf[xs_vec.ptr + i].data = xs[i];
    }

    // Forward pass
    Vector out_vec = mlp_forward_pass(nn, &tp, xs_vec);

    for (size_t i = 0; i < out_size; ++i){
        out[i] = tp.val_buf[out_vec.ptr + i].data;
    }

    // Clean tape
    ad_destroy_tape(&tp);
}

void mlp_print(MLP* nn){
    printf("------------- MLP model -------------\nlearning_rate = %g\n", nn->learning_rate);
    printf("Input layer,   (in: %3d):             ", nn->layers[0].weights.cols);
    for (size_t j = 0; j < nn->layers[0].weights.cols; ++j){
            printf("[n]  ");
    }
    printf("\n");
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
