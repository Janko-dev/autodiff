#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "../src/autodiff.h"

// Vector and matrix structs that have a size_t ptr
// that points to a value in a tape structure. 
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
// - Tanh (ad_tanh())
// - Sigmoid (ad_sigm())
typedef struct {
    Matrix weights;
    Vector biases;
    size_t (*activation)(Tape* tp, size_t a);
} Layer;

// Multi-Layer Perceptron struct
// It manages its own tape of parameters
// that gets copied into a new tape at every start of the fitness function (mlp_fit)
typedef struct {
    Tape params;
    Layer* layers;
    size_t num_layers;
    size_t max_layers;
    float learning_rate;
} MLP;

// Returns a floating point number between -1 and 1
float mlp_rand(){
    return ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

// Vector is creating by consecutively creating leaf nodes in the computation graph.
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

// 2D Matrix is created by flattening the matrix into a 1D array and 
// consecutively creating leaf nodes in the computation graph. 
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

void mlp_print_mat(Tape* tp, Matrix mat){
    printf("shape (%d, %d)\n", mat.rows, mat.cols);
    for (size_t i = 0; i < mat.rows; ++i){
        for (size_t j = 0; j < mat.cols; ++j){
            printf("[%f] ", GET(mat.ptr + i*mat.cols + j).data);
        }
        printf("\n");
    }
}

void mlp_print_vec(Tape* tp, Vector vec){
    printf("shape (%d, 1)\n", vec.rows);
    for (size_t i = 0; i < vec.rows; ++i) 
        printf("[%f]\n", GET(vec.ptr+i).data);
}

// Initialise MLP struct by providing learning rate
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

// Add a dense layer to the neural network by providing 
// - the number of input nodes,
// - the number of neurons in the layer, and 
// - the activation function ("relu", "tanh", "sigm")
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

// Forward pass through the components of a layer,
// i.e., the input vector, the weight matrix, the bias vector, and the activation function
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
    size_t* out_ptr = malloc(sizeof(size_t) * mat.rows);

    for (size_t i = 0; i < mat.rows; ++i){
        size_t res = ad_create(tp, 0.0f);
        for (size_t j = 0; j < mat.cols; ++j){
            res = ad_add(tp, 
                res,
                ad_mul(tp, 
                    mat.ptr + i*mat.cols + j,
                    vec.ptr + j)
            );
        }
        res = ad_add(tp, res, bias.ptr + i);
        res = a_fun(tp, res);
        out_ptr[i] = res;
    }

    Vector out = mlp_create_vector(tp, mat.rows);
    for (size_t i = 0; i < mat.rows; ++i){
        GET(out.ptr + i).data = GET(out_ptr[i]).data;
        GET(out.ptr + i).left_child = GET(out_ptr[i]).left_child;
        GET(out.ptr + i).right_child = GET(out_ptr[i]).right_child;
        GET(out.ptr + i).op = GET(out_ptr[i]).op;
    }
    free(out_ptr);

    return out;
}

// Pass through all layers 
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

Vector _predict(MLP* nn, Tape* tp, float* xs, size_t xs_size){
    
    // Copy over model params into new tape 
    for (size_t i = 1; i < nn->params.count; ++i){
        ad_create(tp, nn->params.val_buf[i].data);
    }
    
    // Create and fill input vector
    Vector xs_vec = mlp_create_vector(tp, xs_size);
    for (size_t i = 0; i < xs_size; ++i){
        tp->val_buf[xs_vec.ptr + i].data = xs[i];
    }

    // Forward pass
    Vector out = mlp_forward_pass(nn, tp, xs_vec);
    return out;
}

float mlp_fit(MLP* nn, float* X, size_t X_size, float* Y, size_t Y_size){
    
    Tape tp = {0};
    ad_init_tape(&tp);
    
    Vector out = _predict(nn, &tp, X, X_size);

    // Create and fill ground truth vector
    Vector ys = mlp_create_vector(&tp, Y_size);
    for (size_t i = 0; i < Y_size; ++i){
        tp.val_buf[ys.ptr + i].data = Y[i];
    }
    
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

    // Save loss value
    float ret_loss = tp.val_buf[loss].data;
    
    // Destroy computation graph
    ad_destroy_tape(&tp);

    return ret_loss;
}

void mlp_predict(MLP* nn, float* xs, size_t xs_size, float* out, size_t out_size){

    Tape tp = {0};
    ad_init_tape(&tp);

    Vector out_vec = _predict(nn, &tp, xs, xs_size);
    
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

#define TRAINING_SIZE 4

// Input dataset for the XOR problem 
float X[TRAINING_SIZE][2] = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 1.0f},
};

// Ground truth dataset for the XOR problem 
float Y[TRAINING_SIZE] = {
    0.0f, 
    1.0f,
    1.0f,
    0.0f,
};

int main(void){

    // Initialise multi-layer perceptron
    MLP nn = {0};
    float learning_rate = 1.5f;
    mlp_init(&nn, learning_rate);
    
    // Add layers of neurons 
    mlp_add_layer(&nn, 2, 4, "sigm");
    mlp_add_layer(&nn, 4, 1, "sigm");

    mlp_print(&nn);

    // Train model and print average loss
    printf("Training start...\n");
    #define BATCH_SIZE 1000
    float loss;
    for (size_t n = 0; n < BATCH_SIZE; ++n){
        loss = 0.0f;
        for (size_t i = 0; i < TRAINING_SIZE; ++i){
            loss += mlp_fit(&nn, X[i], 2, Y+i, 1);
        }
        printf("Average loss: %g\n", loss/TRAINING_SIZE);
    }    
    printf("...Training end\n");

    // Prediction
    float out1, out2, out3, out4;
    mlp_predict(&nn, X[0], 2, &out1, 1);
    mlp_predict(&nn, X[1], 2, &out2, 1);
    mlp_predict(&nn, X[2], 2, &out3, 1);
    mlp_predict(&nn, X[3], 2, &out4, 1);

    printf("Prediction for input {0, 0} is %f\n", out1);
    printf("Prediction for input {1, 0} is %f\n", out2);
    printf("Prediction for input {0, 1} is %f\n", out3);
    printf("Prediction for input {1, 1} is %f\n", out4);

    // Destroy model 
    mlp_destroy(&nn);
    
    return 0;
}
