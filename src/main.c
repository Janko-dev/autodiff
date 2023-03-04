#include "mlp.h"

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

int main(){

    Tape tp = {0};
    ad_init_tape(&tp);

    // The inputs x1, x2
    size_t x1 = ad_create(&tp, -1.0f);
    size_t x2 = ad_create(&tp, 2.0f);

    // The params w1, w2, b
    size_t w1 = ad_create(&tp, 4.0f);
    size_t w2 = ad_create(&tp, -2.0f);
    size_t b  = ad_create(&tp, .5f);

    // Intermediate computations
    size_t xw1 = ad_mul(&tp, x1, w1);
    size_t xw2 = ad_mul(&tp, x2, w2);
    size_t xw  = ad_add(&tp, xw1, xw2);
    size_t xwb = ad_add(&tp, xw, b);

    // The result 
    size_t y = ad_tanh(&tp, xwb);

    ad_reverse(&tp, y);
    ad_print_tree(&tp, y);

    ad_destroy_tape(&tp);
    return 0;
}

int main2(void){

    // Initialise multi-layer perceptron
    MLP nn = {0};
    float learning_rate = 0.5f;
    mlp_init(&nn, learning_rate);
    
    // Add layers of neurons 
    mlp_add_layer(&nn, 2, 16, "sigm");
    mlp_add_layer(&nn, 16, 1, "sigm");

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
