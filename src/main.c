#include "mlp.h"

#define TRAINING_SIZE 4

// Input dataset for the XOR problem 
float X[TRAINING_SIZE][2] = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 1.0f}
};

// Ground truth dataset for the XOR problem 
float Y[TRAINING_SIZE] = {
    0.0f, 
    1.0f,
    1.0f,
    0.0f
};

int main(void){

    // Initialise multi-layer perceptron
    MLP nn = {0};
    float learning_rate = 0.1f;
    mlp_init(&nn, learning_rate);
    
    // Add layers of neurons 
    mlp_add_layer(&nn, 2, 6, "sigm");
    mlp_add_layer(&nn, 6, 8, "relu");
    mlp_add_layer(&nn, 8, 1, "relu");

    mlp_print(&nn);

    // Train model and print loss
    printf("Training start...\n");
    #define BATCH_SIZE 40
    for (size_t n = 0; n < 50; ++n){
        float loss = 0.0f;
        for (size_t i = 0; i < BATCH_SIZE; ++i){
            // float loss = 0;
            size_t index = i % TRAINING_SIZE;
            loss += mlp_fit(&nn, X[index], 2, Y+index, 1);
            // for (size_t j = 0; j < TRAINING_SIZE; ++j){
            // }
        }
        printf("Average loss: %g\n", loss/BATCH_SIZE);
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

    ad_print_tape(&nn.params);

    // Destroy model 
    mlp_destroy(&nn);
    
    return 0;
}
