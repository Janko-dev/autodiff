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
    float learning_rate = 0.01f;
    mlp_init(&nn, learning_rate);
    
    // Add layers of neurons 
    mlp_add_layer(&nn, 2, 3, "tanh");
    mlp_add_layer(&nn, 3, 1, "tanh");

    mlp_print(&nn);
    ad_print_tape(&nn.params);

    // Train model and print loss
    printf("Training start...\n");
    #define BATCH_SIZE 80
    for (size_t n = 0; n < BATCH_SIZE; ++n){
        float loss = 0.0f;
        // nn.learning_rate = 1.0f - 0.9f * (float)n/(float)BATCH_SIZE;
        for (size_t i = 0; i < TRAINING_SIZE; ++i){
            loss += mlp_fit(&nn, X[i], 2, Y+i, 1);
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
