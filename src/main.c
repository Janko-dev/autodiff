#include "nn.h"
#include <time.h>

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

    srand(time(NULL));

    // Initialise multi-layer perceptron
    MLP nn = {0};
    float learning_rate = 0.05f;
    init_nn(&nn, learning_rate);
    
    // add layers of neurons 
    add_layer(&nn, 2, 4);
    add_layer(&nn, 4, 4);
    add_layer(&nn, 4, 1);
    print_nn(&nn);

    // Train model and print loss
    printf("Training start...\n");
    for (size_t i = 0; i < 10; ++i){
        float loss = 0;
        for (size_t j = 0; j < TRAINING_SIZE; ++j){
            loss += fit(&nn, X[j], 2, Y+j, 1);
        }
        printf("Average loss: %f\n", loss/TRAINING_SIZE);
    }
    printf("...Training end\n");

    destroy_nn(&nn);
    
    return 0;
}