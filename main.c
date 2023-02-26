// #include "autodiff.h"
#include "nn.h"
#include <time.h>

#define TRAINING_SIZE 4

float X[TRAINING_SIZE][2] = {
    {0.0f, 0.0f},
    {1.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 1.0f}
};

float Y[TRAINING_SIZE] = {
    0.0f, 
    1.0f,
    1.0f,
    0.0f
};

int main(void){

    // Value x1, x2, w1, w2, bias, xw1, xw2, xw1xw2, n, out, diff, loss;

    // w1 = VAL(2.0f);
    // w2 = VAL(-0.5f);
    // bias = VAL(1.0f);
    // float learning_rate = 0.01;

    // for (size_t j = 0; j < 20; ++j){

    //     for (size_t i = 0; i < 2; ++i){
    //         x1 = VAL(dataset[i]);
    //         x2 = VAL(dataset[i+1]);
            
    //         xw1 = ad_mul(&x1, &w1);
    //         xw2 = ad_mul(&x2, &w2);
            
    //         xw1xw2 = ad_add(&xw1, &xw2);
    //         n = ad_add(&xw1xw2, &bias);
    //         out = ad_tanh(&n);

    //         diff = ad_add(&VAL(-dataset[i+2]), &out);
    //         loss = ad_pow(&diff, &VAL(2));
            
    //         ad_reverse(&loss);
    //         ad_print_tree(&loss);

    //         w1.data += learning_rate * w1.grad;
    //         w2.data += learning_rate * w2.grad;
    //         bias.data += learning_rate * bias.grad;
    //     }
    // }

    // srand(time(NULL));
    // MLP* nn = create_nn();
    // add_layer(nn, 2, 2);
    // add_layer(nn, 2, 1);
    // // fit(nn, X, TRAINING_SIZE, Y, TRAINING_SIZE);
    
    // destroy_nn(nn);

    // Value* a = ad_create(3.0f);
    // Value* b = ad_create(5.0f);
    // a = ad_mul(a, b);
    // ad_reverse(a);
    // ad_print_tree(a);
    // ad_destroy(a);

    // Matrix mat = create_matrix(3, 2, true);
    // Vector vec = create_vector(2, true);

    // print_mat(mat);
    // print_vec(vec);

    // Vector output = mat_vec_prod(mat, vec);
    // print_vec(output);

    // // ad_print_tree(output.data[0]);
    // ad_print_tree(output.data[1]);
    // // ad_reverse(output.data[1]);
    // // ad_print_tree(output.data[1]);
    // // ad_print_tree(output.data[2]);
    // destroy_vector(output);
    // printf("(%p, %p)\n", output.data[0]->left_child, output.data[0]->right_child);
    // printf("(%p, %p)\n", output.data[1]->left_child, output.data[1]->right_child);
    // printf("(%p, %p)\n", output.data[2]->left_child, output.data[2]->right_child);
    // destroy_vector(vec);
    // destroy_matrix(mat);

    Value* a = ad_create(2.0f, true); 
    Value* b = ad_create(3.0f, true);
    b = ad_mul(a, b);
    b = ad_mul(b, b);
    ad_print_tree(b);
    ad_destroy(b);
    
    return 0;
}