#include "autodiff.h"

float mlp_rand(){
    return ((float)rand() / (float)RAND_MAX) * 2.0 - 1.0;
}

int main(void){

    Tape tp = {0};
    ad_init_tape(&tp);

    // f(a, b) = (a + b) + ((a + b) + a) -> 3a + 2b
    // f' w.r.t. a is 3
    // f' w.r.t. b is 2
    size_t a = ad_create(&tp, 5);
    size_t b = ad_create(&tp, 10);
    size_t c = ad_add(&tp, a, b);
    c = ad_add(&tp, c, ad_add(&tp, c, a));
    
    // ad_reverse(&tp, c);
    ad_reverse_toposort(&tp, c);

    ad_print_tape(&tp);
    ad_print_tree(&tp, c);

    printf("grad of a: %g\n", tp.val_buf[a].grad);
    printf("grad of b: %g\n", tp.val_buf[b].grad);

    ad_destroy_tape(&tp);
    return 0;
}
