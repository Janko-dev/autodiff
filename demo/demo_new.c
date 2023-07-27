#include "../src/autodiff.h"

void do_demo(float a_val) {
    Tape tape = {0};
    Tape *tp = &tape;
    ad_init_tape(tp);
    size_t a = ad_create(tp, a_val);
    size_t b = ad_create(tp, 10);
    size_t c = ad_add(tp, a, b);
    c = ad_add(tp, c, ad_add(tp, c, a));
    ad_reverse(tp, c);
    printf("a: data: %f | grad: %f\n", GET(a).data, GET(a).grad);
    printf("b: data: %f | grad: %f\n", GET(b).data, GET(b).grad);
    printf("c: data: %f | grad: %f\n", GET(c).data, GET(c).grad);
}

int main() {
    printf("--------------\n");
    printf("Now with the new implementation\n");
    printf("--------------\n");
    float a_val = 5;
    do_demo(a_val);
    printf("--------------\n");
    printf("The gradient of a is now 3\n");
    printf("Which is correct, after increasing the value of\n");
    printf("a from 5->6 the value of c goes from 35->38\n");
    printf("--------------\n");
    do_demo(a_val + 1);
}
