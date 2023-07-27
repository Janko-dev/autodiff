#include "./autodiff_old.h"

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
    float a_val = 5;
    do_demo(a_val);
    printf("--------------\n");
    printf("But the gradient of a is not 4\n");
    printf("When I increase a from 5 -> 6 the value of\n");
    printf("c increases by 3, so the grad of a is 3\n");
    printf("--------------\n");
    do_demo(a_val + 1);
}
