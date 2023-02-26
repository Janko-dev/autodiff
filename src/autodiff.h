#ifndef _AUTODIFF_H
#define _AUTODIFF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#define Extend(n) (n == 0 ? 8 : (n*2))

typedef enum {
    ADD,
    MUL,
    POW,
    TANH,
    COUNT,
} OpType;

typedef struct Value Value;
struct Value{
    float data;
    float grad;
    bool has_grad;
    OpType op;
    Value* left_child;
    Value* right_child;
};

// Macro for allocating value on the stack
#define VAL(d) (Value){.data=(d), .grad=0.0, .op=COUNT, .has_grad=true, .left_child=NULL, .right_child=NULL}

Value* ad_create(float value, bool has_grad);
void ad_destroy(Value* val);

Value* ad_add(Value* a, Value* b);
Value* ad_mul(Value* a, Value* b);
Value* ad_pow(Value* a, Value* b);
Value* ad_tanh(Value* a);

Value* ad_dot_product(Value** xs, Value** w, size_t num);

void ad_reverse(Value* y);
void ad_print_tree(Value* y);

#endif //_AUTODIFF_H

