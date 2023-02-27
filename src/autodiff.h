#ifndef _AUTODIFF_H
#define _AUTODIFF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

// Macro for extending dynamic array
#define Extend(n) (n == 0 ? 8 : (n*2))

// Current set of operators
typedef enum {
    ADD,
    SUB,
    MUL,
    POW,
    TANH,
    COUNT,
} OpType;

// Value struct behaving like a node in a graph
// the left/right child node is used for binary operators, 
// while only the left child is used for unary operators
typedef struct Value Value;
struct Value{
    float data;
    float grad;
    OpType op;
    Value* left_child;
    Value* right_child;
};

// Macro for allocating value on the stack
#define VAL(d) (Value){.data=(d), .grad=0.0, .op=COUNT, .left_child=NULL, .right_child=NULL}

// Create a differentiable value 
Value* ad_create(float value);

// Destroy a differentiable value
// Note: this function will also destroy every connected node
void ad_destroy(Value* val);

// autodiff API
Value* ad_add(Value* a, Value* b);
Value* ad_sub(Value* a, Value* b);
Value* ad_mul(Value* a, Value* b);
Value* ad_pow(Value* a, Value* b);
Value* ad_tanh(Value* a);

// Compute gradients of value 
void ad_reverse(Value* y);
// Print computation tree
void ad_print_tree(Value* y);

#endif //_AUTODIFF_H

