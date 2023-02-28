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
typedef struct {
    float data;
    float grad;
    OpType op;
    size_t left_child;
    size_t right_child;
} Value;

typedef struct {
    Value* val_buf;
    size_t count;
    size_t cap;
} Tape;

#define GET(v) tp->val_buf[(v)]
#define INIT_TAPE_SIZE 8

void init_tape(Tape* tape);
void destroy_tape(Tape* tape);
void ad_print_tape(Tape* tp);

// Create a differentiable value 
size_t ad_create(Tape* tp, float value);

// autodiff API
size_t ad_add(Tape* tp, size_t a, size_t b);
size_t ad_sub(Tape* tp, size_t a, size_t b);
size_t ad_mul(Tape* tp, size_t a, size_t b);
size_t ad_pow(Tape* tp, size_t a, size_t b);
size_t ad_tanh(Tape* tp, size_t a);

// Compute gradients of value 
void ad_reverse(Tape* tp, size_t y);
// Print computation tree
void ad_print_tree(Tape* tp, size_t y);

#endif //_AUTODIFF_H

