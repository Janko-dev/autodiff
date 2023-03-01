#ifndef _AUTODIFF_H
#define _AUTODIFF_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

// Macro for extending dynamic array
#define Extend(n) (n == 0 ? 8 : (n*2))

// Current set of operators
// Adding more operators implies adding the corresponding functions
typedef enum {
    ADD,
    SUB,
    MUL,
    POW,
    TANH,
    RELU,
    SIGM,
    COUNT,
} OpType;

// Value struct behaving like a node in a graph
// It is aware of its operator type and has a 
// reference to its operands in an array linked list
typedef struct {
    float data;
    float grad;
    OpType op;
    size_t left_child;
    size_t right_child;
} Value;

// The tape struct is a dynamic array of values
// which function as an array linked list
typedef struct {
    Value* val_buf;
    size_t count;
    size_t cap;
} Tape;

// Most params of the tape are Tape pointers 'tp'
// which leads to the usefulness of this macro 
#define GET(v) tp->val_buf[(v)]

// The initial tape size.
// recommended to be a multiple of 2
#define INIT_TAPE_SIZE 8

// Gradient tape functions 
void ad_init_tape(Tape* tape);
void ad_destroy_tape(Tape* tape);
void ad_print_tape(Tape* tp);

// Create a differentiable value
// that gets added to the provided tape
size_t ad_create(Tape* tp, float value);

// autodiff API for common operations
// note that the operands are 'size_t' and serve as array pointers for the tape
size_t ad_add(Tape* tp, size_t a, size_t b);
size_t ad_sub(Tape* tp, size_t a, size_t b);
size_t ad_mul(Tape* tp, size_t a, size_t b);
size_t ad_pow(Tape* tp, size_t a, size_t b);

// Common activation functions
size_t ad_tanh(Tape* tp, size_t a);
size_t ad_relu(Tape* tp, size_t a);
size_t ad_sigm(Tape* tp, size_t a);

// Compute gradients of value in reverse mode
void ad_reverse(Tape* tp, size_t y);

// Print computation tree
void ad_print_tree(Tape* tp, size_t y);

#endif //_AUTODIFF_H

