#include "autodiff.h"

const char* operators[] = {
    "add ", "sub ", "mul ", "pow ", "tanh", "cons"
};


void init_tape(Tape* tp) {
    tp->val_buf = calloc(INIT_TAPE_SIZE, sizeof(Value));
    tp->cap = INIT_TAPE_SIZE;
    tp->count = 1;
}

void destroy_tape(Tape* tp) {
    free(tp->val_buf);
}

size_t ad_create(Tape* tp, float value){
    if (tp->count >= tp->cap){

        // printf("sizeof(tp) = %d\n", sizeof(*tp->val_buf));
        // for (size_t i = 0; i < tp->cap; ++i){
        //     printf("val: %g, index: %d, left: %d, right: %d\n", 
        //         tp->val_buf[i].data, i, tp->val_buf[i].left_child, tp->val_buf[i].right_child);
        // }
        // printf("-----------------------------------------------------\n");
        tp->cap = Extend(tp->cap);
        tp->val_buf = realloc(tp->val_buf, sizeof(Value) * tp->cap);
        if (!tp->val_buf) {
            fprintf(stderr, "Not enough memory, buy more ram!\n");
            exit(1);
        }
        // printf("sizeof(tp) = %d\n", sizeof(*tp->val_buf));
        // for (size_t i = 0; i < tp->cap; ++i){
        //     printf("val: %g, index: %d, left: %d, right: %d\n", 
        //         tp->val_buf[i].data, i, tp->val_buf[i].left_child, tp->val_buf[i].right_child);
        // }
    }
    
    Value* res = tp->val_buf + tp->count;
    res->data = value;
    res->op = COUNT;
    tp->count++;
    return tp->count-1;
}

size_t ad_add(Tape* tp, size_t a, size_t b){
    float data = GET(a).data + GET(b).data;
    size_t out = ad_create(tp, data);
    GET(out).left_child = a;
    GET(out).right_child = b;
    GET(out).op = ADD;
    return out;
}

size_t ad_sub(Tape* tp, size_t a, size_t b){
    float data = GET(a).data - GET(b).data;
    size_t out = ad_create(tp, data);
    GET(out).left_child = a;
    GET(out).right_child = b;
    GET(out).op = SUB;
    return out;
}

size_t ad_mul(Tape* tp, size_t a, size_t b){
    float data = GET(a).data * GET(b).data;
    size_t out = ad_create(tp, data);
    GET(out).left_child = a;
    GET(out).right_child = b;
    GET(out).op = MUL;
    return out;
}

size_t ad_pow(Tape* tp, size_t a, size_t b){
    float data = powf(GET(a).data, GET(b).data);
    size_t out = ad_create(tp, data);
    GET(out).left_child = a;
    GET(out).right_child = b;
    GET(out).op = POW;
    return out;
}

size_t ad_tanh(Tape* tp, size_t a){
    size_t out = ad_create(tp, tanh(GET(a).data));
    GET(out).left_child = a;
    GET(out).right_child = 0;
    GET(out).op = TANH;
    return out;
}

void _ad_reverse(Tape* tp, size_t y){
    Value y_deref = GET(y);
    switch (GET(y).op){
        case SUB: {
            GET(y_deref.left_child).grad += y_deref.grad * 1.0f;
            GET(y_deref.right_child).grad += y_deref.grad * -1.0f;
        } break;
        case ADD: {
            GET(y_deref.left_child).grad += y_deref.grad * 1.0f;
            GET(y_deref.right_child).grad += y_deref.grad * 1.0f;
        } break;
        case MUL: {
            GET(y_deref.left_child).grad += y_deref.grad * GET(y_deref.right_child).data;
            GET(y_deref.right_child).grad += y_deref.grad * GET(y_deref.left_child).data;
        } break;
        case POW: {
            float l_data = GET(y_deref.left_child).data;
            float r_data = GET(y_deref.right_child).data;
            GET(y_deref.left_child).grad += y_deref.grad * r_data * powf(l_data, r_data - 1);
            GET(y_deref.right_child).grad += y_deref.grad * log(l_data) * powf(l_data, r_data);
        } break;
        case TANH: {
            GET(y_deref.left_child).grad += y_deref.grad * (1 - y_deref.data*y_deref.data);
        } break;
        default: break;
    }
    if (y_deref.left_child != 0) _ad_reverse(tp, y_deref.left_child);
    if (y_deref.right_child != 0) _ad_reverse(tp, y_deref.right_child);
}

void ad_reverse(Tape* tp, size_t y){
    GET(y).grad = 1.0;
    _ad_reverse(tp, y);
}

void _ad_print_tree(Tape* tp, size_t y, size_t indent){
    if (y == 0) return;
    Value y_deref = GET(y);
    for (size_t i = 0; i < indent; ++i) printf(" ");
    printf("[%s] node (data: %g, grad: %g)\n", operators[y_deref.op], y_deref.data, y_deref.grad);
    _ad_print_tree(tp, y_deref.left_child, indent+4);
    _ad_print_tree(tp, y_deref.right_child, indent+4);
}

void ad_print_tree(Tape* tp, size_t y){
    printf("------------- Computation graph -------------\n");
    _ad_print_tree(tp, y, 0);
    printf("--------------------------------------------\n");
}

void ad_print_tape(Tape* tp){
    for (size_t i = 0; i < tp->count; ++i){
        printf("val: %g, index: %d, left: %d, right: %d, op: %d\n", 
            tp->val_buf[i].data, i, tp->val_buf[i].left_child, tp->val_buf[i].right_child, tp->val_buf[i].op);
    }
}