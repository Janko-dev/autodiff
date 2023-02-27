#include "autodiff.h"

const char* operators[] = {
    "add ", "sub ", "mul ", "pow ", "tanh", "cons"
};

Value* ad_create(float value){
    Value* res = calloc(1, sizeof(Value));
    res->data = value;
    res->op = COUNT;
    return res;
}

bool visited(Value** list, size_t len, Value* val){
    for (size_t i = 0; i < len; ++i) {
        if (list[i] == val) return true;
    }
    return false;
}

void traverse_add(Value* val, Value** list, size_t* len, size_t* cap){
    if (val == NULL) return;
    if (*len >= *cap){
        *cap = Extend(*cap);
        list = realloc(list, sizeof(Value*) * *cap);
    }
    if (!visited(list, *len, val)) 
        list[(*len)++] = val;
    if (val->left_child != NULL) 
        traverse_add(val->left_child, list, len, cap);
    if (val->right_child != NULL) 
        traverse_add(val->right_child, list, len, cap);
}

void ad_destroy(Value* val){
    Value** list = malloc(sizeof(Value*) * 8);
    size_t len = 0;
    size_t cap = 8;
    traverse_add(val, list, &len, &cap);
    for (size_t i = 0; i < len; ++i){
        free(list[i]);
    }
    free(list);
}

Value* ad_add(Value* a, Value* b){
    Value* out = ad_create(a->data + b->data);
    out->left_child = a;
    out->right_child = b;
    out->op = ADD;
    return out;
}

Value* ad_sub(Value* a, Value* b){
    Value* out = ad_create(a->data - b->data);
    out->left_child = a;
    out->right_child = b;
    out->op = SUB;
    return out;
}

Value* ad_mul(Value* a, Value* b){
    Value* out = ad_create(a->data * b->data);
    out->left_child = a;
    out->right_child = b;
    out->op = MUL;
    return out;
}

Value* ad_pow(Value* a, Value* b){
    Value* out = ad_create(pow(a->data, b->data));
    out->left_child = a;
    out->right_child = b;
    out->op = POW;
    return out;
}

Value* ad_tanh(Value* a){
    Value* out = ad_create(tanh(a->data));
    out->left_child = a;
    out->op = TANH;
    return out;
}

void _ad_reverse(Value* y){
    switch (y->op){
        case SUB: {
            y->left_child->grad += y->grad * 1.0f;
            y->right_child->grad += y->grad * -1.0f;
        } break;
        case ADD: {
            y->left_child->grad += y->grad * 1.0f;
            y->right_child->grad += y->grad * 1.0f;
        } break;
        case MUL: {
            y->left_child->grad += y->grad * y->right_child->data;
            y->right_child->grad += y->grad * y->left_child->data;
        } break;
        case POW: {
            y->left_child->grad += y->grad * y->right_child->data * pow(y->left_child->data, y->right_child->data-1); 
            y->right_child->grad += y->grad * log(y->left_child->data) * pow(y->left_child->data, y->right_child->data);
        } break;
        case TANH: {
            y->left_child->grad += y->grad * (1 - y->data*y->data);
        } break;
        default: break;
    }
    if (y->left_child != NULL) _ad_reverse(y->left_child);
    if (y->right_child != NULL) _ad_reverse(y->right_child);
}

void ad_reverse(Value* y){
    y->grad = 1.0;
    _ad_reverse(y);
}

void _ad_print_tree(Value* y, size_t indent){
    if (y == NULL) return;
    for (size_t i = 0; i < indent; ++i) printf(" ");
    printf("[%s] node (data: %g, grad: %g)\n", operators[y->op], y->data, y->grad);
    _ad_print_tree(y->left_child, indent+4);
    _ad_print_tree(y->right_child, indent+4);
}

void ad_print_tree(Value* y){
    printf("------------- Computation graph -------------\n");
    _ad_print_tree(y, 0);
    printf("--------------------------------------------\n");
}