#include "autodiff.h"

const char* operators[] = {
    "add ", "mul ", "pow ", "tanh", "noop"
};

Value* ad_create(float value, bool has_grad){
    Value* res = calloc(1, sizeof(Value));
    res->has_grad = has_grad;
    res->data = value;
    res->op = COUNT;
    return res;
}

void ad_destroy(Value* val){
    if (val == NULL) return;
    printf("destroying val ptr: %p ", val);
    printf("with val: %f op: %s and (left: %p, right: %p)\n", val->data, operators[val->op], val->left_child, val->right_child);
    if (val->left_child != NULL) ad_destroy(val->left_child);
    if (val->right_child != NULL) ad_destroy(val->right_child);
    free(val);
    val = NULL;
}

Value* ad_add(Value* a, Value* b){
    Value* out = ad_create(a->data + b->data, a->has_grad + b->has_grad);
    out->left_child = a;
    out->right_child = b;
    out->op = ADD;
    return out;
}

Value* ad_mul(Value* a, Value* b){
    Value* out = ad_create(a->data * b->data, a->has_grad + b->has_grad);
    out->left_child = a;
    out->right_child = b;
    out->op = MUL;
    return out;
}

Value* ad_pow(Value* a, Value* b){
    Value* out = ad_create(pow(a->data, b->data), a->has_grad + b->has_grad);
    out->left_child = a;
    out->right_child = b;
    out->op = POW;
    return out;
}

Value* ad_tanh(Value* a){
    Value* out = ad_create(tanh(a->data), a->has_grad);
    out->left_child = a;
    out->op = TANH;
    return out;
}

Value* ad_dot_product(Value** xs, Value** w, size_t num){
    Value* out = ad_create(0.0f, true);
    for (size_t i = 0; i < num; ++i){
        Value* xw = ad_mul(xs[i], w[i]);
        out = ad_add(xw, out);
    }
    return out;
}

void _ad_reverse(Value* y){
    if (!y->has_grad) return;
    switch (y->op){
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
    printf("[%s] node (data: %f, grad: %f)   %p\n", operators[y->op], y->data, y->grad, y);
    _ad_print_tree(y->left_child, indent+2);
    _ad_print_tree(y->right_child, indent+2);
}

void ad_print_tree(Value* y){
    printf("------------- Computation tree -------------\n");
    _ad_print_tree(y, 0);
    printf("--------------------------------------------\n");
}