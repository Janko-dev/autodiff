#include "autodiff.h"

// For debugging purposes (such as printing the computation tree)
// note that the order matters.
const char* operators[] = {
    "add ", "sub ", "mul ", "pow ", "tanh", "relu", "sigm", "nil"
};

// Initialise tape by providing pointer to Tape.
// Tape should be destroyed after its use by calling `void ad_destroy_tape(Tape* tp)`
void ad_init_tape(Tape* tp) {
    tp->val_buf = calloc(INIT_TAPE_SIZE, sizeof(Value));
    tp->cap = INIT_TAPE_SIZE;
    tp->count = 1;
}

void ad_destroy_tape(Tape* tp) {
    free(tp->val_buf);
}

// Create new floating point (f32) variable value, 
// which is a leaf node of the computation graph. 
size_t ad_create(Tape* tp, float value){
    if (tp->count >= tp->cap){

        tp->cap = Extend(tp->cap);
        tp->val_buf = realloc(tp->val_buf, sizeof(Value) * tp->cap);
        if (!tp->val_buf) {
            fprintf(stderr, "Not enough memory, buy more ram!\n");
            exit(1);
        }
    }
    
    Value* res = tp->val_buf + tp->count;
    res->data = value;
    res->grad = 0.0f;
    res->left_child = 0;
    res->right_child = 0;
    res->op = COUNT;
    tp->count++;
    return tp->count-1;
}

// Macro to define addition, subtraction, and multiplication functions.
#define AD_OPERATOR_FUNC_BINARY(op_symbol, op_type, op_name) \
size_t ad_##op_name(Tape* tp, size_t a, size_t b) { \
    float data = GET(a).data op_symbol GET(b).data; \
    size_t out = ad_create(tp, data); \
    GET(out).left_child = a; \
    GET(out).right_child = b; \
    GET(out).op = op_type; \
    return out; \
}\

AD_OPERATOR_FUNC_BINARY(+, ADD, add)
AD_OPERATOR_FUNC_BINARY(-, SUB, sub)
AD_OPERATOR_FUNC_BINARY(*, MUL, mul)

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

size_t ad_relu(Tape* tp, size_t a){
    size_t out = ad_create(tp, GET(a).data > 0 ? GET(a).data : 0);
    GET(out).left_child = a;
    GET(out).right_child = 0;
    GET(out).op = RELU;
    return out;
}

float sigmoid(float x) {
    return 1/(1 + exp(-x));
}

size_t ad_sigm(Tape* tp, size_t a){
    size_t out = ad_create(tp, sigmoid(GET(a).data));
    GET(out).left_child = a;
    GET(out).right_child = 0;
    GET(out).op = SIGM;
    return out;
}

// For each value on the tape, the gradient of its parents
// is updated, i.e., the local gradients are flowing from the root
// of the computation graph towards the leaves in a topological order. 
void _ad_reverse(Tape* tp, size_t y){
    Value y_deref = GET(y);
    switch (y_deref.op){
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
        case RELU: {
            if (y_deref.data > 0) 
                GET(y_deref.left_child).grad += y_deref.grad * 1.0f;
        } break;
        case SIGM: {
            GET(y_deref.left_child).grad += y_deref.grad * y_deref.data * (1 - y_deref.data);
        } break;
        default: break;
    }
}

void ad_reverse(Tape* tp, size_t y){
    // Initial gradient is always 1 
    // because the derivative of x w.r.t. x = 1
    GET(y).grad = 1.0;
    // We traverse the computation graph topologically in reverse order, 
    // so that every node is visited exactly once. Note that this only works 
    // when adhering to the provided ad_ api for constructing the computation graph.
    //
    // When the order of the tape is not topologically consistent, i.e., is not sorted, 
    // then consider using `void ad_reverse_toposort(Tape* tp, size_t y)` to first sort the nodes
    // and only then traverse the graph. 
    for (size_t i = tp->count-1; i >= 1; --i){
        _ad_reverse(tp, i);
    }
}

// Depth first search through computation graph to recursively obtain topologically sorted nodes 
void _ad_topo(Tape *tp, size_t* topo_nodes, bool* visited, size_t y, size_t* count) {
    visited[y] = true;
    if(GET(y).left_child != 0 && !visited[GET(y).left_child]) {
        _ad_topo(tp, topo_nodes, visited, GET(y).left_child, count);
    }
    if (GET(y).right_child != 0 && !visited[GET(y).right_child]) {
        _ad_topo(tp, topo_nodes, visited, GET(y).right_child, count);
    }
    topo_nodes[*count] = y;
    (*count)++;
}

void ad_reverse_toposort(Tape* tp, size_t y) {
    GET(y).grad = 1.0f;
    size_t count = 0;
    
    // List of sorted graph indices   
    size_t *sorted_nodes = malloc(tp->count * sizeof(size_t));
    bool *visited = calloc(tp->count, sizeof(bool));

    _ad_topo(tp, sorted_nodes, visited, y, &count);

    // traverse the topologically sorted nodes in reverse order
    for (size_t i = count-1; i > 0; --i) {
        _ad_reverse(tp, sorted_nodes[i]);
    }

    free(sorted_nodes);
    free(visited);
}

// Printing utility
void _ad_print_tree(Tape* tp, size_t y, size_t indent){
    if (y == 0) return;
    Value y_deref = GET(y);
    for (size_t i = 0; i < indent; ++i) printf(" ");
    printf("[idx: %d, %s] node (data: %g, grad: %g)\n", y, operators[y_deref.op], y_deref.data, y_deref.grad);
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
        printf("val: %2g, index: %3zu, left: %3zu, right: %3zu, op: %s\n", 
            tp->val_buf[i].data, i, tp->val_buf[i].left_child, tp->val_buf[i].right_child, operators[tp->val_buf[i].op]);
    }
}