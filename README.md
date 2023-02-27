# Simple Automatic Differentiation library

This repository contains the implementation of scalar-valued reverse mode `autodiff` written in the C language. I implemented autodiff in C for educational and recreational purposes. Languages that offer operator overloading and a garbage collector would be ideal for an autodiff implementation. Nonetheless, implementing autodiff in C allows for many interesting implementation details.

## What is `autodiff`?
Automatic differentiation (or simply `autodiff`) is the key method used by sophisticated deep learning libraries (e.g. Pytorch and Tensorflow) to garner the gradients of arbitrary computations. The gradients are then used to perform backpropagation through an artificial neural network model. The key difference between this implementation and that of Pytorch or Tensorflow is that this implementation considers gradients of scalar values, while Pytorch/Tensorflow operate on multi-dimensional arrays, and thus consider gradients of tensor values. As opposed to other methods for computing gradients, the strength of autodiff lies in accuracy and simplicity to implement. We can distinguish two variants of autodiff, forward mode and reverse mode. This repository is concerned with the reverse mode autodiff, which means the derivative of a result with respect to its inputs is obtained by starting at the result and propagating the gradient backwards through previous computations.  

Generally, we are used to computing gradients based on differentiating the symbolic expression of a function $f(x)$ to another expression that is the derivative of $f$. For instance, $f(x) = x^2$ can be differentiated, which results in $f'(x) = 2x$. Symbolically parsing and transforming functions into its derivative is not scalable as the functions grow in size. Take for instance a moderately sized neural network, which consists of possibly hundreds of function compositions that would result in a massive symbolic expression. The derivative of which is not feasable to compute using symbolic differentiation. Thus, this approach is rejected for computing gradients. 

Another method that we can use is the limit definition of the derivative. That is, the following limit.

$$
    \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

This is very easy to compute. We just choose a small value for `h` and plug it into the formula. However, this will not be accurate as we are limited to the finite size of floating point values. 

At last, consider automatic differentiation, which is neither symbolic differentiation nor uses the limit definition. Rather, it uses the elementary building blocks of simple mathematical expressions and chains them together using the chain rule of calculus. In the `autodiff.c` file, the `Value` structure is used as a wrapper for arbitrary float values. Using the functions provided in the API prefixed with `ad_`, one can build the computation graph. 

For example the expression `(2 + 3) * 4` can be computed using the following code snippet.
```C
Value* a = ad_create(2.0f, true); 
Value* b = ad_create(3.0f, true);
Value* c = ad_create(4.0f, true);

Value* d = ad_add(a, b);
Value* e = ad_mul(d, c);

ad_destroy(e);
```
The code snippet above generates a graph of computations. Consider the following graph. 

![comp_graph](img/comp_graph.drawio.svg)

We can encode custom functions to differentiate simple additions, multiplications, and any other computation (this is not limited to only mathematical computations) and use the chain rule to derive the derivative of the output with respect to every possible node in the graph. The reverse mode method starts at the end of the computation, i.e., at the result, and propagates the gradients backwards towards the beginning. 

In the above example, it is easy to compute the local derivatives of the computations. Starting at node `e`, the derivative of `e` with respect to (w.r.t.) `e` is just 1. Looking at the children of `e`, we have `d` and `c`. The derivative of `e` w.r.t. `c` is `d`, since `e = d * c`. And the derivative of `e` w.r.t. `d` is `c`, since `e = d * c`. Now `d` also has children, so let's compute the derivatives. The derivative of `d` w.r.t. `a` and w.r.t. `b` is 1, since `d = a + b`. Up until now, we have computed the local derivatives. Using the chain rule we can multiply the local derivatives to obtain the global derivatives with respect to the inputs `a` and `b`. Mathematics has a nice way to formulate this notion using the partial form ($\partial$).

$$
    \frac{\partial e}{\partial a} = \frac{\partial e}{\partial d} \cdot \frac{\partial d}{\partial a} \qquad \text{and} \qquad \frac{\partial e}{\partial b} = \frac{\partial e}{\partial d} \cdot \frac{\partial d}{\partial b}
$$

The following code snippet performs this method. Notice the addition of the function `ad_reverse(e)` which performs autodiff on its computation graph. Also notice the addition of `ad_print_tree(e)`, which prints the computation graph, its values, and its gradients. 
```C
Value* a = ad_create(2.0f, true); 
Value* b = ad_create(3.0f, true);
Value* c = ad_create(4.0f, true);

Value* d = ad_add(a, b);
Value* e = ad_mul(d, c);

ad_reverse(e);
ad_print_tree(e);

ad_destroy(e);
```
```
------------- Computation graph -------------
[mul ] node (data: 20, grad: 1)
    [add ] node (data: 5, grad: 4)
        [noop] node (data: 2, grad: 4)
        [noop] node (data: 3, grad: 4)
    [noop] node (data: 4, grad: 5)
--------------------------------------------
```
## Usage
The build system is `make` and there are no external dependencies. The implementation of autodiff is contained in `autodiff.c` and `autodiff.h`. Furthermore, I included `nn.c` and `nn.h` as an example of autodiff. The example shows the application of autodiff in a simple multi layer perceptron.  
```
$ make 
$ ./autodiff
```

## Example

Consider the following example for an artificial `Neuron` with 2 inputs ($x_1, x_2$), 2 weights ($w_1, w_2$), and a bias ($b$), which uses the `tanh(x)` activation function. 

$$ f(x_1, x_2) = \tanh((w_1 x_1 + w_2 x_2) + b) $$

Consider the following computation graph. The goal is to find the derivatives of `y` w.r.t. the parameters of the neuron. These are `w1`, `w2`, and `b`.  

![neuron](img/neuron.drawio.svg)

The following code snippet computes this expression for some arbitrary values, and thereafter, computes and prints the gradients. 
```C
// The inputs x1, x2
Value* x1 = ad_create(-1.0f, false);
Value* x2 = ad_create(2.0f, false);

// The params w1, w2, b
Value* w1 = ad_create(4.0f, true);
Value* w2 = ad_create(-2.0f, true);
Value* b  = ad_create(.5f, true);

// Intermediate computations
Value* xw1 = ad_mul(x1, w1);
Value* xw2 = ad_mul(x2, w2);
Value* xw  = ad_add(xw1, xw2);
Value* xwb = ad_add(xw, b);

// The result 
Value* y = ad_tanh(xwb);

ad_reverse(y);
ad_print_tree(y);

ad_destroy(y);
```
```
------------- Computation graph -------------
[tanh] node (data: -0.999999, grad: 1)
    [add ] node (data: -7.5, grad: 1.19209e-006)      
        [add ] node (data: -8, grad: 1.19209e-006)    
            [mul ] node (data: -4, grad: 1.19209e-006)
                [noop] node (data: -1, grad: 4.76837e-006)
                [noop] node (data: 4, grad: -1.19209e-006)
            [mul ] node (data: -4, grad: 1.19209e-006)
                [noop] node (data: 2, grad: -2.38419e-006)
                [noop] node (data: -2, grad: 2.38419e-006)
        [noop] node (data: 0.5, grad: 1.19209e-006)
--------------------------------------------
```

## References
- Introduction to autodiff: https://arxiv.org/pdf/2110.06209.pdf 
- Tensorflow autodiff API: https://www.tensorflow.org/guide/autodiff
- Video detailing the intuition of autodiff: https://www.youtube.com/watch?v=VMj-3S1tku0&ab_channel=AndrejKarpathy