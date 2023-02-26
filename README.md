# Simple Automatic Differentiation library
This repository contains the code for the `autodiff` reverse mode algorithm written in C. Autodiff stands for 'Automatic differentiation', and is used by sophisticated deep learning libraries. This library provides an API for scalar-valued autodiff, rather than tensor-valued autodiff. The algorithm allows the user to compute gradients of mathematical expressions by providing a wrapper for basic arithmetic operations. The wrapper is responsible for building a computation graph, where each operation is stored in the sequence that the computation was performed. Through reverse mode differentiation, we start at the end of the computation, i.e., at the result, and propagate the gradients backwards towards the beginning. 

Consider the following example for a `Neuron` with 2 inputs ($x_1, x_2$), 2 weights ($w_1, w_2$), and a bias ($b$), which uses the `tanh(x)` activation function. 

$$ f(x_1, x_2) = \tanh((w_1 x_1 + w_2 x_2) + b) $$

Its computation graph looks like this:
```bash
tanh(x)
├── (w_1 x_1 + w_2 x_2)
│   ├── css
│   │   ├── **/*.css
│   ├── favicon.ico
│   ├── images
│   ├── index.html
│   ├── js
│   │   ├── **/*.js
│   └── partials/template
├── b
```

## References
- Video detailing the intuition about autodiff: https://www.youtube.com/watch?v=VMj-3S1tku0&ab_channel=AndrejKarpathy