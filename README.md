1. Tensors
2. Loss Functions
3. Layers
4. Neural Nets
5. Optimizers
6. Data
7. Training


After building different classes, and in order to make the idea of a feedforward network more concrete, we begin with the Learning XOR.
Basically, it is an example of a fully functioning feedforward network on a very simple task: learning the XOR function.

The XOR function (“exclusive or”) is an operation on two binary values, x1 and x2. When exactly one of these binary values is equal to 1, the XOR function
returns 1. Otherwise, it returns 0. The XOR function provides the target function y = f∗(x) that we want to learn. Our model provides a function y = f(x;θ) and
our learning algorithm will adapt the parameters θ to make f as similar as possible to f∗.



