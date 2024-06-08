r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

student_name_1 = 'Hadar Eshed'
student_ID_1 = '205947856'
student_name_2 = 'Yuval Tomer'
student_ID_2 = '207163783'

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**1.1.** Since ${X}$ has shape $64\times512$ and ${Y}$ has shape $64\times1024$, the shape of the Jacobian tensor is
         $(64\times512)\times(64\times1024)$.\
**1.2.** The Jacobian is not sparse, as each element of the output for the $i$-th sample depends on the elements of ${X}$
         and ${W}$.\
**1.3.** We don't need to materialize the Jacobian in order to calculate the downstream gradient, because we can use the
         chain rule and compute it with matrix multiplication.\
\
**2.1.** Since ${Y}$ has shape $64\times512$ and ${W}$ has shape $512\times1024$, the shape of the Jacobian tensor is
         $(64\times512)\times(512\times1024)$.\
**2.2.** Yes, it's sparse. Each output element depends only on the weights connecting it to the corresponding input
         features, so it has a block diagonal structure. Each block corresponds to a single output feature and has
         non-zero elements only in the rows corresponding to the input features connected to that output feature.\
**2.3.** We don't need to materialize the Jacobian in order to calculate the downstream gradient, because we can use the
         chain rule to compute $X^T\times\delta Y$.\
\
\
"""

part1_q2 = r"""
Back-propagation is not required to train neural networks, but it is the a very efficient method for calculating
gradients and updating weights. An alternative for training neural networks with decent-based optimization could be
random weight changes. This approach is impractical and inefficient compared to back-propagation. Other methods like
reinforcement learning can be used, but they are sometimes less efficient.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0

    wstd, lr, reg = 0.1, 0.1, 0

    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    wstd = 0.001
    lr_vanilla = 0.035
    lr_momentum = 0.005
    lr_rmsprop = 0.0006
    reg = 0.001

    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )

    wstd = 0.001
    lr = 0.003

    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    n_layers = 4
    hidden_dims = 32
    activation = "relu"
    out_activation = "none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.0001
    momentum = 0.9

    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


High Optimization Error refers to a model failure to fit the training data well. 
Indicates issues with the training process or model complexity being too low. <br>
We can observe 93.7% accuracy on the training set, which is a good indication of a low optimization error. <br><br>
High Generalization Error refers to a model that fits the training data well but fails on new, unseen data. 
Indicates overfitting or issues with data distribution. <br>
We can observe 92.4% accuracy on the validation set, which is a good indication of a low generalization error. <br><br>
High Approximation Error refers to model fundamental inability to capture the underlying data patterns.
Indicates the need for a more complex model or better features.
We also know that the datasets are not sampled from the same distribution, which can lead to high approximation error. <br>

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""