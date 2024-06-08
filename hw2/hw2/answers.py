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
**1.** Dropout is expected to result in worse accuracy on the training set, but higher accuracy on the test set
       (compared to a model without dropout).\
       Indeed, with `dropout=0` we obtained a 100% accuracy on the training set, and only 70% accuracy with
       `dropout=0.4` and only 20% accuracy with `dropout=0.8`.\
       The dropout prevents overfit - with `dropout=0` we got an overfit (test set accuracy of 22.5%, with 100% accuracy
       on the training set), while with `dropout=0.4` the accuracy on the test set was 27.5%.\
       
**2.** With high dropout we couldn't obtain a model that fits the data well enough, because of a too-small adaptation to
       the training data. With `dropout=0.4`, we could generate a model that fits the data well enough, while avoiding
       overfit.\
\
\
"""

part2_q2 = r"""
Yes, it is possible for the test loss to increase while the test accuracy also increases.\
Loss measures how well predicted probabilities match the true labels and is sensitive to prediction confidence, and
accuracy simply measures the percentage of correct predictions.\
During training, the model might make more confident predictions. If these predictions are correct, accuracy increases.
However, if the model becomes overly confident and makes even slight errors, the loss can increase despite improving
accuracy.\
For example, with `dropout=0`
"""

part2_q3 = r"""
**1.** Gradient descent is an optimization algorithm to minimize the loss function, while back-propagation computes the
       gradient of the loss function with respect to each weight by the chain rule.\
**2.** GD uses the entire dataset for each update, resulting in stable but slow convergence.\
       SGD updates weights using a single sample at a time, leading to faster but noisier convergence.\
**3.** Each epoch of SGD converges faster. Moreover, SGD brings noise, which can be used to escape local minima during
       training and reach the global minima. In addition, SGD uses less memory as it saves only one sample at a time,
       instead of the entire training dataset.\
**4.1.** Yes. Summing losses from disjoint batches before a backward pass results in a gradient equivalent to GD, as it 
         is identical to the sum of all gradients mathematically.\
         In GD, we sum $m$ gradients and divide by $m$. In the suggested approach, we sum $m_j$ gradients for each of
         the $j$ batches. For each batch we divide by $m_j$, and then we divide the sum by the number of batches $j$.\
**4.2.** Accumulating gradients and losses over all batches before backpropagation required more memory than available.
"""

part2_q4 = r"""
**1.1.** To reduce memory complexity in forward mode AD, we can store only the current value and its derivative during
         the computation. The memory complexity in this case is $\mathcal{O}(1)$.\
**1.2.** To reduce memory complexity in backward mode AD, we can store values at certain intervals and recompute
         intermediate values as needed. The memory complexity in this case is $\mathcal{O}(\sqrt n)$.\
**2.** Yes. For forward computation we use single storage for current values and derivatives. For backward computation
       we use single storage for each interval and recomputation.\
**3.** In deep architectures we can store only intervals of intermediate results and use less recomputing during
       backpropagation.
"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer

    n_layers = 4
    hidden_dims = 32
    activation = "relu"
    out_activation = "none"

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

    # raise NotImplementedError()
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.0001
    momentum = 0.9

    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
High Optimization Error refers to a model failure to fit the training data well. 
Indicates issues with the training process or model complexity being too low. <br>
We can observe`93.7%` accuracy on the training set, which is a good indication of a low optimization error. <br>
Altough, We have not thorouly tweaked the hyperparameters of the model, so we could potentialy get better results.<br><br>

High Generalization Error refers to a model that fits the training data well but fails on new, unseen data. 
Indicates overfitting or issues with data distribution. <br>
We can observe `92.4%` accuracy on the validation set, which is a good indication of a low generalization error. <br>
We can also compare the test and validation sets, which exemplify close results, thus we can say the 
generelization has mostly succeeded.<br><br>

High Approximation Error refers to model fundamental inability to capture the underlying data patterns/distribution.
Indicates the need for a more complex model or better features.
We also know that the datasets are not sampled from the same distribution, which can lead to high approximation error. <br>
Lastly, we can observe the boundry of the model, which is not perfect, but is fairly good, and can be improved by tweaking the hyperparameters.<br><br>

"""

part3_q2 = r"""
**Your answer:**

The noise in the generatad data has no tendency to affect FPR/FNR, as the noise is random and does not have a specific pattern. <br>
The other difference is that the rotation given to the data is not random, and has a specific pattern,<br> which can affect the model's performance. <br>

In our case, we can predict that the rotation will affect mostly the false classification of the `1 class` (orange) as `0 class`,<br>
 because of the structure of the data and the type of rotation. <br> Therefore, we can predict that the FNR (falsly classifying the `0 class`) will be higher than the FPR. <br>


"""

part3_q3 = r"""
**Your answer:**


The question presented to us is that of minimizing the risk of death while also minimizing financial costs. 
We have an opportunity to tweak the HP of the model, such that we control the tradeoff between the two risks. <br>
This really depends on the risks posed from the advanced diagnostics vs the risks posed from not testing at all (and relying on late diagnostic). <br>
With those considerations in mind, we can choose a threshold that will consider those risks and the costs associated with testing (and potentially costs of death). <br>
Without knowing the exact costs and risks, we can't give a definitive answer, so for now the optimal threshold will have to suffice. <br><br>

In the case of 1., if we can rely over non-lethal late symptoms (instead of actual testing),<br> then we can shift the threshold to a value which will lower the costs of testing (more false negatives->less advanced diagnostics). <br><br>
In the case of 2., where high probability of death is associated with late diagnostics,<br> we need shift the threshold to a value which will lower the risk of death (more false positives->more advanced diagnostics). <br>



"""

part3_q4 = r"""

#### Question 4
**Your answer:**

1. When analyzing by fixed depth (analyzing the graphs vertically), we can see that the model with depth=1 has a simple decision boundary, which is not able to capture the complexity of the data. <br>
The boundary for the model with depth=1 is almost a straight line, for each of the 3 widths. <br>
We can also observe that for each increase in width, the decision boundary stays in the same general shape, but becomes more complex and fine-grained (as if it gains resolution), and the model is able to separate the data better.
This also means that the model tends to overfit the data, as the decision boundary becomes more complex, which also manifests in slightly lower test accuracy results. <br>

2. We can see when analyzing by fixed width (analyzing the graphs horizontally), the whole shape of the boundary changes (as if we increased a polynomial function's degree). <br>
We can see that the model with `depth=1` is quite a straight line, depth=2 becomes almost a Z shape (2 parallels and 1 connecting between) and for depth=8 we get much more curvature. <br>

3. Between `depth=1, width=32` and `depth=4, width=8` we can see complete change. <br> 
Although the number of parameters is the same, the model with depth=4 and `width=8` has a much more complex decision boundary, which is able to capture the complexity of the data. <br>
The model with `depth=1 and width=32` has a simple decision boundary, almost a straight line (which can only classify well linearly seprable date). <br>
This is because the model with `depth=4 and width=8` has more layers, which allows for more complex decision boundaries. <br>
This also makes sense as a model with 1 layer is practically a linear model which cannot classify complex datasets well. <br>

4. The threshold selection is supposed to improve the results on the test set, as it is chosen to maximize the accuracy on the validation set. <br>
The role of the validation set is to help us choose the best hyperparameters and threshold, which will generalize well to the test set. <br>
Generally, the threshold did improve the results on the test set, as the test accuracy was higher than the validation accuracy. <br>
In the few cases where the test accuracy was lower than the validation accuracy, it can be attributed to the fact that the validation set was not sampled from the same distribution as the test set. <br>
  

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
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    weight_decay = 0.0001
    momentum = 0.9
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