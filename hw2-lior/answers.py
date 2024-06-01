r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

student_name_1 = 'Leeor Rofeim' # string
student_ID_1 = '212356398' # string
student_name_2 = 'Omri Pinhas' # string
student_ID_2 = '201313061' # string

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
##1##
1.1. $\frac{\partial \mathbf{Y}}{\partial \mathbf{X}} = (\text{N, out\_features, N, in\_features}) = (64, 512, 64, 1024)$

1.2. Yes, this Jacobian tensor is highly sparse.
   Each output element $Y_{ij}$ is only influenced by the corresponding input row $X_{i*}$,
   meaning $\dfrac{\partial Y_{i,j}}{\partial X_{k,l}}$ is non-zero only when $k = i$.
   $Y_{i*}$ depend only on $X_{i*}$ thus if $k \neq i$ $X_{kl}$ dont change $Y_{ij}$ meaning the partial derivative is 0.

1.3. no we dont have to materialize the above Jacobian.
For a fully-connected (linear) layer, the relationship between the input $\mathbf{X}$
and the output $\mathbf{Y}$ is given by:
$\mathbf{Y} =  \mathbf{W} \mathbf{X} + \mathbf{b}$
For a single sample i, the output $\mathbf{Y}_i$ for that sample is:
$\mathbf{Y}_i = \mathbf{W} \mathbf{X}_i + \mathbf{b}$

This means that each element of the output $\mathbf{Y}_i$ is linearly dependent 
on the corresponding row of the weight matrix $\mathbf{W}$. 
Therefore, the Jacobian matrix $\frac{\partial \mathbf{Y}}{\partial \mathbf{X}}$ 
contains the weight matrix $\mathbf{W}$ as its non-zero elements.
thus we can use the chain rule and multiply the weight matrix with $\delta Y$ to get:


$\delta X = \frac{\partial X}{\partial L} = \frac{\partial X}{\partial Y} \frac{\partial Y}{\partial L} = \frac{\partial X}{\partial Y} \delta Y = \delta \mathbf{Y} \mathbf{W}^\top$



##2##


**2.1.** the shape of the weight tensor W us (512,1024) and the shape of the output tensor Y is (64,512). thus we get $\frac{\partial \mathbf{Y}}{\partial \mathbf{W}} = (64, 512, 512, 1024)$

**2.2.** Yes, this Jacobian tensor is highly sparse.
   Each output element $Y_{ij}$ is only influenced by the corresponding input row $W_{j*}$ (the weights that connected to it),
   meaning $\dfrac{\partial Y_{i,j}}{\partial W_{k,l}}$ is non-zero only when $j = k$.
   $Y_{*j}$ depend only on $W_{j*}$ thus if $k \neq j$ , $W_{kl}$ doesnt change $Y_{ij}$ meaning the partial derivative is 0.
   the number of zero entrys is determine by the ratio between the in_features and out_features.

   In our case `in_features=1024` and `out_features=512`. assume N=1 - fully connected meaning we have 1024*512 = 524288 weights. each out_feature is only depand on
   1024 weights so all 524288 - 1024 other entrys in the Jacobian tensor for this sample is 0. Hence, most elements of the Jacobian tensor $\frac{\partial \mathbf{Y}}{\partial \mathbf{W}}$ are zero by defeniotion 



2.3. we again don't need to materialize the Jacobian. we can again use the chain rule and use the fact that for each sample, the non zero elements of 
 
$\frac{\partial \mathbf{L}}{\partial \mathbf{Y}}$ are the inputs of the corresponding sample. we can therefore multiply the input tensor with the scalar loss $\delta \mathbf{Y}$ 
so we get 
$\delta \mathbf{W} = \mathbf{X}^\top \delta \mathbf{Y}$

"""

part1_q2 = r"""
**Your answer:**
Training neural networks with descent-based optimization requires calculation of
the gradient in order to know the direction of the step that will bring the
loss function to its minimum. The only alternative to back-propagation would be
to manually calculate the derivatives of each and every function with respect
to every parameter. This is impractical for larger networks, and would
potentially require repeated calculations following changes in the netowrk's
structure. Therefore, while theoretically it is possible to manually calculate
the gradient without using back-propagation, in practice it is infeasible and
back-propagation is **required** in order to train neural networks when using
descent-based optimization.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.1, 0.1, 0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.001  
    lr_vanilla = 0.035     
    lr_momentum = 0.002      
    lr_rmsprop = 0.0005        
    reg = 0.001
   

    # ========================
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
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd= 0.001
    lr =0.003
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**
1. expectations : we expect that without dropout the model
will perform better on the training set then the model with
dropout. on the test set we expect that the model with dropout
will preform better then the model without dropout
for dropout = 0 we where asked to choose parameters that will overfit the data and that
is exactly what we got. we can clearly see that with dropout the model performs better on the test set
overfits less than the one with dropout = 0

as we can clearly see the results match the expectations.
 
dropout improves model performance on test sets by preventing
overfitting through introducing noise promoting and neuron independence.
which prevents neurons from co-adapting too much and make the model more general.

2. too high dropout may restrict the model too much resulting in a model which is
too simple for the data. dropout = 0.8 means 80% of the neurons are dropped during 
traning and as we can see in the graph the model struggling to learn the training set data 
underlying patterns (accurecy = 37.0%). and low accurecy on the test set (26.5%) compair to more moderate 
dropout (dropout = 0.4)

on the other hand dropout = 0.4 - only 40% of the neurons are dropped during
training. this enables the model to capture more complex patterns and learn better. thus we get way better
accuracy on the traning set and test set (75.2% , 30.1%)
"""

part2_q2 = r"""
**Your answer:**
Yes it is possible.
Cross-Entropy Loss - Measures the difference between the predicted
probability distribution and the actual distribution (labels).

Accuracy - Measures the percentage of correct predictions.

it may happen when the model starts making more correct predictions (increasing accuracy) 
but does so with lower confidence (or makes some high-confidence errors) 
leading to a temporary increase in the cross-entropy loss.

for example from the dropout implimentation test (Training with dropout= 0) we get those 2 lines

test_batch (Avg. Loss 2.314, Accuracy 21.4):
test_batch (Avg. Loss 2.337, Accuracy 22.1):

as we can see both loss and accurecy increased from epoch 2 to 3


"""

part2_q3 = r"""
**Your answer:**

1. gradient descent is a method used to solve optimization problem using gradients 
Backpropagation is not solving the optimization problem but calculates these gradients efficiently 
using the chain rule.

2. GD uses the entire training dataset to compute the gradient of the cost function for each update 
SGD uses only one training example (or a batch) to do so.
thus GD requires more computation for each update (aslo more memory)
SGD will probably converge faster but may have more fluctuations in the loss function.
since SGD pick diffrent example each update step it may act as a form of regularization - 
and may prevent overfitting. (in addition SGD may help escape local minima)

3. as we mentioned above
a. SGD may act as a form of regularization - and may prevent overfitting. 
b. SGD may help escape local minima
c. lower computation coast and memory usage which is importent in DL as the data sets van get big
thus it is common to use mini batch SDG

4. A.
Yes this approach is called gradient accumulation it produce a gradient equivalent to GD
lets look at the update step of the weights
this is GD step formula going over all samples at once
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{m} \sum_{i=1}^{m} \nabla_{\mathbf{w}} J(\mathbf{w}_t, \mathbf{x}^{(i)}, y^{(i)})
$$

and this is gradient accumulation step formula calculating batch by batch 

$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \frac{1}{k} \sum_{j=1}^{k} \frac{1}{m_j} \sum_{i=1}^{m_j} \nabla_{\mathbf{w}} J(\mathbf{w}_t, \mathbf{x}^{(i,j)}, y^{(i,j)})
$$

as k is the number of batches.
we can clearly see the 2 formulas are mathematically identical.

B. maybe there is enough memory to load the bathc itself but the
accumulation of intermediate gradients and activations during the forward passes
cause the out of memory error.
"""

part2_q4 = r"""
**Your answer:**

Compute the value and the derivative simultaneously by maintaining a dual number $(u_i, v_i)$ where $u_i$ is the value of the function and $v_i$ is the derivative.

For each function $f_i$, update $(u_i, v_i)$ as follows:

$u_{i+1} = f_i(u_i)$

$v_{i+1} = f_i'(u_i)v_i$

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
    n_layers = 5
    hidden_dims = 32
    activation = "relu"
    out_activation = "logsoftmax"
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
    loss_fn = torch.nn.NLLLoss()
    lr = 0.01
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. High Optimization error - A high optimization error means that the model has
not been trained well enough on the training data (for example due to too short
training time). This would manifest in the form of high training loss and low
training accuracy. We can see the training accuracy is very high at the
end of the training process, however, we do not know whether it had reached a
maximum point, or further training could improve it. Additionally we can see
the loss graph looking fairly noisy (although the general trend is downwards).
This leads us to believe that the model could still be improved by further
training, and therefore it has high optimization error (we would try to train
it further).

2. High Generalization error - A high generalization error means that the model
does not generalize well to unseen data, which is usually visible by the model's
performance on the training set and the validation / test set being very 
different (high accuracy on training with low accuracy on validation / test).
In effect it means the model has overfit the training data.
In our case we can see that the performance on both the training and validation
sets is relatively similar (with the validation set having slightly lower 
accuracy as expected). This means the model has generalized well to dealing with
unseen data, and there's no high generalization error.

3. High Approximation error - A high approximation error means that the model
is unable to capture the true underlying distribution of the data, which usually
happens when a model is too simple for the task.
We can look at the decision boundary plot to get an idea of whether the model
has high approximation error (if the boundary is too simple).
Looking at the decision boundary plot we can see that the model managed to
capture relatively well the crescent shapes of the underlying distribution as
evident from the boundaries it separated along. This means that the model does
not have a high approximation error - it is complex enough to correctly identify
the patterns in the data.
"""

part3_q2 = r"""
**Your answer:**
Looking at the way the data for the validation set was generated, and how it
compares to the data in the training set on which the model was trained, we can
see two key differences:
1. The data in the validation set is noisier than the data in the training set.
2. The data in the validation set is rotated compared to the data in the
training set.

Since the noise from point 1 is evenly distributed between the two classes we
can surmise that it alone will not make much difference on the FPR vs FNR, so
we can ignore it.

Regarding the rotation, we estimate that it would result in a higher FNR on the
validation set compared to the FPR.
The reason is that the decision boundaries the model learned for classifying the
training set were calculated according to the distribution of the data in the
training set. Since the validation set is rotated compared to the training set,
part of class 1 in the validation set falls in an area which was previously
populated by class 0 samples in the training set, and therefore the model
has learned a decision boundary that would mistakenly classify those samples as
class 0 (there is no symmetry here with the opposite mistake due to the shape of
the classes and the degree of rotation). Therefore we anticipate some of the
class 1 samples to be labeled as class 0 by the model, which leads to a higher 
FNR. We can see this clearly in the decision boundary plot and the confusion
matrix of the validation set as well.
"""

part3_q3 = r"""
**Your answer:**
1. In the first scenario, failure to diagnose the patient with the disease does
not incur a very high cost, neither in loss of life, time or money. On the other
hand, mistakenly diagnosing the disease where it is not present will endanger
the patient with high-risk tests that will additionally cost a lot of money.
In this case, we can say that the cost of a mistaken positive diagnosis is
greater than the cost of mistakenly failing to diagnose it, that is, the cost of
a false positive is larger than that of a false negative. In such a case, we
should strive to decrease the false positives as much as possible, even at the
risk of increasing the number of false negatives. Therfore our optimal point
this time will not be the same as our previous choice, because a mistake has
a different cost depending on its direction.
We would opt for a point on the far left side of the curve, where both the FPR
and TPR are low (and by extension the FNR is high). This would prevent us from
unnecessarily over-testing patients and risking their lives (and wasting our
money) where it is not strictly required.

2. In the second scenario, the disease is much more deadly leading to a complete
shift in our priorities. This time, not diagnosing a patient in time means that
the risk to the patient's life is very high, so we should optimize for as low
FNR as possible in order to not give wrong negative diagnoses. Therefore we will
once again choose a threshold which is different than our original optimal
point, such that false positives are preferred over false negatives. This means
our threshold will be more to the right on the curve compared to our original
threshold, which implies higher false positive rate, but also a lower false
negative rate (how far to the right would depend on when the disadvantages of
over-testing would outweigh the risks of not enough testing).

Overall we can see that the choice of threshold is very task-specific and the
optimal threshold depends on the cost of making different kinds of mistakes.
"""


part3_q4 = r"""
**Your answer:**
1. Fixed depth, variable width: We can see that for a fixed depth (fixed
number of layers), changing the width (number of neurons per layer) changes the
form of the resulting decision boundary (and by extension, the performance of
the model). Specifically, increasing the width of the layer enables the network
to learn a more complex decision boundary, which, to an extent, improves the
performance of the model. However, increasing the width too much might create
too many parameters which would cause the model to overfit. We can see this in
the decision boundary of the widest network, where the learned decision boundary
creates unnecessary curves in places where a simple straight line might've been
enough.

2. Fixed width, variable depth: Similarly to increasing the width, increasing the
depth allows the model to learn more complex patterns in the data and create
more accurate decision boundaries as well. A deeper network enables the boundary
to learn more abstract high level features, so the boundary is able to correctly
capture more complex patterns. However, we once again risk overfitting the model
to the training data if the model is too deep, resulting in the boundary curving
around noise. Too much depth might also lead to problems with training the
network such as vanishing gradients.

3. Comparing the two configurations, we can see that while they indeed have the
same number of parameters, one is a shallow but wide network, while the other is
a deep but narrow one. A wide, shallow network can model complex interactions
in the first layer directly, but lacks the hierarchical feature extraction
which deeper networks provide. Therefore the features it learns will be less
abstract, so while the decision boundary is still complex, we can see it is
"less smooth" compared to the deeper network, whose depth enables it to learn
more abstract features which are higher level. Those high level features enable
the creation of a smoother decision boundary that generalizes better. While in
this simple toy example the difference might not be immediately apparent, in
more complex tasks which require the network to learn more abstract features
the advantage of the deeper network would become apparent.

4. Threshold selection using the validation set improves the performance on the
test set in this case. The reason is that the threshold is specifically selected
to optimize some metric (for example accuracy) on the validation set and give
the best results on it. Provided that the test set came from a similar
distribution as the validation set, the threshold chosen for the validation
set should generalize well to boost the performance on the test set as well,
since we expect the data to behave similarly. In our case, while the validation
set and test set do not come from the exact same distribution (the validation
set is rotated a little and is noisier), in practice the distributions are close
enough to each other such that the threshold will generalize well to the test
set. Therefore using the optimal threshold obtained from the validation set
improves the results on the test set.
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
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

regular block:

Each 3x3 convolution has a kernel size of 3x3 and operates on all 256 input channels 
to produce 256 output channels.
 The number of parameters for each convolution is
(3*3*256)*256+256 = 3,147,008
ew have 2 convolutions  thus overall
3,147,008*2 = 6,294,016

Bottleneck Block:

First 1x1 convolution to reduce the dimensionality from 256 to 64 channels
(1*1*256)*64 + 64 = 16,448

3x3 convolution on 64 channels:
(3*3*64)*64+64 = 36,928

Second 1x1 convolution to expand the dimensionality back to 256 channels
(1*1*64)*256 + 256 = 16,640

total = 16,448+36,928+16,640 = 70,016




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