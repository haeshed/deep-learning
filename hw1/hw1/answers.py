r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

student_name_1 = 'Yuval Tomer' # string
student_ID_1 = '207163783' # string
student_name_2 = 'Hadar Eshed' # string
student_ID_2 = '205947856' # string


# ==============
# Part 1 answers

part1_q1 = r"""
**1. False.**
    The test set is used to estimate the out-of-sample error. It is kept separately from the training set, and is used to 
    evaluate the model performance on unseen data.

**2. False.**
    The split should represent, approximately, the data distribution.
    For example, a certain split could include samples from only one class in the training set, while the data is
    distributed uniformly across multiple classes.

**3. True.**
    The test set is not used in any step of the model training. During cross validation, we use a fold from the training set
    for validation and adjustments of the hyperparameters of the model.

**4. False.**
    The generalization error is estimated and measured using the test set. The folds in cross validation are used for
    the selection of hyperparameters.
"""

part1_q2 = r"""
This approach is **wrong**.
the test set shouldn't be used at any point of the model training. Validating the performance of each model and
selecting the best hyperparameters should be done using a validation set (or methods like cross validation). The test
set should be mutually exclusive with the training set, and using it to train the model may lead to an overfit to the
test set.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

In the SVM loss function 𝐿(𝑊) as defined above, the parameter Δ represents the margin,
 which is a hyperparameter that controls the trade-off between the margin and the loss incurred by violating the margin. 
 The choice of Δ is arbitrary
   in the sense that it determines the threshold beyond which the model adds a loss for misclassification.<br>
    A larger Δ will result in a larger margin, which may lead to a simpler model that generalizes better to unseen data. <br>
    However, a larger margin may also result in a higher loss for misclassified samples, which may lead to a model that
    underfits the training data. <br>
    On the other hand, a smaller Δ will result in a smaller margin, which may lead to a
    more complex model that overfits the training data. <br> Therefore, the choice of Δ should be made based on the specific
    dataset and the desired trade-off between model complexity and generalization performance.


"""

part2_q2 = r"""
**Your answer:**

In a linear model for classification, such as the SVM model, 
the model learns a linear decision boundary using a set of weights over the given data. <br> 
I our case of handwritten digits, the computer receives pixel intensity values matrices (28x28 = 784), 
and needs to learn some relevant features from it. <br>
The model may learn patterns associated with different digits based on their pixel intensity distributions and patterns. <br><br>
The MNIST database should allow our model to learn that a digit 1 correspondes to a straight line, while the digit 8
corresponds to 2 circles in a vertical orientation. <br>This is very shallow learning, as it does not account for translation, line thickness etc.
(these could be better learned using a CNN for example). <br>This also means that if the database does not contain uniformally distributed (or close)
for  changes in the handwriting style, then the model could learn (for example)<br> that thick lines correspondes to a 7, 
which is obviously not something we want to acheive. <br><br>
According to the visualization section of our notebook, we can observe what are the feature weights for each digit.<br>
In our human eyes, these does not mean much, but we can try to infer, for example that  0 is closest to 9 (maybe because of the circle they share?) <br>
and that 3 is quite close to 5 (maybe because they both have a half circular leg and some top?). <br><br>
We can observe the errors of classification and see that in some cases, they are quite understandable. <br>
For example. if the handwritten 9 digit is not closed at the top of the circle, it surly resembles a 4. <br>
Or, if a 6 is written with a down extension of the circle, it resembles a 4. <br>
Other errors are more difficult to explain, this is due to the complex nature of the weights and the method.


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

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

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

# ==============

# ==============
