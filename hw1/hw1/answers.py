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
