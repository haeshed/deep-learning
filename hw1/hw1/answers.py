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
The ideal pattern for a residual plot should be a random scattering of the points along the x-axis. This plot shows the 
differences between the real labels and the labels predicted by our model, so we dont want to see any patterns of error 
(otherwise, we should improve the model accordingly). If the distribution is non-linear, it may imply that there's a 
non-linear connection between the features and the value and linear regression model would not be a good fit.
Moreover, outliers in the plot may harm the model's accuracy and might need to be removed.
The fitness of the trained model improved, as the points are more randomly distributed across the x-axis.
"""

part3_q2 = r"""
**1.** This is not a linear regression model, because it results in a non-linear combination of the original features, 
although it may be a linear combination of the new features.

**2.** Yes. We can create a non-linear function of any feature or combination of features as an additional feature.

**3.** No, it will not be a hyperplane, rather a non-linear surface. The decision boundary is now defined by a
non-linear combination of the original features, that can capture non-linear relationships between the features and
classes.
"""

part3_q3 = r"""
**1.** We use ```np.logspace``` instead of ```np.linspace``` to cover a wide range of possible $\lambda$ values from
$10^{-3}$ to $10^{2}$. We examine both very small and large values for $\lambda$ to best adapt the regularization.

**2.** The model was trained with a 3-fold cross-validation, so the model is trained and evaluated 3 times, each time
using a different fold for validation. We repeat this process for each combination of degree and lambda, so without
considering the final fit on the entire training set, the model was fitted $3\cdot 3\cdot 20 = 180$ times.
"""

# ==============

# ==============
