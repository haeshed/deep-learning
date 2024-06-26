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
The SVM loss function 𝐿(𝑊) as defined above, contains the parameter Δ that represents the regularization,<br>
 which is a hyperparameter that controls the loss incurred by violating the margin. 
 The choice of Δ is arbitrary in the sense that any positive number would attain this goal. <br>
 The regularization term affects the difference in scores and not the absolute scores, therefore it affects the scale of weights
 (the model is encouraged to have a smaller weights) <br> and not the hyperplane itself. <br>
"""

part2_q2 = r"""
In a linear model for classification, such as the SVM model, 
the model learns a linear decision boundary using a set of weights over the given data. <br> 
I our case of handwritten digits, the computer receives pixel intensity values matrices (28x28 = 784), 
and needs to learn some relevant features from it. <br>
The model may learn patterns associated with different digits based on their pixel intensity distributions and patterns. <br><br>
The MNIST database should allow our model to learn that a digit 1 correspondes to a straight line, while the digit 8
corresponds to 2 circles in a vertical orientation. <br>This is very shallow learning, as it does not account for translation,
line thickness etc. (these could be better learned using a CNN for example). <br>
This also means that if the database does not contain uniformally distributed (or close)
for changes in the handwriting style, then the model could learn (for example)<br> that thick lines correspondes to a 7, 
which is obviously not something we want to acheive. <br><br>
According to the visualization section of our notebook, we can observe what are the feature weights for each digit.<br>
In our human eyes, these does not mean much, but we can try to infer, for example that 0 is closest to 9
(maybe because of the circle they share?) <br>
and that 3 is quite close to 5 (maybe because they both have a half circular leg and some top?). <br><br>
We can observe the errors of classification and see that in some cases, they are quite understandable. <br>
For example. if the handwritten 9 digit is not closed at the top of the circle, it surly resembles a 4. <br>
Or, if a 6 is written with a down extension of the circle, it resembles a 4. <br>
Other errors are more difficult to explain, this is due to the complex nature of the weights and the method.
"""

part2_q3 = r"""
We think that the current learning rate is a good choice. <br>
A learning rate that is too small may result in a slow convergence of the model, as the model parameters are updated
by a small amount in each iteration. <br>
On the other hand, a learning rate that is too large may result in the model overshooting the minimum of the loss function
and diverging. <br>
The chocen learning rate of 0.02 is a small value, which allows the model to converge to a minimum of the loss function
without overshooting it. <br> <br>
A graph with a good learning rate choice should have these attributes: <br>
At the beginning of training, the loss rapidly decreases as the model starts to learn from the training data.<br>
After the initial rapid decrease, the loss continues to decrease steadily over subsequent iterations and converge to some
low minimum.<br>
Also, The loss function graph shows a smooth curve with no large fluctuations or spikes. <br>
Also, the accuracy graph should show a steady increase over the training iterations. <br>
Our solution attains all these attributes, therefore we believe it is a well solution. <br> 
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
**1.** This is not a linear regression model, rather a polynomial regression model, because it results in a non-linear
combination of the original features (although it may be a linear combination of the new features).

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
