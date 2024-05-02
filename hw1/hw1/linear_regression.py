import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        y_pred = None
        y_pred = np.matmul(X, self.weights_)

        return y_pred

    def fit(self, X, y):
        ########################################################################
        ########################### NEED TO FIX!!!!! ###########################
        ########################################################################
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        w_opt = None

        regularization_matrix = self.reg_lambda * np.eye(X.shape[1]) * X.shape[0]
        regularization_matrix[0][0] = 0     # do not apply regularization on bias

        X_transpose = np.transpose(X)
        pinv = np.linalg.inv(np.matmul(X_transpose, X) + regularization_matrix)

        w_opt = np.matmul(np.matmul(pinv, X_transpose), y)

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    if feature_names is None:
        x = df.drop(target_name, axis=1)
    else:
        x = df[[feature for feature in feature_names if feature is not target_name]]

    y_pred = model.fit_predict(x, df[target_name])

    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        xb = None

        ones = np.ones((X.shape[0], 1))
        xb = np.hstack((ones, X))

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree
        self.polynomial_features = PolynomialFeatures(degree=self.degree)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        X_transformed = None

        X_transformed = np.delete(X, 3, axis=1)
        X_transformed[:, 0] = np.log(X_transformed[:, 0])
        X_transformed[:, 12] = np.log(X_transformed[:, 12])

        if self.degree > 1:
            X_transformed = self.polynomial_features.fit_transform(X_transformed)

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """
    if n > len(df.columns):
        n = len(df.columns)

    correlations = df.corr()[target_feature].drop(target_feature)
    top_n_features = correlations.abs().nlargest(n).index
    top_n_corr = correlations[top_n_features]

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """
    mse = np.square(y - y_pred).mean()
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """
    numerator = np.sum(np.square(y - y_pred))
    denominator = np.sum(np.square(y - y.mean()))

    if denominator != 0:
        r2 = 1 - numerator / denominator
    else:
        r2 = 0

    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    ########################################################################
    ########################### NEED TO FIX!!!!! ###########################
    ########################################################################
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    #param_grid = {'linearregressor__reg_lambda': lambda_range, 'bostonfeaturestransformer__degree': degree_range}

    # Get all model hyperparameters
    all_params = model.get_params()

    # Create a parameter grid to search (include lambda and degree)
    param_grid = {
        "linearregressor__reg_lambda": lambda_range,
        "bostonfeaturestransformer__degree": degree_range,
    }

    # Add other relevant hyperparameters to param_grid
    for param_name, param in all_params.items():
        if param_name not in param_grid and hasattr(param, "values"):
            # Only consider hyperparameters with defined search space
            param_grid[param_name] = param.values

    # Use GridSearchCV for k-fold cross-validation
    grid_search = sklearn.model_selection.GridSearchCV(model, param_grid, cv=k_folds, scoring="neg_mean_squared_error")
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    # ========================

    return best_params
