import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss


class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        size = (n_features, n_classes)
        self.weights = torch.normal(mean=0, std=weight_std, size=size)

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """
        y_pred, class_scores = None, None
        
        class_scores = torch.matmul(x, self.weights)
        y_pred = torch.argmax(class_scores, dim=1)

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """
        acc = None

        pred = y - y_pred
        count = len(torch.where(pred == 0)[0])
        acc = count / len(y)

        return acc * 100

        
    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            self.single_epoch(dl_train, loss_fn, train_res, learn_rate, weight_decay)
            self.single_epoch(dl_valid, loss_fn, valid_res, learn_rate, weight_decay)

        return train_res, valid_res

    def single_epoch(self, data_loader: DataLoader, loss_fn: ClassifierLoss,
                            result: namedtuple, learn_rate: float, weight_decay: float):

        x, y = next(iter(data_loader))
        y_pred, class_scores = self.predict(x)
        loss = loss_fn.loss(x, y, class_scores, y_pred)
        regularization = 0.5 * weight_decay * torch.sum(self.weights ** 2)
        loss += regularization
        accuracy = LinearClassifier.evaluate_accuracy(y, y_pred)
        gradient = loss_fn.grad() + weight_decay * self.weights
        self.weights -= learn_rate * gradient

        result.accuracy.append(accuracy)
        result.loss.append(loss)

        return result

    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """
        if has_bias:
            w_images = self.weights[1:]
        else:
            w_images = self.weights

        return w_images.reshape((self.n_classes,) + img_shape)


def hyperparams(weight_std=0.002, learn_rate=0.02, weight_decay=0.002):
    
    hp = dict(weight_std=weight_std, learn_rate=learn_rate, weight_decay=weight_decay)

    return hp
