from abc import ABC, abstractmethod

import numpy as np
import sklearn.metrics as skm
import utils.evaluation_metrics as em
from sklearn.linear_model import LogisticRegressionCV
from utils.logger import logger


class EnsembleMethod(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def __call__(self, in_score: np.ndarray, out_score: np.ndarray, *args, **kwargs):
        ...


class MeanScore(EnsembleMethod):
    def __init__(self, ignore_dim=0) -> None:
        self.ignore_dim = ignore_dim
        self.__name__ = "mean_score"

    def __call__(self, in_score: np.ndarray, out_score: np.ndarray, *args, **kwargs):
        return np.mean(in_score[:, self.ignore_dim :], 1), np.mean(
            out_score[:, self.ignore_dim :], 1
        )


class WeightRegression(EnsembleMethod):
    def __init__(
        self,
        verbose=False,
        ignore_dim=0,
        split_size=1000,
        regressor=LogisticRegressionCV,
        recall_level=0.95,
    ):

        self.verbose = verbose
        self.split_size = split_size
        self.regressor = regressor(
            n_jobs=-1,
            scoring=skm.make_scorer(self.scoring_obj, greater_is_better=False),
        )
        self.__name__ = f"weight_regression"
        self.ignore_dim = ignore_dim
        self.recall_level = recall_level

    def __call__(self, in_data: np.ndarray, out_data: np.ndarray, *args, **kwargs):
        in_data = in_data[:, self.ignore_dim :]
        out_data = out_data[:, self.ignore_dim :]

        # split
        x_train, y_train, x_test, y_test = self.split_data(in_data, out_data)
        # fit
        self.regressor.fit(x_train, y_train)
        logger.info(f"Coeficients: {self.regressor.coef_}")

        # predict
        y_pred_test = self.regressor.predict_proba(x_test)[:, 1]
        # evaluate
        if self.verbose:
            print(self.regressor.coef_)
            y_pred_train = self.regressor.predict_proba(x_train)[:, 1]
            logger.info(
                "training fpr: {:.4f}".format(self.scoring_obj(y_train, y_pred_train)),
            )
            logger.info(
                "test fpr: {:.4f}".format(self.scoring_obj(y_test, y_pred_test)),
            )
        in_scores = y_pred_test[y_test == 1]
        out_scores = y_pred_test[y_test == 0]

        return in_scores, out_scores

    def split_data(self, data_in, data_out):
        # Train
        if len(data_in.shape) == 1:
            data_in = data_in.reshape(-1, 1)
            data_out = data_out.reshape(-1, 1)

        x1_train = data_in[: self.split_size]
        x2_train = data_out[: self.split_size]
        x_train = np.concatenate((x1_train, x2_train))
        y_train = np.concatenate(
            (np.ones((len(x1_train), 1)), np.zeros((len(x2_train), 1)))
        ).reshape(-1)

        # Test
        x1_test = data_in[self.split_size :]
        x2_test = data_out[self.split_size :]
        x_test = np.concatenate((x1_test, x2_test))
        y_test = np.concatenate(
            (np.ones((len(x1_test), 1)), np.zeros((len(x2_test), 1)))
        ).reshape(-1)
        return x_train, y_train, x_test, y_test

    def scoring_obj(self, y_true, y_pred):
        fprs, tprs, thresholds = skm.roc_curve(y_true, y_pred, pos_label=1)
        return em.compute_fpr_tpr(tprs, fprs, self.recall_level)


class AdvWeightRegression(WeightRegression):
    def __init__(
        self,
        verbose=False,
        ignore_dim=0,
        split_size=500,
        regressor=LogisticRegressionCV,
        recall_level=0.95,
        coef_=None,
    ) -> None:
        super().__init__(verbose, ignore_dim, split_size, regressor, recall_level)
        self.__name__ = "adv_weight_regression"
        self.coef_ = coef_

    def __call__(self, in_data: np.ndarray, out_data: np.ndarray, adv_data: np.ndarray):
        # split
        x_train, y_train = self.split_data(
            in_data[:, self.ignore_dim :],
            adv_data[:, self.ignore_dim :],
        )
        # fit
        self.regressor.fit(x_train, y_train)
        if self.coef_ is not None:
            self.regressor.coef_ = self.coef_
        logger.info(f"Coeficients: {self.regressor.coef_}")

        # eval
        if self.verbose:
            print(self.regressor.coef_)
            y_pred_train = self.regressor.predict_proba(x_train)[:, 1]
            print(
                "training fpr: {:.4f}".format(self.scoring_obj(y_train, y_pred_train)),
            )

        # predict
        in_scores = self.regressor.predict_proba(
            in_data[self.split_size :, self.ignore_dim :]
        )[:, 1]
        out_scores = self.regressor.predict_proba(out_data[:, self.ignore_dim :])[:, 1]
        return in_scores, out_scores

    def split_data(self, data_in, data_out):
        # Train
        if len(data_in.shape) == 1:
            data_in = data_in.reshape(-1, 1)
            data_out = data_out.reshape(-1, 1)

        x1_train = data_in[: self.split_size]
        x2_train = data_out
        x_train = np.concatenate((x1_train, x2_train))
        y_train = np.concatenate(
            (np.ones((len(x1_train), 1)), np.zeros((len(x2_train), 1)))
        ).reshape(-1)

        return x_train, y_train
