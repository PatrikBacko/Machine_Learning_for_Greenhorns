#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt


import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.compose

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.
    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features
    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        

        train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
														train.data, train.target,
														test_size=0.20, random_state=args.seed)
                                            

        pipeline = sklearn.pipeline.Pipeline([("ColumnTransformer", sklearn.compose.ColumnTransformer([
                            ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"),  list(range(0,15))),
                            ("Scaler", sklearn.preprocessing.StandardScaler(), list(range(15,21)))])),
                            ("poly",sklearn.preprocessing.PolynomialFeatures()),
                            ("lr",sklearn.linear_model.LogisticRegression(random_state=args.seed))])
          
        grid = {"poly__degree":[2,3], "lr__C":[0.001,0.1,1,10,1000,10000], "lr__solver":["lbfgs","sag"], "lr__penalty":["none","l2"], "lr__tol":[0.0001,  0.001, 0.00001]}
        

        model = sklearn.model_selection.GridSearchCV(pipeline, grid, cv=sklearn.model_selection.StratifiedKFold(5), refit=True)
        model.fit(train.data, train.target)

        # print(model.score(test_data, test_target))
        

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)