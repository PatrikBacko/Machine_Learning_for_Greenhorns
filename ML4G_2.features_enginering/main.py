#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    train_data, test_data = sklearn.model_selection.train_test_split(dataset.data, test_size=args.test_size, random_state=args.seed)

    columns_ohe = []
    columns_scaler = []

    for i in range(train_data.shape[1]):
        bool = True
        data = train_data[:,i]
        for feature in data:
            bool = bool and feature.is_integer()
        if bool:
            columns_ohe.append(i)
        else:
            columns_scaler.append((i))

    pipe = sklearn.pipeline.Pipeline([  ("ColumnTransformer", sklearn.compose.ColumnTransformer([
                                        ("OneHotEncoder", sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), columns_ohe),
                                        ("Scaler", sklearn.preprocessing.StandardScaler(), columns_scaler)])),
                                        ("PolynomialFeatures", sklearn.preprocessing.PolynomialFeatures(2, include_bias=False).fit(train_data))])

    pipe.fit(train_data)
    train_data = pipe.transform(train_data)
    test_data = pipe.transform(test_data)

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))))