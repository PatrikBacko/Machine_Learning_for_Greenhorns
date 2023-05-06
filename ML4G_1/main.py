#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()

parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")


def main(args: argparse.Namespace) -> float:

    dataset = sklearn.datasets.load_diabetes()
    data = dataset.data

    data = np.concatenate([data, np.ones([data.shape[0], 1])], axis=1)
    data = sklearn.model_selection.train_test_split(data, dataset.target,
                                                    test_size=args.test_size, random_state=args.seed)
    train_data = data[0]
    test_data = data[1]
    train_targets = data[2]
    test_targets = data[3]

    train_data_transposed = train_data.transpose()
    train_data_inverted = np.linalg.inv(train_data_transposed @ train_data)

    weights = (train_data_inverted @ train_data_transposed) @ train_targets
    predictions = test_data @ weights
    rmse = 0

    for i in range(len(predictions)):
        rmse += (predictions[i] - test_targets[i])**2

    rmse = (rmse * 1/len(test_targets))**(1/2)

    return rmse

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
