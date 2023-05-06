#!/usr/bin/env python3
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx IDs anywhere
# in a comment block in the source file (on a line beginning with `#`).
#
# You can find out ReCodEx ID in the URL bar after navigating
# to your User profile page. The ID has the following format:
# 6f539690-213a-11ec-986f-f39926f24a9c
# 6a0970ad-213a-11ec-986f-f39926f24a9c
# 6eea65f3-213a-11ec-986f-f39926f24a9c

import argparse
import lzma
import pickle
import os
import urllib.request
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt
import sklearn.feature_extraction.text
import sklearn.svm
import sklearn.naive_bayes
import sklearn.model_selection
import sklearn.metrics
import sklearn.ensemble
import sklearn.linear_model


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

class Model:
    def fit(self, train_data, train_target):
        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,5))
        #self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(lowercase=False, ngram_range=(1,10))
        train_data = self.vectorizer.fit_transform(train_data)
        #self.classifier = sklearn.svm.SVC()
        #self.classifier = sklearn.naive_bayes.BernoulliNB()
        self.classifier = sklearn.naive_bayes.MultinomialNB()
        #self.classifier = sklearn.linear_model.LogisticRegression()
        self.classifier.fit(train_data, train_target)

    def predict(self, test_data):
        test_data = self.vectorizer.transform(test_data)
        prediction = self.classifier.predict(test_data)
        return prediction


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        f1_score = 0
        
        kf = sklearn.model_selection.StratifiedKFold()
        for train_index, test_index in kf.split(train.data, train.target):
            train_data = [train.data[i] for i in train_index]
            test_data = [train.data[i] for i in test_index]
            train_target = [train.target[i] for i in train_index]
            test_target = [train.target[i] for i in test_index]

            model = Model()
            model.fit(train_data, train_target)

            predictions = model.predict(test_data)
            score = sklearn.metrics.f1_score(test_target, predictions)
            print(score*100)
            f1_score += score

        print(f"final score:{f1_score/5 * 100}")

        model = Model()
        model.fit(train.data, train.target)
        

        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)