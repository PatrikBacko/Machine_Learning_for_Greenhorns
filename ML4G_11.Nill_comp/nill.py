#!/usr/bin/env python3

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
import sklearn.decomposition

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)

levels = ["P1","P2","P3","P4","P5","P6","P7","P8"]

class Model:
    classifiers =  [sklearn.svm.SVC(verbose=True), sklearn.svm.SVC(verbose=True), sklearn.svm.SVC(verbose=True), sklearn.svm.SVC(verbose=True),
                    sklearn.svm.SVC(verbose=True), sklearn.svm.SVC(verbose=True), sklearn.svm.SVC(verbose=True), sklearn.svm.SVC(verbose=True)]
    vectorizers = [sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3)), sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3)),
                    sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3)),sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3)),
                    sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3)), sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3)),
                    sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3)),sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3))]

    def fit(self, train_data, train_target, train_prompts, train_levels):
        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False, ngram_range=(1,3), min_df=11)
        #self.vectorizer = sklearn.feature_extraction.text.CountVectorizer(lowercase=False, ngram_range=(1,10))
        
        self.classifier = sklearn.linear_model.SGDClassifier()
        train_data = self.vectorizer.fit_transform(train_data)
        #self.classifier = sklearn.svm.SVC(verbose=True)
        #self.classifier = sklearn.naive_bayes.BernoulliNB()
        #self.classifier = sklearn.naive_bayes.MultinomialNB()
        #self.classifier = sklearn.linear_model.LogisticRegression()

        # train_prompts = np.array(train_prompts, np.str)
        # train_data = np.array(train_data, np.str)

        # indexes =  [np.where(train_prompts == 'P1'),
        #             np.where(train_prompts == 'P2'),
        #             np.where(train_prompts == 'P3'),
        #             np.where(train_prompts == 'P4'),
        #             np.where(train_prompts == 'P5'),
        #             np.where(train_prompts == 'P6'),
        #             np.where(train_prompts == 'P7'),
        #             np.where(train_prompts == 'P8')]

        # for i in range(8):
        #     data = self.vectorizers[i].fit_transform(train_data[indexes[i][0]])
        #     self.classifiers[i].fit(data, train_target[indexes[i][0]])

        self.classifier.fit(train_data, train_target)

    def predict(self, test_data, test_prompts, test_levels):
        test_data = self.vectorizer.transform(test_data)
        prediction = self.classifier.predict(test_data)
        return prediction

        # indexes =  [np.where(test_prompts == 'P1'),
        #             np.where(test_prompts == 'P2'),
        #             np.where(test_prompts == 'P3'),
        #             np.where(test_prompts == 'P4'),
        #             np.where(test_prompts == 'P5'),
        #             np.where(test_prompts == 'P6'),
        #             np.where(test_prompts == 'P7'),
        #             np.where(test_prompts == 'P8')]

        # test_data = np.array(test_data, np.str)

        # for i in range(8):
        #     test_data[indexes[i,0]] = self.vectorizers[i].fit_transform(test_data[indexes[i][0]])

        # # test_data = self.vectorizer.transform(test_data)
        # for i in range(test_data.shape[0]):
        #     if test_prompts[i] == "P1": prediction = self.classifiers[0].predict(test_data[i])
        #     if test_prompts[i] == "P2": prediction = self.classifiers[1].predict(test_data[i])
        #     if test_prompts[i] == "P3": prediction = self.classifiers[2].predict(test_data[i])
        #     if test_prompts[i] == "P4": prediction = self.classifiers[3].predict(test_data[i])
        #     if test_prompts[i] == "P5": prediction = self.classifiers[4].predict(test_data[i])
        #     if test_prompts[i] == "P6": prediction = self.classifiers[5].predict(test_data[i])
        #     if test_prompts[i] == "P7": prediction = self.classifiers[6].predict(test_data[i])
        #     if test_prompts[i] == "P8": prediction = self.classifiers[7].predict(test_data[i])
        #     predictions.append(prediction[0])

       

        # for i in range(test_data.shape[0]):
        #     if test_prompts[i] == "P1":
        #         test_dato = self.vectorizers[0].transform(test_data[i])
        #         prediction = self.classifiers[0].predict(test_dato)

        #     if test_prompts[i] == "P2":
        #         test_dato = self.vectorizers[1].transform(test_data[i])
        #         prediction = self.classifiers[1].predict(test_dato)

        #     if test_prompts[i] == "P3":
        #         test_dato = self.vectorizers[2].transform(test_data[i])
        #         prediction = self.classifiers[2].predict(test_dato)
        #     if test_prompts[i] == "P4":
        #         test_dato = self.vectorizers[3].transform(test_data[i])
        #         prediction = self.classifiers[3].predict(test_dato)
        #     if test_prompts[i] == "P5":
        #         test_dato = self.vectorizers[4].transform(test_data[i])
        #         prediction = self.classifiers[4].predict(test_dato)
        #     if test_prompts[i] == "P6":
        #         test_dato = self.vectorizers[5].transform(test_data[i])
        #         prediction = self.classifiers[5].predict(test_dato)
        #     if test_prompts[i] == "P7":
        #         test_dato = self.vectorizers[6].transform(test_data[i])
        #         prediction = self.classifiers[6].predict(test_dato)
        #     if test_prompts[i] == "P8":
        #         test_dato = self.vectorizers[7].transform(test_data[i])
        #         prediction = self.classifiers[7].predict(test_dato)
        #     predictions.append(prediction[0])


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.

        # train_data, test_data, train_target, test_target, train_prompts, test_prompts, train_levels, test_levels = sklearn.model_selection.train_test_split(
        # train.data, train.target, train.prompts, train.levels, test_size=0.10, random_state=args.seed)

        # model = Model()
        # model.fit(train_data, train_target, train_prompts, train_levels)

        # predictions = model.predict(test_data, test_prompts, test_levels)
        # score = sklearn.metrics.accuracy_score(test_target, predictions)

        # print(score)


        model = Model()
        model.fit(train.data, train.target, train.prompts, train.levels)

        # kf = sklearn.model_selection.StratifiedKFold()
        # for train_index, test_index in kf.split(train.data, train.target):
        #     train_data = [train.data[i] for i in train_index]
        #     test_data = [train.data[i] for i in test_index]
        #     train_target = [train.target[i] for i in train_index]
        #     test_target = [train.target[i] for i in test_index]

        #     model = Model()
        #     model.fit(train_data, train_target)

        #     predictions = model.predict(test_data)
        #     score = sklearn.metrics.accuracy_score(test_target, predictions)

        #     print(score)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data, test.prompts, test.levels)
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)