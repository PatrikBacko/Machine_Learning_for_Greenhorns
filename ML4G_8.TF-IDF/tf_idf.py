#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import re

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=True, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=177, type=int, help="Random seed")
parser.add_argument("--tf", default=True, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
	def __init__(self,
				 name="20newsgroups.train.pickle",
				 data_size=None,
				 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
		if not os.path.exists(name):
			print("Downloading dataset {}...".format(name), file=sys.stderr)
			urllib.request.urlretrieve(url + name, filename=name)

		with lzma.open(name, "rb") as dataset_file:
			dataset = pickle.load(dataset_file)

		self.DESCR = dataset.DESCR
		self.data = dataset.data[:data_size]
		self.target = dataset.target[:data_size]
		self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
	# Load the 20newsgroups data.
	newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

	# Create train-test split.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

	train_doc_dicts_list = []
	features_dict = {}
	for i in range (len(train_data)):
		doc_dict = {}
		doc = train_data[i]
		list = re.findall("\w+", doc)

		for word in list:
			if word in features_dict.keys():
				features_dict[word] += 1 
			else:
				features_dict[word] = 1

			if word in doc_dict.keys():
				doc_dict[word] += 1 
			else:
				doc_dict[word] = 1
		
		train_doc_dicts_list.append(doc_dict)


	features_dict = { k:v for k, v in features_dict.items() if v >=2 }

	features_count = len(features_dict.keys())


	train_features = np.zeros((len(train_data), features_count))
	docs_containing_words = np.zeros((features_count))


	for i in range (len(train_data)):
		j=0
		for key in features_dict.keys():
			doc_dict = train_doc_dicts_list[i]
			if key in doc_dict.keys():
				if args.tf:
					train_features[i,j] = doc_dict[key]
				else:
					train_features[i,j] = 1
				docs_containing_words[j] += 1
			j += 1 
				

	idf = np.zeros((len(train_data), features_count))
	idf[:,:] = len(train_data)
	
	for i in range(len(train_data)):
		idf[i,:] /= (docs_containing_words+1)

	idf = np.log(idf)

	if args.idf:
		train_features = train_features * idf


	test_doc_dicts_list = []
	for i in range (len(test_data)):
		doc_dict = {}
		doc = test_data[i]
		list = re.findall("\w+", doc)

		for word in list:
			if word in doc_dict.keys():
				doc_dict[word] += 1 
			else:
				doc_dict[word] = 1
		
		test_doc_dicts_list.append(doc_dict)

	test_features = np.zeros((len(test_data), features_count))

	for i in range (len(test_data)):
		j=0
		for key in features_dict.keys():
			doc_dict = test_doc_dicts_list[i]
			if key in doc_dict.keys():
				if args.tf:
					test_features[i,j] = doc_dict[key]
				else:
					test_features[i,j] = 1
			j += 1
	
	if args.idf:
		test_features = test_features * idf[0:test_data,:]

	
	knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors= args.k, algorithm="brute", metric="cosine")
	knn.fit(train_features, train_target)
	predictions = knn.predict(test_features)


	# TODO: Create a feature for every term that is present at least twice
	# in the training data. A term is every maximal sequence of at least 1 word character,
	# where a word character corresponds to a regular expression `\w`.

	# TODO: For each document, compute its features as
	# - term frequency(TF), if `args.tf` is set;
	# - otherwise, use binary indicators (1 if a given term is present, else 0)
	#
	# Then, if `args.idf` is set, multiply the document features by the
	# inverse document frequencies (IDF), where
	# - use the variant which contains `+1` in the denominator;
	# - the IDFs are computed on the train set and then reused without
	#   modification on the test set.

	# TODO: Perform classification of the test set using the k-NN algorithm
	# from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
	# neighbors. For TF-IDF vectors, the cosine similarity is usually used, where
	#   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
	#
	# To employ this metric, you have several options:
	# - you could try finding out whether `KNeighborsClassifier` supports it directly;
	# - or you could compute it yourself, but if you do, you have to precompute it
	#   in a vectorized way, so using `metric="precomputed"` is fine, but passing
	#   a callable as the `metric` argument is not (it is too slow);
	# - finally, the nearest neighbors according to cosine_similarity are equivalent to
	#   the neighbors obtained by the usual Euclidean distance on L2-normalized vectors.

	# TODO: Evaluate the performance using a macro-averaged F1 score.
	f1_score = sklearn.metrics.f1_score(test_target, predictions, average="macro")

	return f1_score


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	f1_score = main(args)
	print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))