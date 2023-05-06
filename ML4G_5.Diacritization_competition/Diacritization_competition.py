#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.neural_network
import sklearn.preprocessing


import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Dataset:
	LETTERS_NODIA = "acdeeinorstuuyz"
	LETTERS_DIA = "áčďéěíňóřšťúůýž"

	# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
	DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

	def __init__(self,
				 name="fiction-train.txt",
				 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
		if not os.path.exists(name):
			print("Downloading dataset {}...".format(name), file=sys.stderr)
			licence_name = name.replace(".txt", ".LICENSE")
			urllib.request.urlretrieve(url + licence_name, filename=licence_name)
			urllib.request.urlretrieve(url + name, filename=name)

		# Load the dataset and split it into `data` and `target`.
		with open(name, "r", encoding="utf-8-sig") as dataset_file:
			self.target = dataset_file.read()
		self.data = self.target.translate(self.DIA_TO_NODIA)

classes = {
	'á': 1,
	'č': 2,
	'ď': 2,
	'é': 2,
	'ě': 2,
	'í': 1,
	'ň': 2,
	'ó': 1,
	'ř': 2,
	'š': 2,
	'ť': 2,
	'ú': 1,
	'ů': 3,
	'ý': 1,
	'ž': 2,

	'Á': 1,
	'Č': 2,
	'Ď': 2,
	'É': 2,
	'Ě': 2,
	'Í': 1,
	'Ň': 2,
	'Ó': 1,
	'Ř': 2,
	'Š': 2,
	'Ť': 2,
	'Ú': 1,
	'Ů': 3,
	'Ý': 1,
	'Ž': 2
	}

inverse_classes = {
	('a', 1): 'á',
	('c', 2): 'č',
	('d', 2): 'ď',
	('e', 1): 'é',
	('e', 2): 'ě',
	('i', 1): 'í',
	('n', 2): 'ň',
	('o', 1): 'ó',
	('r', 2): 'ř',
	('s', 2): 'š',
	('t', 2): 'ť',
	('u', 1): 'ú',
	('u', 3): 'ů',
	('y', 1): 'ý',
	('z', 2): 'ž',

	('A', 1): 'Á',
	('C', 2): 'Č',
	('D', 2): 'Ď',
	('E', 1): 'É',
	('E', 2): 'Ě',
	('I', 1): 'Í',
	('N', 2): 'Ň',
	('O', 1): 'Ó',
	('R', 2): 'Ř',
	('S', 2): 'Š',
	('T', 2): 'Ť',
	('U', 1): 'Ú',
	('U', 3): 'Ů',
	('Y', 1): 'Ý',
	('Z', 2): 'Ž',
	}

def main(args: argparse.Namespace) -> Optional[str]:
	if args.predict is None:

		np.random.seed(args.seed)
		train = Dataset()
		global classes
		k = 10

		train_data = np.ndarray((len(train.data) ,2*k+1), dtype=int)

		train_target = np.ndarray((len(train.data)), dtype=int)
		for i in range(len(train.data)):
			previous = train.data[max(i-k, 0):i]
			previous = previous[::-1]
			next = train.data[i+1:min(k+i+1, len(train.data))]

			feature = np.zeros((2*k+1), dtype=int)
			feature[0] = ord(train.data[i])
			for j in range(1, len(previous)+1):
				feature[j] = ord(previous[j-1])
			for j in range(k+1, len(next)+k+1):
				feature[j] = ord(next[j-k-1])
			
			train_data[i] = feature
			if train.target[i] in classes.keys():
				train_target[i] = classes[train.target[i]]
			else:
				train_target[i] = 0

		classifier = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(200), max_iter=400, batch_size=100)
		scaler = sklearn.preprocessing.StandardScaler()
		poly = sklearn.preprocessing.PolynomialFeatures(degree=2)

		train_data = poly.fit_transform(train_data)
		train_data = scaler.fit_transform(train_data)

		model = classifier.fit(train_data, train_target)

		with lzma.open(args.model_path, "wb") as model_file:
			pickle.dump(model, model_file)

	else:

		train = Dataset()
		global inverse_classes
		k=10
		train_data = np.ndarray((len(train.data) ,2*k+1), dtype=int)

		train_target = np.ndarray((len(train.data)), dtype=int)
		for i in range(len(train.data)):
			previous = train.data[max(i-k, 0):i]
			previous = previous[::-1]
			next = train.data[i+1:min(k+i+1, len(train.data))]

			feature = np.zeros((2*k+1), dtype=int)
			feature[0] = ord(train.data[i])
			for j in range(1, len(previous)+1):
				feature[j] = ord(previous[j-1])
			for j in range(k+1, len(next)+k+1):
				feature[j] = ord(next[j-k-1])
			
			train_data[i] = feature
			if train.target[i] in classes.keys():
				train_target[i] = classes[train.target[i]]
			else:
				train_target[i] = 0
		poly = sklearn.preprocessing.StandardScaler()
		train_data = poly.fit_transform(train_data)

		with lzma.open(args.model_path, "rb") as model_file:
			model = pickle.load(model_file)

		file = open("output.txt", "w")

		predictions = model.predict(train_data)
		for i in range(len(train.data)):
			if (train.data[i], predictions[i]) in inverse_classes:
				file.write(inverse_classes[(train.data[i], predictions[i])])
				print(inverse_classes[(train.data[i], predictions[i])], end='')
			else:
				file.write(train.data[i])
				print(train.data[i], end='')
		
		file.close()

		return predictions


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	main(args)
