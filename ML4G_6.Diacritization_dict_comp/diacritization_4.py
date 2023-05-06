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
import sklearn.neighbors


import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=False, type=str, help="Path to the dataset to predict")
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

nodia = ["a","c","d","e","i","n","o","r","s","t","u","y","z"]

classes = {
	'á': [0,1,0,0],
	'č': [0,0,1,0],
	'ď': [0,0,1,0],
	'é': [0,0,1,0],
	'ě': [0,0,1,0],
	'í': [0,1,0,0],
	'ň': [0,0,1,0],
	'ó': [0,1,0,0],
	'ř': [0,0,1,0],
	'š': [0,0,1,0],
	'ť': [0,0,1,0],
	'ú': [0,1,0,0],
	'ů': [0,0,0,1],
	'ý': [0,1,0,0],
	'ž': [0,0,1,0],
	'Á': [0,1,0,0],
	'Č': [0,0,1,0],
	'Ď': [0,0,1,0],
	'É': [0,0,1,0],
	'Ě': [0,0,1,0],
	'Í': [0,1,0,0],
	'Ň': [0,0,1,0],
	'Ó': [0,1,0,0],
	'Ř': [0,0,1,0],
	'Š': [0,0,1,0],
	'Ť': [0,0,1,0],
	'Ú': [0,1,0,0],
	'Ů': [0,0,0,1],
	'Ý': [0,1,0,0],
	'Ž': [0,0,1,0]
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
	('D', 2): 'ď',
	('E', 1): 'É',
	('E', 2): 'Ě',
	('I', 1): 'Í',
	('N', 2): 'Ň',
	('O', 1): 'ó',
	('R', 2): 'Ř',
	('S', 2): 'Š',
	('T', 2): 'Ť',
	('U', 1): 'Ú',
	('U', 3): 'Ů',
	('Y', 1): 'Ý',
	('Z', 2): 'Ž'
	}

class Dictionary:
	def __init__(self,
				 name="fiction-dictionary.txt",
				 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
		if not os.path.exists(name):
			print("Downloading dataset {}...".format(name), file=sys.stderr)
			licence_name = name.replace(".txt", ".LICENSE")
			urllib.request.urlretrieve(url + licence_name, filename=licence_name)
			urllib.request.urlretrieve(url + name, filename=name)

		# Load the dictionary to `variants`
		self.variants = {}
		with open(name, "r", encoding="utf-8-sig") as dictionary_file:
			for line in dictionary_file:
				nodia_word, *variants = line.rstrip("\n").split()
				self.variants[nodia_word] = variants

class Model:
	def __init__(self):
		self.k = 4
		self.mlp = sklearn.neural_network.MLPClassifier(max_iter=500, batch_size=100, hidden_layer_sizes=(500))

	def prepare_data(self, train):
		global classes
		global nodia
		k = 5

		train_data = np.ndarray((len(train.data) ,2*k+1), dtype=np.ushort)
		train_target = []
		self.data_indices = []

		for i in range(len(train.data)):
			previous = train.data[max(i-k, 0):i]
			previous = previous[::-1]
			next = train.data[i+1:min(k+i+1, len(train.data))]

			if train.data[i].lower() in nodia:
				self.data_indices.append(i)

			feature = np.zeros((2*k+1), dtype=int)
			feature[k] = ord(train.data[i])
			for j in range(0, len(previous)):
				feature[j] = ord(previous[j-1])
			for j in range(k+1, len(next)+k+1):
				feature[j] = ord(next[j-k-1])
			
			train_data[i] = feature
			if train.target[i] in classes.keys():
				train_target.append(classes[train.target[i]]) 
			elif train.target[i] in nodia:
				train_target.append([0,0,0,0]) 

		self.one_hot = sklearn.preprocessing.OneHotEncoder(dtype=np.ushort, handle_unknown="ignore")
		train_data = self.one_hot.fit_transform(train_data)
		self.__train_data = train_data
		self.__train_target = train_target

	def fit(self):

		train_data = self.__train_data[self.data_indices]
		train_target = self.__train_target

		# train_data = self.__train_data
		# train_target = self.__train_target

		self.mlp.fit(train_data, train_target)

		self.__train_data = None
		self.__train_target = None

		# for val in self.letters.values():
		# 	val._optimizer = None
		# 	for i in range(len(val.coefs_)): val.coefs_[i] = val.coefs_[i].astype(np.float32)
		# 	for i in range(len(val.intercepts_)): val.intercepts_[i] = val.intercepts_[i].astype(np.float32)

	def prepare_test_data(self, train):
		global classes
		k = 5

		train_data = np.ndarray((len(train) ,2*k+1), dtype=np.ushort)

		for i in range(len(train)):
			previous = train[max(i-k, 0):i]
			previous = previous[::-1]
			next = train[i+1:min(k+i+1, len(train))]

			feature = np.zeros((2*k+1), dtype=int)
			feature[0] = ord(train[i])
			for j in range(1, len(previous)+1):
				feature[j] = ord(previous[j-1])
			for j in range(k+1, len(next)+k+1):
				feature[j] = ord(next[j-k-1])
			
			train_data[i] = feature
			
		train_data = self.one_hot.transform(train_data)
		self.__train_data = train_data

	def predict(self, data):
		global inverse_classes
		global classes

		# knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
		dictionary = Dictionary() 
		self.prepare_test_data(data)

		#predicted_string = np.ndarray((len(data)), dtype=np.char)
		predicted_string = []
		i=0


		while i < len(data):
			predictedwordlist = []
			wordlist = []
			while data[i] != " ":
				if data[i].lower() in nodia:
					predictions = self.mlp.predict(self.__train_data[i])
					prediction = 0

					for j in range(len(predictions[0])):
						if predictions[0][j] == 1:
							prediction = j

					if (data[i], prediction) in inverse_classes:
						#file.write(inverse_classes[(data[i], prediction)])
						#print(inverse_classes[(data[i], prediction)], end='')

						#predicted_string[i] = inverse_classes[(data[i], prediction)]
						predictedwordlist.append(inverse_classes[(data[i], prediction)])
					else:
						#print(data[i], end='')

						#predicted_string[i] = data[i]

						predictedwordlist.append(data[i])
				else:
					#file.write(data[i])
					#print(data[i], end='')
					
					#predicted_string[i] = data[i]
					predictedwordlist.append(data[i])
				wordlist.append(data[i])
				i+=1
				if i == len(data): break

			
			predictedword = "".join(predictedwordlist)
			word = "".join(wordlist)
			
			if word in dictionary.variants:
				if predictedword in dictionary.variants[word]:
					word = predictedword

			predicted_string.append(word)
			if i == len(data): break
			predicted_string.append(data[i])
			i+=1

			wordlist = []
			predictedwordlist = []

		string = "".join(predicted_string) 
		print(string)
		return string


def main(args: argparse.Namespace) -> Optional[str]:
	if args.predict is None:
		# We are training a model.
		np.random.seed(args.seed)
		train = Dataset()
		# train.data = train.data[0:100]
		# train.target = train.target[0:100]
		global classes
		k = 5
		model = Model()
		model.prepare_data(train)
		model.fit()

		# Serialize the model.
		with lzma.open(args.model_path, "wb") as model_file:
			pickle.dump(model, model_file)

	else:
		# Use the model and return test set predictions.
		train = Dataset(args.predict)

		with lzma.open(args.model_path, "rb") as model_file:
			model = pickle.load(model_file)

		#file = open("C:\\Users\\Oliver\\Desktop\\output.txt", "w", encoding="utf-8-sig")
		# TODO: Generate `predictions` with the test set predictions. Specifically,
		# produce a diacritized `str` with exactly the same number of words as `test.data`.
		predictions = model.predict(train.data)
		

		return predictions


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	main(args)