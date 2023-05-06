#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=2, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def sigmoid(dato, weights):
	return 1/(1+np.exp(-(dato @ weights)))


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
	
	def calculate_f1_micro(data, target):
		tp, fn, fp = 0, 0, 0
		for i in range(data.shape[0]):
			dato = data[i]
			for j in range(args.classes):
				prediction = 0
				if sigmoid(dato, weights[:,j]) > 0.5:
					prediction = 1
				if (prediction == 1 and target[i,j] == 1):
					tp += 1
				elif (prediction == 0 and target[i,j] == 1):
					fn += 1
				elif (prediction == 1 and target[i,j] == 0):
					fp +=1
		return (2*tp)/(2*tp + fp + fn)

			
	def calculate_f1_macro(data, target):
		f1 = 0
		for j in range(args.classes):
			tp, fn, fp = 0, 0, 0
			for i in range(data.shape[0]):
				dato = data[i]
				prediction = 0
				if sigmoid(dato, weights[:,j]) > 0.5:
					prediction = 1
				if (prediction == 1 and target[i,j] == 1):
					tp += 1
				elif (prediction == 0 and target[i,j] == 1):
					fn += 1
				elif (prediction == 1 and target[i,j] == 0):
					fp +=1
			f1 += (2*tp)/(2*tp + fp + fn)
		return f1/args.classes
			



	# Create a random generator with a given seed.
	generator = np.random.RandomState(args.seed)

	# Generate an artificial classification dataset.
	data, target_list = sklearn.datasets.make_multilabel_classification(
		n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
		return_indicator=False, random_state=args.seed)


	
	# target = np.zeros((data.shape[0], args.classes))

	# for i in range(data.shape[0]):
	# 	for j in target_list[i]:
	# 		target[i,j] = 1



	n_hot = sklearn.preprocessing.MultiLabelBinarizer()
	target = n_hot.fit_transform(target_list)

	# Append a constant feature with value 1 to the end of every input data.
	# Then we do not need to explicitly represent bias - it becomes the last weight.
	data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

	# Split the dataset into a train set and a test set.
	# Use `sklearn.model_selection.train_test_split` method call, passing
	# arguments `test_size=args.test_size, random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	# Generate initial model weights.
	weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

	for epoch in range(args.epochs):
		permutation = generator.permutation(train_data.shape[0])

		for i in range(int(train_data.shape[0]/args.batch_size)):
			current_batch = permutation[i*args.batch_size : (i+1)*args.batch_size]
			gradient = np.zeros([ train_data.shape[1] , args.classes])  

			for j in current_batch:
				dato = train_data[j]
				for k in range(args.classes):
					
					sigmoidf = sigmoid(dato, weights[:,k])
					gradient[:,k] += (sigmoidf - train_target[j,k]) * dato

			gradient /= args.batch_size
			weights = weights - args.learning_rate * gradient

		# TODO: Process the data in the order of `permutation`. For every
		# `args.batch_size` of them, average their gradient, and update the weights.
		# You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.


		# TODO: After the SGD epoch, compute the micro-averaged and the
		# macro-averaged F1-score for both the train test and the test set.
		# Compute these scores manually, without using `sklearn.metrics`.
		train_f1_micro = calculate_f1_micro(train_data, train_target)
		train_f1_macro = calculate_f1_macro(train_data, train_target)
		test_f1_micro = calculate_f1_micro(test_data, test_target)
		test_f1_macro =	calculate_f1_macro(test_data, test_target)

		print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
			epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

	return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	weights, metrics = main(args)
	print("Learned weights:",
		  *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")