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
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.005, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def softmax(vector: np.ndarray) -> np.ndarray:
	sum = 0 

	for j in range(vector.shape[1]):
		sum += np.exp(vector[0,j])

	return np.exp(vector)/sum


def calculate_loss(data, target, weights):
	loss = 0
	for i in range(data.shape[0]): 

		dato = data[i].reshape([data[i].shape[0],1])
		linear_part = dato.transpose() @ weights

		prediction = softmax(linear_part - np.max(linear_part))
		
		for j in range(target.shape[1]):
			if target[i,j] == 1:
				loss += - np.log(prediction[0, j])

	loss /= data.shape[0]

	return loss

def calculate_accuracy(data, target, weights):
	accuracy = 0
	for i in range(data.shape[0]):

		dato = data[i].reshape([data[i].shape[0],1])
		linear_part = dato.transpose() @ weights

		prediction = softmax(linear_part - np.max(linear_part))

		curr_target = target[i]

		if  prediction @ curr_target.T == np.max(prediction):
			accuracy += 1

	return accuracy/data.shape[0]



def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
	# Create a random generator with a given seed.
	generator = np.random.RandomState(args.seed)

	# Load the digits dataset.
	data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

	# Append a constant feature with value 1 to the end of every input data.
	# Then we do not need to explicitly represent bias - it becomes the last weight.
	data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

	# Split the dataset into a train set and a test set.
	# Use `sklearn.model_selection.train_test_split` method call, passing
	# arguments `test_size=args.test_size, random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	features_count = data.shape[1]

	train_target = train_target.reshape([train_target.shape[0],1])
	test_target = test_target.reshape([test_target.shape[0],1])

	onehot = sklearn.preprocessing.OneHotEncoder(sparse = False)

	train_target  = onehot.fit_transform(train_target)
	test_target = onehot.fit_transform(test_target)

	# Generate initial model weights.
	weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

	for epoch in range(args.epochs):
		permutation = generator.permutation(train_data.shape[0])

		for i in range(int(train_data.shape[0]/args.batch_size)):
			current_batch = permutation[i*args.batch_size : (i+1)*args.batch_size]
			gradient = np.zeros([ args.classes, features_count])  

			for j in current_batch:
				dato = train_data[j].reshape([train_data[j].shape[0],1])

				linear_part = dato.transpose() @ weights

				gradient += (softmax((linear_part - np.max(linear_part)))-train_target[j]).T @ dato.transpose()

			gradient /= args.batch_size
			weights = weights - args.learning_rate * gradient.transpose()

		train_loss = calculate_loss(train_data, train_target, weights)
		test_loss = calculate_loss(test_data, test_target, weights)

		train_accuracy = calculate_accuracy(train_data, train_target, weights)
		test_accuracy = calculate_accuracy(test_data, test_target, weights)

		print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
			epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

	return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	weights, metrics = main(args)
	print("Learned weights:",
		  *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")