#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=5, type=int, help="Number of classes")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=2, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=20, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Stopping condition")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.8, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, z: np.ndarray) -> np.ndarray:
	if args.kernel == "poly":
		ker = (args.kernel_gamma * x.T @ z + 1) ** args.kernel_degree
		return ker

	elif args.kernel == "rbf":
		vector = x-z
		ker = np.exp(- args.kernel_gamma * (vector @ vector))

		return ker


def predict(args, dato, data, weights, b):
	y= 0
	for i in range(len(weights)):
		y += weights[i]*kernel(args, dato, data[i])
	return y + b


def smo(
	args: argparse.Namespace,
	train_data: np.ndarray, train_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:

	def predict_smo(a, target, dato, data, b):
		y= 0
		for i in range(len(a)):
			y += a[i]*target[i]*kernel(args, dato, data[i])
		return y + b
	
	a, b = np.zeros(len(train_data)), 0
	generator = np.random.RandomState(args.seed)

	passes_without_as_changing = 0

	ker = np.zeros([train_data.shape[0], train_data.shape[0]])

	for i in range(train_data.shape[0]):
		for j in range(train_data.shape[0]):
			ker[i][j] = kernel(args, train_data[i], train_data[j])

	for _ in range(args.max_iterations):
		as_changed = 0
		for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
			j = j + (j >= i)

			Ei = predict_smo(a, train_target, train_data[i], train_data, b) - train_target[i]

			if not ((a[i] < args.C - args.tolerance and train_target[i]*Ei < - args.tolerance) or (a[i]>args.tolerance and train_target[i]*Ei > args.tolerance)):	continue

			Ej = predict_smo(a, train_target, train_data[j], train_data, b) - train_target[j]

			L_2_derivated = 2*ker[i][j]-ker[i][i]-ker[j][j]

			if L_2_derivated > - args.tolerance : continue
			
			a_j_new = a[j] - train_target[j] * ((Ei-Ej)/(L_2_derivated))

			if train_target[i] == train_target[j]:
				l = max(0, a[i]+a[j]-args.C)
				h = min(args.C, a[i]+a[j])
			elif train_target[i] == - train_target[j]:
				l = max(0, a[j]-a[i])
				h = min(args.C, args.C + a[j]-a[i])

			if a_j_new > h: a_j_new = h
			if a_j_new < l: a_j_new = l

			if abs(a[j]- a_j_new) < args.tolerance: continue

			a_i_new = a[i] - train_target[i]*train_target[j]*(a_j_new - a[j])

			b_j_new = b - Ej - train_target[i]*(a_i_new-a[i])*ker[i][j]- train_target[j]*(a_j_new-a[j])*ker[j][j]
			b_i_new = b - Ei - train_target[i]*(a_i_new-a[i])*ker[i][i]- train_target[j]*(a_j_new-a[j])*ker[j][i]

			if a_j_new > args.tolerance and a_j_new < args.C - args.tolerance:
				b = b_j_new
			elif  a_i_new > args.tolerance and a_i_new < args.C - args.tolerance:
				b = b_i_new
			else:
				b = (b_i_new+b_j_new)/2

			a[j] = a_j_new
			a[i] = a_i_new

			as_changed += 1


		passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
		if passes_without_as_changing >= args.max_passes_without_as_changing:
			break

	support_vectors = []
	support_vector_weights = []

	for i in range(len(a)):
		if a[i] > args.tolerance:
			support_vectors.append(train_data[i])
			support_vector_weights.append(a[i]*train_target[i])

	return support_vectors, support_vector_weights, b


def main(args: argparse.Namespace) -> float:
	# Load the digits dataset.
	data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
	data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

	# Split the dataset into a train set and a test set.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	classes_indices = []

	for _ in range(args.classes):
		classes_indices.append([])

	for i in range(train_data.shape[0]):
		classes_indices[train_target[i]].append(i)

	classifiers = {}

	for i in range(args.classes):
		for j in range(i+1, args.classes):
			indices = classes_indices[i] + classes_indices[j]
			indices.sort()

			temp_target = train_target.copy()
			temp_target[classes_indices[i]] = 1
			temp_target[classes_indices[j]] = -1

			temp_target = temp_target[indices]
			temp_data =	train_data[indices]

			classifier = smo(args, temp_data, temp_target)
			classifiers[(i,j)] = classifier

	predictions = np.zeros([len(test_target)])

	for i in range(len(test_target)):
		prediction = []
		frequencies = np.zeros([args.classes])
		for j in classifiers:

			classifier = classifiers[j]
			prediction= predict(args, test_data[i], classifier[0], classifier[1], classifier[2])

			if prediction > 0: prediction = j[0]
			else: prediction = j[1]

			frequencies[prediction] += 1
			
			predictions[i] = np.argmax(frequencies)

	test_accuracy = sklearn.metrics.accuracy_score(predictions, test_target)

	return test_accuracy


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	accuracy = main(args)
	print("Test set accuracy: {:.2f}%".format(100 * accuracy))