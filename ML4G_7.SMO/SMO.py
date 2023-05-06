#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="rbf", type=str, help="Kernel type (poly/rbf)")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Stopping condition")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args: argparse.Namespace, x: np.ndarray, z: np.ndarray) -> np.ndarray:


	if args.kernel == "poly":
		ker = (args.kernel_gamma * x.T @ z + 1) ** args.kernel_degree
		return ker

	elif args.kernel == "rbf":
		norm = 0
		for j in range(z.shape[0]):
			norm += (x[j] - z[j])**2
		ker = np.exp(- args.kernel_gamma * norm)
		return ker



# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def smo(
	args: argparse.Namespace,
	train_data: np.ndarray, train_target: np.ndarray,
	test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:

	def predict(a, target, dato, data, b):
		y= 0
		for i in range(len(a)):
			y += a[i]*target[i]*kernel(args, dato, data[i])
		return y + b

	# Create initial weights.
	a, b = np.zeros(len(train_data)), 0
	generator = np.random.RandomState(args.seed)

	passes_without_as_changing = 0
	train_accs, test_accs = [], []
	for _ in range(args.max_iterations):
		as_changed = 0
		# Iterate through the data.
		for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
			# We want j != i, so we "skip" over the value of i.
			j = j + (j >= i)

			Ei = predict(a, train_target, train_data[i], train_data, b) - train_target[i]

			if not ((a[i] < args.C - args.tolerance and train_target[i]*Ei < - args.tolerance) or (a[i]>args.tolerance and train_target[i]*Ei > args.tolerance)):	continue

			Ej = predict(a, train_target, train_data[j], train_data, b) - train_target[j]

			L_2_derivated = 2*kernel(args, train_data[i], train_data[j])-kernel(args, train_data[i], train_data[i])-kernel(args, train_data[j], train_data[j])

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

			b_j_new = b - Ej - train_target[i]*(a_i_new-a[i])*kernel(args, train_data[i], train_data[j])- train_target[j]*(a_j_new-a[j])*kernel(args, train_data[j], train_data[j])
			b_i_new = b - Ei - train_target[i]*(a_i_new-a[i])*kernel(args, train_data[i], train_data[i])- train_target[j]*(a_j_new-a[j])*kernel(args, train_data[j], train_data[i])

			if a_j_new > args.tolerance and a_j_new < args.C - args.tolerance:
				b = b_j_new
			elif  a_i_new > args.tolerance and a_i_new < args.C - args.tolerance:
				b = b_i_new
			else:
				b = (b_i_new+b_j_new)/2

			a[j] = a_j_new
			a[i] = a_i_new

			as_changed += 1

		train_predictions = np.zeros((train_data.shape[0]))

		for i in range (train_predictions.shape[0]):
			prediction = predict(a, train_target, train_data[i], train_data, b)
			if prediction > 0:
				train_predictions[i] = 1
			else: train_predictions[i] = -1
		
		train_accs.append(sklearn.metrics.accuracy_score(train_predictions, train_target))

		test_predictions = np.zeros((test_data.shape[0]))

		for i in range (test_predictions.shape[0]):
			prediction = predict(a, train_target, test_data[i], train_data, b)
			if prediction > 0:
				test_predictions[i] = 1
			else: test_predictions[i] = -1

		test_accs.append(sklearn.metrics.accuracy_score(test_predictions, test_target))

		# Stop training if `args.max_passes_without_as_changing` passes were reached.
		passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
		if passes_without_as_changing >= args.max_passes_without_as_changing:
			break

		if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
			print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
				len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

	support_vectors = []
	support_vector_weights = []

	for i in range(len(a)):
		if a[i] > args.tolerance:
			support_vectors.append(train_data[i])
			support_vector_weights.append(a[i]*train_target[i])


	print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
		len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

	return support_vectors, support_vector_weights, b, train_accs, test_accs


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
	# Generate an artificial regression dataset, with +-1 as targets.
	data, target = sklearn.datasets.make_classification(
		n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
	target = 2 * target - 1

	# Split the dataset into a train set and a test set.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	# Run the SMO algorithm.
	support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
		args, train_data, train_target, test_data, test_target)

	if args.plot:
		import matplotlib.pyplot as plt

		def plot(predict, support_vectors):
			xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
			ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
			predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
			test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
			plt.figure()
			plt.contourf(xs, ys, predictions, levels=0, cmap="RdBu")
			plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
			plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap="RdBu", zorder=2)
			plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#0d0")
			plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap="RdBu", zorder=2)
			plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ff0")
			plt.legend(loc="upper center", ncol=4)

		# If you want plotting to work (not required for ReCodEx), you need to
		# define `predict_function` computing SVM value `y(x)` for the given x.
		def predict_function(x):
			return ...

		plot(predict_function, support_vectors)
		plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

	return support_vectors, support_vector_weights, bias, train_accs, test_accs


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	main(args)