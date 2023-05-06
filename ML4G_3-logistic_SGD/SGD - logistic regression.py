#!/usr/bin/env python3
import argparse
from cmath import exp

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=9, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.5, type=float, help="Learning rate")
parser.add_argument("--plot", default=True, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.







def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
	# Create a random generator with a given seed
	generator = np.random.RandomState(args.seed)

	# Generate an artificial classification dataset
	data, target = sklearn.datasets.make_classification(
		n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

	data = np.concatenate([data, np.ones([data.shape[0], 1])], axis = 1)

	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
														data, target, 
														test_size=args.test_size, random_state=args.seed)
	features_count = data.shape[1]


	weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

	for epoch in range(args.epochs):
		permutation = generator.permutation(train_data.shape[0])

		for i in range(int(train_data.shape[0]/args.batch_size)):
			current_batch = permutation[i*args.batch_size : (i+1)*args.batch_size]
			gradient = np.zeros([features_count])   

			for j in current_batch:
				dato = train_data[j]

				sigmoid = 1/(1+np.exp(-(dato @ weights)))

				gradient += (sigmoid - train_target[j]) * dato

			gradient /= args.batch_size
			weights = weights - args.learning_rate * gradient

		train_loss = calculate_loss(train_data, train_target, weights)
		test_loss = calculate_loss(test_data, test_target, weights)

		train_accuracy = calculate_accuracy(train_data, train_target, weights)
		test_accuracy = calculate_accuracy(test_data, test_target, weights)


		print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
					epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

		if args.plot:
			import matplotlib.pyplot as plt
			if args.plot is not True:
				plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
				plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
			xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
			ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
			predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights))) for x in xs] for y in ys]
			plt.contourf(xs, ys, predictions, levels=21, cmap="RdBu", alpha=0.7)
			plt.contour(xs, ys, predictions, levels=[0.25, 0.5, 0.75], colors="k")
			plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, label="train", marker="P", cmap="RdBu")
			plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, label="test", cmap="RdBu")
			plt.legend(loc="upper right")
			plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")

	return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	weights, metrics = main(args)
	print("Learned weights", *("{:.2f}".format(weight) for weight in weights))