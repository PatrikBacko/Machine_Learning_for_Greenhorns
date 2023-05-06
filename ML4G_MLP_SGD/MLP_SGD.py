#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=20, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.



def softmax(vector: np.ndarray) -> np.ndarray:  #gets vector, returns vector (1 dim)

	# matrix = np.exp(matrix)
	# sum= np.ndarray([matrix.shape[0],1])

	# for i in range(matrix.shape[0]):
	# 	for j in range(matrix.shape[1]):
	# 		sum[i] += matrix[i,j]
	# 	sum[i] = 1/sum[i]
	# return sum @ matrix

	sum = 0
	vector = np.exp(vector)

	for scalar in vector:
		sum += scalar

	return vector/sum


def ReLU(vector: np.ndarray) -> np.ndarray:    #gets vector, returns vector (1 dim)
	# for i in range(matrix.shape[0]):
	# 	for j in range(matrix.shape[1]):
	# 		matrix[i,j] = max(matrix[i,j], 0)
	# return matrix

	for i in range(vector.shape[0]):
		vector[i] = max(0, vector[i])
	
	return vector

def ReLU_derivate(vector: np.ndarray) -> np.ndarray:
	ReLU_der = np.zeros([vector.shape[0]])
	for i in range(vector.shape[0]):
		if vector[i] > 0:
			ReLU_der[i] = 1
	return ReLU_der





def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
	# Create a random generator with a given seed.
	generator = np.random.RandomState(args.seed)

	# Load the digits dataset.
	data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

	# Split the dataset into a train set and a test set.
	# Use `sklearn.model_selection.train_test_split` method call, passing
	# arguments `test_size=args.test_size, random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)


	train_target = train_target.reshape((train_target.shape[0],1))
	test_target = test_target.reshape((test_target.shape[0],1))

	onehot = sklearn.preprocessing.OneHotEncoder(sparse = False)

	train_target  = onehot.fit_transform(train_target)
	test_target = onehot.fit_transform(test_target)


	# Generate initial model weights.
	weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
			   generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
	biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

	features_count = data.shape[1]

	def forward(inputs):
		# TODO: Implement forward propagation, returning *both* the value of the hidden
		# layer and the value of the output layer.
		#
		# We assume a neural network with a single hidden layer of size `args.hidden_layer`
		# and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
		# activation.
		#
		# The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
		# The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
		#
		# Note that you need to be careful when computing softmax, because the exponentiation
		# in softmax can easily overflow. To avoid it, you should use the fact that
		# $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
		# That way we only exponentiate values which are non-positive, and overflow does not occur.

		hidden_layer = ReLU(inputs @ weights[0] + biases[0])
		softmax_input = hidden_layer @ weights[1] + biases[1]
		output = softmax(softmax_input - np.max(softmax_input))

		return (hidden_layer, output)

		
		# TODO: Process the data in the order of `permutation`. For every
		# `args.batch_size` of them, average their gradient, and update the weights.
		# You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
		#
		# The gradient used in SGD has now four parts, gradient of `weights[0]` and `weights[1]`
		# and gradient of `biases[0]` and `biases[1]`.
		#
		# You can either compute the gradient directly from the neural network formula,
		# i.e., as a gradient of $-log P(target | data)$, or you can compute
		# it step by step using the chain rule of derivatives, in the following order:
		# - compute the derivative of the loss with respect to *inputs* of the
		#   softmax on the last layer,
		# - compute the derivative with respect to `weights[1]` and `biases[1]`,
		# - compute the derivative with respect to the hidden layer output,
		# - compute the derivative with respect to the hidden layer input,
		# - compute the derivative with respect to `weights[0]` and `biases[0]`.

		# TODO: After the SGD epoch, measure the accuracy for both the
		# train test and the test set.

	def calculate_accuracy(data, target):
		accuracy = 0
		for i in range(data.shape[0]):
			_, prediction = forward(data[i])
			curr_target = target[i]
			xd = prediction @ curr_target
			if  xd == np.max(prediction):
				accuracy += 1
		return accuracy/data.shape[0]
		
	for epoch in range(args.epochs):
		permutation = generator.permutation(train_data.shape[0])

		for i in range(int(train_data.shape[0]/args.batch_size)):
			current_batch = permutation[i*args.batch_size : (i+1)*args.batch_size]

			gradient_bias = np.zeros([args.classes])
			gradient_hidden_layer_in = np.zeros([ args.hidden_layer])
			gradient_Weights_hidden_layer = np.zeros([ train_data.shape[1], args.hidden_layer])
			gradient_Weights = np.zeros([ args.hidden_layer, args.classes])

			for j in current_batch:

				dato = train_data[j]
				curr_target = train_target[j]

				hidden_layer, prediction = forward(dato)

				ReLU_der = ReLU_derivate(hidden_layer)

				gradient_bias += prediction - curr_target
				gradient_Weights += np.outer(hidden_layer, (prediction - curr_target))
				gradient_hidden_layer_in +=  (weights[1] @ (prediction - curr_target)) * ReLU_der
				gradient_Weights_hidden_layer += np.outer(dato, (weights[1] @ (prediction - curr_target)) * ReLU_der)
				
			gradient_Weights /= args.batch_size
			gradient_hidden_layer_in /= args.batch_size
			gradient_Weights_hidden_layer /= args.batch_size
			gradient_bias /= args.batch_size

			weights[1] -=  args.learning_rate * gradient_Weights
			weights[0] -= args.learning_rate * gradient_Weights_hidden_layer
			biases[1] -= args.learning_rate * gradient_bias
			biases[0] -= args.learning_rate * gradient_hidden_layer_in

		train_accuracy = calculate_accuracy(train_data, train_target)
		test_accuracy = calculate_accuracy(test_data, test_target)
 

		print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
			epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

	return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	parameters, metrics = main(args)
	print("Learned parameters:",
		  *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")