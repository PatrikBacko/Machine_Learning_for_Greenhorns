#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

import collections
from Node import *

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=True, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="digits", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=0.5, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=3, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=10, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
	# Use the given dataset.
	data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

	# Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
	# with `test_size=args.test_size` and `random_state=args.seed`.
	train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
		data, target, test_size=args.test_size, random_state=args.seed)

	# Create random generators.
	generator_feature_subsampling = np.random.RandomState(args.seed)
	def subsample_features(number_of_features: int) -> np.ndarray:
		return generator_feature_subsampling.uniform(size=number_of_features) <= args.feature_subsampling

	generator_bootstrapping = np.random.RandomState(args.seed)
	def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
		return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

	# TODO: Create a random forest on the training data.
	#
	# Use a simplified decision tree from the `decision_tree` assignment:
	# - use `entropy` as the criterion
	# - use `max_depth` constraint, to split a node only if:
	#   - its depth is less than `args.max_depth`
	#   - the criterion is not 0 (the corresponding instance targets are not the same)
	# When splitting nodes, proceed in the depth-first order, splitting all nodes
	# in the left subtree before the nodes in right subtree.
	#
	# Additionally, implement:
	# - feature subsampling: when searching for the best split, try only
	#   a subset of features. Notably, when splitting a node (i.e., when the
	#   splitting conditions [depth, criterion != 0] are satisfied), start by
	#   generating a feature mask using
	#     subsample_features(number_of_features)
	#   which gives a boolean value for every feature, with `True` meaning the
	#   feature is used during best split search, and `False` it is not
	#   (i.e., when `feature_subsampling == 1`, all features are used).
	#
	# - train a random forest consisting of `args.trees` decision trees
	#
	# - if `args.bagging` is set, before training each decision tree
	#   create a bootstrap sample of the training data by calling
	#     dataset_indices = bootstrap_dataset(train_data)
	#   and if `args.bagging` is not set, use the original training data.
	#
	# During prediction, use voting to find the most frequent class for a given
	# input, choosing the one with the smallest class number in case of a tie.

	# TODO: Finally, measure the training and testing accuracy.

	class Decision_tree:
		args = None
		root = None

		split_candidates = None
		leaves_count= 0
		
		def __init__(self, args, train_data, train_target) -> None:
			self.args = args
			self.split_candidates = collections.deque()
			self.build_tree(train_data, train_target)

		def build_tree(self, train_data, train_target):
			self.root = Node(train_data, train_target, 0, self.calculate_criterion_value(train_target))
			self.leaves_count += 1
			
			if self.check_candidate_validity(self.root): self.split_candidates.append(self.root)
			
			while len(self.split_candidates) != 0:
				node = self.split_candidates.pop() 
				feature, boundry = self.find_best_boundry(node)

				self.split(node, feature, boundry)
		
		def split(self, node, feature, boundry):
			indexes_right = np.where(node.data[:,feature] > boundry)
			indexes_left = np.where(node.data[:,feature] < boundry)

			right_data = node.data[indexes_right]
			right_target = node.target[indexes_right]
			
			right_son = Node(right_data, right_target, node.depth +1, self.calculate_criterion_value(right_target))
			if self.check_candidate_validity(right_son): self.split_candidates.append(right_son)
			
			left_data = node.data[indexes_left]
			left_target = node.target[indexes_left]
			
			left_son = Node(left_data, left_target, node.depth +1, self.calculate_criterion_value(left_target))
			if self.check_candidate_validity(left_son):	self.split_candidates.append(left_son)

			self.leaves_count += 1

			node.feature = feature
			node.boundry = boundry
			node.right_son = right_son
			node.left_son = left_son

		def find_best_boundry(self, node) -> tuple[float, int, int]:
			best_feature, best_boundry = None, None
			best_crit_value = np.Infinity
			mask = subsample_features(node.data.shape[1])
			for i in range(node.data.shape[1]):
				if mask[i] == 0: continue
				boundries = np.unique(node.data[:,i])
				for j in range(len(boundries)-1):
					boundry  = (boundries[j] + boundries[j+1]) / 2

					indexes_left = np.where(node.data[:,i] < boundry)
					indexes_right = np.where(node.data[:,i] > boundry)

					crit_value = self.calculate_criterion_value(node.target[indexes_left]) + self.calculate_criterion_value(node.target[indexes_right])
					
					if crit_value < best_crit_value:
						best_crit_value = crit_value
						best_feature, best_boundry = i, boundry

			return (best_feature, best_boundry)

		def check_candidate_validity(self, candidate) -> bool:
			if self.args.max_depth != None:
				if candidate.depth >= self.args.max_depth:
					return False
			if candidate.crit_value <= 0:
				return False
			return True

		def calculate_criterion_value(self, target) -> float:
			instances_count = len(target)
			unique, counts = np.unique(target, return_counts=True)

			entropy = 0
			for i in counts:
				Pt_k = (i/instances_count)
				entropy += Pt_k*np.log(Pt_k)

			return -instances_count*entropy

		def predict(self, data) -> np.ndarray:
			node = self.root
			predictions = np.zeros((data.shape[0]))

			for i in range(data.shape[0]):
				while node.left_son != None:
					if data[i, node.feature] <= node.boundry:
						node = node.left_son
					elif data[i, node.feature] > node.boundry:
						node = node.right_son
				predictions[i] = node.predicted_class
				node = self.root

			return predictions

	trees = []
	for _ in range(args.trees):
		if args.bagging:
			bootstrap = bootstrap_dataset(train_data)
			trees.append(Decision_tree(args, train_data[bootstrap], train_target[bootstrap]))
		else:
			trees.append(Decision_tree(args, train_data, train_target))

	def predict_all_trees(data_to_predict):
		predictions_all_trees = np.zeros((0, data_to_predict.shape[0]))
		for tree in trees:
			prediction = tree.predict(data_to_predict)
			predictions_all_trees = np.concatenate((predictions_all_trees, np.reshape(prediction, (1,prediction.shape[0]))), axis = 0)
		
		predictions = np.zeros((data_to_predict.shape[0]))
		for i in range(data_to_predict.shape[0]):
			unique, counts = np.unique(predictions_all_trees[:,i], return_counts=True)
			predictions[i] = unique[np.argmax(counts)]

		return predictions

	train_accuracy = sklearn.metrics.accuracy_score(train_target, predict_all_trees(train_data))
	test_accuracy =sklearn.metrics.accuracy_score(test_target, predict_all_trees(test_data))

	return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
	args = parser.parse_args([] if "__file__" not in globals() else None)
	train_accuracy, test_accuracy = main(args)

	print("Train accuracy: {:.1f}%".format(train_accuracy))
	print("Test accuracy: {:.1f}%".format(test_accuracy))