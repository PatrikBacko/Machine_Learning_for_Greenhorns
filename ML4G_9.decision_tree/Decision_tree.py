#!/usr/bin/env python3
import argparse

import collections
import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from Node import *


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
			if self.args.max_leaves != None:
				if self.leaves_count >= self.args.max_leaves: break
				node, feature, boundry = self.find_node_to_split()
				self.split_candidates.remove(node)

			else:
				node = self.split_candidates.pop() 
				_, feature, boundry = self.find_best_boundry(node)

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

	def find_node_to_split(self) -> tuple[Node, int, int]:
		best_node, best_feature, best_boundry = None, None, None
		best_crit_value = np.Infinity

		for node in self.split_candidates:
			crit_value, feature, boundry = self.find_best_boundry(node)
			if crit_value - node.crit_value < best_crit_value:
				best_node, best_feature, best_boundry = node, feature, boundry
				best_crit_value = crit_value - node.crit_value

		return (best_node, best_feature, best_boundry)

	def find_best_boundry(self, node) -> tuple[float, int, int]:
		best_feature, best_boundry = None, None
		best_crit_value = np.Infinity
		for i in range(node.data.shape[1]):
			boundries = np.unique(node.data[:,i])
			for j in range(len(boundries)-1):
				boundry  = (boundries[j] + boundries[j+1]) / 2

				indexes_left = np.where(node.data[:,i] < boundry)
				indexes_right = np.where(node.data[:,i] > boundry)

				crit_value = self.calculate_criterion_value(node.target[indexes_left]) + self.calculate_criterion_value(node.target[indexes_right])
				
				if crit_value < best_crit_value:
					best_crit_value = crit_value
					best_feature, best_boundry = i, boundry

		return (best_crit_value, best_feature, best_boundry)

	def check_candidate_validity(self, candidate) -> bool:

		if self.args.max_depth != None:
			if candidate.depth >= self.args.max_depth:
				return False
		if self.args.min_to_split != None:
			if candidate.instances < self.args.min_to_split:
				return False
		if candidate.crit_value <= 0:
			return False

		return True

	def calculate_criterion_value(self, target) -> float:
		if self.args.criterion == "gini":
			return self.calculate_gini(target)
		elif self.args.criterion == "entropy":
			return self.calculate_entropy(target)

	def calculate_gini(self, target) -> float:
		instances_count = len(target)
		unique, counts = np.unique(target, return_counts=True)

		gini = 0
		for i in counts:
			Pt_k = (i/instances_count)
			gini += Pt_k*(1-Pt_k)

		return instances_count*gini

	def calculate_entropy(self, target) -> float:
		instances_count = len(target)
		unique, counts = np.unique(target, return_counts=True)

		entropy = 0
		for i in counts:
			Pt_k = (i/instances_count)
			entropy += Pt_k*np.log(Pt_k)

		return -instances_count*entropy

	def predict(self, test_data) -> list:
		node = self.root
		predictions = []

		for i in range(test_data.shape[0]):
			while node.left_son != None:
				if test_data[i, node.feature] <= node.boundry:
					node = node.left_son
				elif test_data[i, node.feature] > node.boundry:
					node = node.right_son
			predictions.append(node.predicted_class)
			node = self.root

		return predictions