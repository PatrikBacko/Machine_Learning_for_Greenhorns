#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


class Node:
    right_son = None
    left_son = None

    data = None
    target = None

    crit_value = None
    depth = None
    instances = None

    predicted_class = None
    feature = None
    boundry = None
    
    def __init__(self, data, target, depth, criterion_value) -> None:
        self.data = data
        self.crit_value = criterion_value
        self.depth = depth
        self.target = target
        self.instances = len(target)

        unique, counts = np.unique(target, return_counts=True)
        self.predicted_class = unique[np.argmax(counts)]