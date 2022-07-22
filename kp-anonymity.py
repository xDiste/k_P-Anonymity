import os
import numpy as np
import pandas as pd
import sys
import random
from node import Node
from dataset_anonymized import DatasetAnonymized


def recycleBadLeaves(good_leaf_nodes, bad_leaf_nodes, p_value, paa_values):
    '''
    while sum of all bad leaves' size >= P:
        if any bad leaves can merge then:
            merge them to a new node leaf-merge;
            if leaf-merge.size >= P then:
                leaf-merge.label = good-leaf
            else
                leaf-merge.label = bad-leaf
        current-level--
    suppress all time series contained in bad leaves
    '''
    # Bad leaves can mearge means that a BL1 and BL2 (nodes in bad_leaf_nodes) have same level and same pattern rappresentation

    # I have to find the max-bad-level
    max_bad_level = 0
    for bad_node in bad_leaf_nodes:
        bad_node.level > max_bad_level:
        max_bad_level = bad_node.level

    current_level = max_bad_level

    all_bad_leaves_size = 0
    for bad_node in bad_leaf_nodes:
        all_bad_leaves_size += bad_node.size

    while all_bad_leaves_size >= p_value:
        nodeInCurrentLevel = dict() # node: PR and level is the current level
        for bad_node in bad_leaf_nodes:
            patternRappresentation = bad_node.pattern_representation
            nodeInCurrentLevel[patternRappresentation].append(bad_node) 

        # now merge all
        for pr, nodeList in nodeInCurrentLevel.items():
            group = dict()
            for node in nodeList:
                bad_leaf_nodes.remove(node)
                group.update(node.group)

            leaf_merge = Node(level=current_level, pattern_representation=pr, group=group, paa_value=paa_value)

            if leaf_merge.size >= p_value:
                leaf_merge.label = "G"
                good_leaf_nodes.append(leaf_merge)
                all_bad_leaves_size -= leaf_merge.size
            else
                leaf_merge.label = "B"
                bad_leaf_nodes.append(leaf_merge)
        current_level -= 1

def main_KAPRA(k_value=None, p_value=None, paa_value=None dataset_path=None):
    if os.path.isfile(dataset_path):
        # read time_series_from_file
        time_series = pd.read_csv(dataset_path)

        # get columns name
        columns = list(time_series.columns)
        columns.pop(0)  # remove
        # save all maximum value for each attribute
        attributes_maximum_value = list()
        attributes_minimum_value = list()
        for column in columns:
            attributes_maximum_value.append(time_series[column].max())
            attributes_minimum_value.append(time_series[column].min())
        time_series_dict = dict()

        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row[time_series_index]] = list(row[columns])

        # Create-tree phase
        good_leaf_nodes = list()
        bad_leaf_nodes = list()

        # root creation
        node = Node(level=1, group=time_series_dict, paa_value=paa_value)
        node.start_splitting(p_value=p_value, good_leaf_nodes, bad_leaf_nodes)

        # Recycle bad-leaves phase
        recycleBadLeaves(good_leaf_nodes, bad_leaf_nodes, p_value, paa_values)

        # Group formation phase


if __name__ == "__main__":
    if len(sys.argv) == 5:
        k_value = int(sys.argv[1])
        p_value = int(sys.argv[2])
        paa_value = int(sys.argv[3])
        dataset_path = sys.argv[4]
        if k_value > p_value:
            main_KAPRA(k_value=k_value, p_value=p_value, paa_value=paa_value, dataset_path=dataset_path)
        else:
            print("[*] Usage: python3 ./kp-anonymity.py k_value p_value dataset.csv")
            print("[*] k_value should be greater than p_value")
    else:
        print("[*] Usage: python3 ./kp-anonymity.py k_value p_value dataset.csv")