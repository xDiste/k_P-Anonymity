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


def top_down_clustering(p_subgroup, k_value, p_value):


def instant_value_loss(groupList):
    upperBoundList = list(); lowerBoundList = list()
    for i in range(0, len(groupList)):
        for j in range(0, len(groupList[i])):
            tmpLow = float('inf'); tmpMax = 0
            if groupList[i] > tmpMax:
                tmpMax = groupList[i]
            elif groupList[i] < tmpLow:
                tmpLow = groupList[i]
        upperBoundList.append(tmpMax)
        lowerBoundList.append(tmpLow)
    valueLossSum = 0	
    for i in range(0, len(groupList)):	
        valueLossSum += (pow((upperBoundList[i] - lowerBoundList[i]), 2) / len(groupList))	
    return np.sqrt(valueLossSum)


def group_with_minimum_instant_value_loss(group):
    p_group_min = dict()
    vl_tmp = float('inf')
    for g in group: 
        vl = compute_instant_value_loss(list(group.values()))
        if vl < vl_tmp:
            p_group_min = g; vl_tmp = vl
    return p_group_min


def groupFormation(good_leaf_nodes, k_value, p_value):
    '''
    for each P-subgroup that size >= 2*P do
        Split it by top-down clustering
    if any P-subgroup that size >= k then:
        add it into GL and remove it from PGL (p_subgroup)
    while |PGL| >= k do
        find s1 and G = s1
        while |G| < k do
            find s_min and add s_min into G
        Remove all P-subgroup in G from PGL and put G in GL
    for each remaining P-subgroup s' do
        Find corrisponding G' and add s' into G'
    '''
    p_subgroup = list()
    for node in good_leaf_nodes:
        p_subgroup.append(node)

    while len(p_subgroup) >= 2 * p_value:
        top_down_clustering(p_subgroup, k_value, p_value)
    
    GL_list = list(); group_to_remove = list()
    for i, group in p_subgroup:
        if group.size >= k_value:
            GL_list.append(group)
            group_to_remove.append(group)
    
    tmp_p_subgroup = list()
    for i, group in p_subgroup:
        for group not in group_to_remove:
            tmp_p_subgroup.append(group)

    p_subgroup = tmp_p_subgroup

    while len(p_subgroup) >= k_value:
        s1 = group_with_minimum_instant_value_loss(p_subgroup)
        while len(s1) < k_value:
            s_min = group_with_minimum_instant_value_loss(p_group + s1)
        
        tmp_p_subgroup = list()
        for i, group in p_subgroup:
            for group not in s1:
                tmp_p_subgroup.append(group)

        p_subgroup = tmp_p_subgroup
    
    for group in p_subgroup:
        s1 = minimum_instant_value_loss(p_subgroup)


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
        groupFormation(good_leaf_nodes, k_value, p_value)

        # Save all (to check if it's right)
        dataset_anonymized = DatasetAnonymized()
        for group in time_series_k_anonymized:
            dataset_anonymized.anonymized_data.append(group)

            good_leaf_nodes = list()
            bad_leaf_nodes = list()

            node = Node(level=1, group=group, paa_value=paa_value)
            node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)

            if len(bad_leaf_nodes) > 0:
                Node.postprocessing(good_leaf_nodes, bad_leaf_nodes)
                
            dataset_anonymized.pattern_anonymized_data.append(good_leaf_nodes)
        dataset_anonymized.compute_anonymized_data()
        dataset_anonymized.save_on_file("./Output/kapra_output.csv")


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