import os
import numpy as np
import pandas as pd
import sys
import random
from node import Node
from dataset_anonymized import DatasetAnonymized


def recycleBadLeaves(good_leaf_nodes, bad_leaf_nodes, p_value, paa_value):
    '''
    1) current-level = max-bad-level
    2) while sum of all bad leaves' size >= P:
    3)      if any bad leaves can merge then:
    4)          merge them to a new node leaf-merge;
    5)          if leaf-merge.size >= P then:
    6)              leaf-merge.label = good-leaf
    7)          else
    8)              leaf-merge.label = bad-leaf
    9)      current-level--
    10) suppress all time series contained in bad leaves
    NB: Bad leaves can mearge means that a BL1 and BL2 (nodes in bad_leaf_nodes) have same level and same pattern rappresentation
    '''

    '''
    1) current-level = max-bad-level
    '''
    current_level = max([bad_node.level for bad_node in bad_leaf_nodes])

    '''
    2) while sum of all bad leaves' size >= P:
    Check on current_level > 0 to avoid errors
    '''
    while sum([bad_node.size for bad_node in bad_leaf_nodes]) >= p_value and current_level > 0:

        '''
        3) if any bad leaves can merge then:
        Select nodes with current level and save a Map of <pr, node>
        '''
        nodeInCurrentLevel = dict()
        for bad_node in bad_leaf_nodes:
            if bad_node.level == current_level:
                patternRappresentation = bad_node.pattern_representation
                nodeInCurrentLevel[patternRappresentation].append(bad_node) 
        
        '''
        4) merge them to a new node leaf-merge;
        '''
        for pr, nodeList in nodeInCurrentLevel.items():
            group = dict()
            for node in nodeList:
                bad_leaf_nodes.remove(node)
                group.update(node.group)
            leaf_merge = Node(level=current_level, pattern_representation=pr, group=group, paa_value=paa_value)

            '''
            5) if leaf-merge.size >= P then:
            '''
            if leaf_merge.size >= p_value:
                '''
                6) leaf-merge.label = good-leaf
                '''
                leaf_merge.label = "G"
                good_leaf_nodes.append(leaf_merge)
            else:
                '''
                8) leaf-merge.label = bad-leaf
                '''
                leaf_merge.label = "B"
                bad_leaf_nodes.append(leaf_merge)

        '''
        9) current-level--
        Decrease the level of all the bad-leaves
        '''
        for bad_node in bad_leaf_nodes:
            if bad_node.level == current_level:
                values_group = list(bad_node.group.values())
                data = np.array(values_group[0])
                data_znorm = znorm(data)
                data_paa = paa(data_znorm, paa_value)
                pr = ts_to_string(data_paa, cuts_for_asize(current_level-1)) if current_level-1 > 1 else 'a' * paa_value
                bad_node.level = current_level-1
                bad_node.pattern_representation = pr
        current_level -= 1

    '''
    10) suppress all time series contained in bad leaves
    '''
    bad_leaf_nodes = list()


def top_down_clustering(p_subgroup, k_value, p_value):
    return


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
    1) for each P-subgroup that size >= 2*P do
    2)      Split it by top-down clustering
    3) if any P-subgroup that size >= k then:
    4)      add it into GL and remove it from PGL
    5) while |PGL| >= k do
    6)      find s1 and G = s1
    7)      while |G| < k do
    8)          find s_min and add s_min into G
    9)      Remove all P-subgroup in G from PGL and put G in GL
    10) for each remaining P-subgroup s' do
    11)     Find corrisponding G' and add s' into G'
    '''

    '''
    0) Init
    '''
    # Create P-subgroup
    p_subgroup = list()
    for node in good_leaf_nodes:
        p_subgroup.append(node)

    # Init variables
    GL_list = list();
    tmp_p_subgroup = list()

    '''
    1) for each P-subgroup that size >= 2*P do
    '''
    while len(p_subgroup) >= 2 * p_value:
        '''
        2) Split it by top-down clustering
        '''
        top_down_clustering(p_subgroup, k_value, p_value)
    
    '''
    3) if any P-subgroup that size >= k then:
    '''
    for group in p_subgroup:
        if group.size >= k_value:
            '''
            4) add it into GL and remove it from PGL (p_subgroup)
            '''
            GL_list.append(group)
            p_subgroup.remove(group)
    
    '''
    5) while |PGL| >= k do
    '''
    while len(p_subgroup) >= k_value:
        '''
        6) find s1 and G = s1
        '''
        # TODO: Check
        G = group_with_minimum_instant_value_loss(p_subgroup)

        '''
        7) while |G| < k do
        '''
        while len(s1) < k_value:
            '''
            8) find s_min and add s_min into G
            '''
            # TODO: Check
            G.append(group_with_minimum_instant_value_loss(s1))
        
        '''
        9) Remove all P-subgroup in G from PGL and put G in GL
        '''
        for group in G:
            p_subgroup.remove(group)
        GL_list.append(G)

    '''
    10) for each remaining P-subgroup s' do
    '''
    for group in p_subgroup:
        '''
        11) Find corrisponding G' and add s' into G'
        '''
        # TODO: TBD + Check 
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