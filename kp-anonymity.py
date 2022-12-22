import os
import numpy as np
import pandas as pd
import sys
import random
from node import Node
from dataset_anonymized import DatasetAnonymized
from saxpy.paa import paa
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize

max_level = 4


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
    0) invariant: 
        - good_leaf_nodes: it must contain the leaves considered GOOD
        - bad_leaf_nodes: it must contain the leaves considered BAD
    '''
    if len(bad_leaf_nodes) == 0:
        return

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
                if patternRappresentation in nodeInCurrentLevel:
                    nodeInCurrentLevel[patternRappresentation].append(bad_node)
                else:
                    nodeInCurrentLevel[patternRappresentation] = [bad_node]
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
                pr = ts_to_string(data_paa,
                cuts_for_asize(current_level - 1)) if current_level - 1 > 1 else 'a' * paa_value
                bad_node.level = current_level - 1
                bad_node.pattern_representation = pr
        current_level -= 1

    '''
    10) suppress all time series contained in bad leaves
    '''
    bad_leaf_nodes = list()


def find_tuple_with_maximum_ivl(fixed_tuple, time_series, key_fixed_tuple):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes IVL(fixed_tuple, t1)
    """
    max_value = -float('inf')
    tuple_with_max_ivl = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            ivl = instant_value_loss([fixed_tuple, time_series[key]])
            if ivl >= max_value:
                tuple_with_max_ivl = key
                max_value = ivl
    return tuple_with_max_ivl


def subset_partition(time_series):
    '''
    partition T into two exclusive subsets T1 and T2 such that T1 and T2 are more local than T, and either T1 or T2 have at least k tuples
    Each si will be partitioned into new subgroups no smaller than P using a top-down partitioning method similar to the top-down greedy
    search algorithm proposed in [25]. This partitioning process is targeted at minimizing the total instant value loss in the partitions.
    '''
    keys = list(time_series.keys())
    rounds = 6  # By up to 6 rounds, we can achieve more than 98.75% of the maximal penalty.

    # pick random tuple
    random_tuple = keys[random.randint(0, len(keys) - 1)]
    group_u = dict()
    group_v = dict()
    group_u[random_tuple] = time_series[random_tuple]
    del time_series[random_tuple]
    last_row = random_tuple
    for round in range(0, rounds * 2 - 1):
        if len(time_series) > 0:
            if round % 2 == 0:
                v = find_tuple_with_maximum_ivl(group_u[last_row], time_series, last_row)
                group_v[v] = time_series[v]
                last_row = v
                del time_series[v]
            else:
                u = find_tuple_with_maximum_ivl(group_v[last_row], time_series, last_row)
                group_u[u] = time_series[u]
                last_row = u
                del time_series[u]

    index_keys_time_series = [x for x in range(0, len(list(time_series.keys())))]
    random.shuffle(index_keys_time_series)

    # add random row to group with lower IVL
    keys = [list(time_series.keys())[x] for x in index_keys_time_series]
    for key in keys:
        row_temp = time_series[key]
        group_u_values = list(group_u.values())
        group_v_values = list(group_v.values())
        group_u_values.append(row_temp)
        group_v_values.append(row_temp)

        ivl_u = instant_value_loss(group_u_values)
        ivl_v = instant_value_loss(group_v_values)

        if ivl_v < ivl_u:
            group_v[key] = row_temp
        else:
            group_u[key] = row_temp
        del time_series[key]
    return group_u, group_v


def top_down_clustering(time_series=None, p_value=None, time_series_k_anonymized=None):
    '''
    k-anonymity based on work of Xu et al. 2006,
    Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    NOTE: T means Timeseries
    Input: a T, parameter k, weights of attributes, hierarchies on categorical attributes;
    Output: a k-anonymous T
    1: IF |T| ≤ k THEN RETURN;
    2: ELSE {
        3: partition T into two exclusive subsets T1 and T2 such
        that T1 and T2 are more local than T, and either T1 or T2 have at least k tuples;
        4: IF |T1| > k THEN recursively partition T1;
        5: IF |T2| > k THEN recursively partition T2;
    }
    6: adjust the groups so that each group has at least k tuples;
    '''
    if len(time_series) <= 2 * p_value:
        time_series_k_anonymized.append(time_series)
        return
    else:
        t1, t2 = subset_partition(time_series)
        top_down_clustering(t1, p_value, time_series_k_anonymized)
        top_down_clustering(t2, p_value, time_series_k_anonymized)

def checkTuple(tuple):
    return not (any(x != x for x in tuple))

def instant_value_loss(groupList):
    boundsList = list()
    for index in range(len(groupList[0])):
        tmpList = list()
        for group in groupList:
            tmpList.append(group[index])
        if (checkTuple((max(tmpList), min(tmpList)))):
            boundsList.append((max(tmpList), min(tmpList)))
    return np.sqrt(sum((pow((upper - lower), 2) / len(boundsList)) for upper, lower in boundsList))

def compute_total_instant_value_loss(anonymized_result):
    VL_TOT = 0
    for d in anonymized_result:
        VL_TOT += instant_value_loss(list(d.values()))
    return VL_TOT


def group_with_minimum_instant_value_loss(main_group, merge_group=None):
    '''
    for each element in main_group:
        compute value loss
        find minimum value loss
    return the group with minimum value loss
    '''
    if merge_group is None:
        merge_group = dict()
    group_min = dict()
    min_value_loss = float('inf')
    for group in main_group:
        tmp_value_loss = instant_value_loss(list(group.values()) + list(merge_group.values()))
        if min_value_loss > tmp_value_loss:
            group_min = group
            min_value_loss = tmp_value_loss
    return group_min


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
    PGL = list()
    for node in good_leaf_nodes:
        PGL.append(node.group)

    # Init variables
    GL_list = list()
    store_time_series_k_anonymized = list()
    time_series_k_anonymized = list()

    '''
    1) for each P-subgroup that size >= 2*P do
    '''
    for i, group in enumerate(PGL):
        if len(group) >= 2 * p_value:
            '''
            2) Split it by top-down clustering
            '''
            top_down_clustering(group, p_value, time_series_k_anonymized)
            PGL.remove(group)
            store_time_series_k_anonymized = store_time_series_k_anonymized + time_series_k_anonymized

    '''
    consequence of 2) Add top-down clustering result to PGL
    '''
    for new_group in store_time_series_k_anonymized:
        PGL.append(new_group)

    '''
    3) if any P-subgroup that size >= k then:
    '''
    for group in PGL:
        if len(group) >= k_value:
            '''
            4) add it into GL and remove it from PGL
            '''
            GL_list.append(group)
            PGL.remove(group)

    '''
    5) while |PGL| >= k do
    '''
    while len(PGL) >= k_value:
        '''
        6) find s1 and G = s1
        Remove now G from PGL to simplify find group with minimum instant value loss (no needs to skip elements)
        '''
        G = group_with_minimum_instant_value_loss(PGL)
        PGL.remove(G)

        '''
        7) while |G| < k do
        '''
        while len(G) < k_value and len(PGL) > 0:
            '''
            8) find s_min and add s_min into G
            '''
            newGroup = group_with_minimum_instant_value_loss(main_group=PGL)
            G.update(newGroup)
            PGL.remove(newGroup)

        '''
        9) Remove all P-subgroup in G from PGL and put G in GL
        G already removed in step 6)
        '''
        for group in G:
            if group in PGL:
                PGL.remove(group)
        GL_list.append(G)

    '''
    10) for each remaining P-subgroup s' do
    '''
    for s_first in PGL:
        '''
        11) Find corrisponding G' and add s' into G'
        '''
        G_first = group_with_minimum_instant_value_loss(main_group=GL_list, merge_group=s_first)
        G_first.update(s_first)

    return GL_list


def main_KAPRA(k_value=None, p_value=None, paa_value=None, dataset_path=None):
    if os.path.isfile(dataset_path):
        # read time_series_from_file
        time_series = pd.read_csv(dataset_path)

        # get columns name
        columns = list(time_series.columns)
        time_series_index = columns.pop(0)  # remove product code

        time_series_dict = dict()

        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row[time_series_index]] = list(row[columns])

        # create-tree phase
        good_leaf_nodes = list()
        bad_leaf_nodes = list()

        # root creation
        node = Node(level=1, group=time_series_dict, paa_value=paa_value)
        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)

        # Recycle bad-leaves phase
        recycleBadLeaves(good_leaf_nodes, bad_leaf_nodes, p_value, paa_value)

        # Group formation phase
        anonymized_result = groupFormation(good_leaf_nodes, k_value, p_value)

        print("IVL: ", compute_total_instant_value_loss(anonymized_result))

        # Save all
        dataset_anonymized = DatasetAnonymized()
        for group in anonymized_result:
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
            print("[*] Usage: python3 ./kp-anonymity.py k_value p_value paa_value dataset.csv")
            print("[*] k_value should be greater than p_value")
    else:
        print("[*] Usage: python3 ./kp-anonymity.py k_value p_value paa_value dataset.csv")
