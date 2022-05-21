import os
import numpy as np
import pandas as pd
import sys
from loguru import logger
import random
from node import Node
from dataset_anonymized import DatasetAnonymized

max_level = 4

def find_tuple_with_maximum_vl(fixed_tuple, time_series, key_fixed_tuple):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes VL(fixed_tuple, t1)
    :param fixed_tuple:
    :param time_series:
    :param key_fixed_tuple:
    :return:
    """
    max_value = 0
    tuple_with_max_vl = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            vl = compute_instant_value_loss([fixed_tuple, time_series[key]])
            if vl >= max_value:
                tuple_with_max_vl = key
    return tuple_with_max_vl

def compute_instant_value_loss(table):
    """
    Compute VL(T)
    :param table:
    :return:
    """
    r_plus = list()
    r_minus = list()

    for index_attribute in range(0, len(table[0])):
        temp_r_plus = 0
        temp_r_minus = float('inf')
        for row in table:
            if row[index_attribute] > temp_r_plus:
                temp_r_plus = row[index_attribute]
            if row[index_attribute] < temp_r_minus:
                temp_r_minus = row[index_attribute]
        r_plus.append(temp_r_plus)
        r_minus.append(temp_r_minus)
    vl_t = 0
    for index in range(0, len(table[0])):
        vl_t += pow((r_plus[index] - r_minus[index]), 2)
    vl_t = np.sqrt(vl_t/len(table[0]))
    vl_T = len(table) * vl_t
    return vl_T


def minValueLossGroup(group_to_search=None, group_to_merge=dict(), index_ignored=list()):
    p_group_min = {"index" : None, "group" : dict(), "vl" : float("inf")} 
    for index, group in enumerate(group_to_search):
        if index not in index_ignored: 
            vl = compute_instant_value_loss(list(group.values()) + list(group_to_merge.values()))
            if p_group_min["vl"] > vl:
                p_group_min["index"] = index; p_group_min["group"] = group; p_group_min["vl"] = vl
    return p_group_min["group"], p_group_min["index"]


def top_down_clustering(time_series=None, k_value=None, maximum_value=None, minimum_value=None, time_series_clustered=None, algorithm=None, tree=None, label='r'):
    """
    top down clustering similar to k-anonymity based on work of Xu et al. 2006,
    Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    :param time_series:
    :param k_value:
    :return:
    """
    if len(time_series) < 2*k_value:
        time_series_clustered.append(time_series)
        tree.append(label)
        return
    else:
        keys = list(time_series.keys())
        rounds = 3

        # pick random tuple
        random_tuple = keys[random.randint(0, len(keys) - 1)]

        group_u = dict()
        group_v = dict()
        group_u[random_tuple] = time_series[random_tuple]

        last_row = random_tuple

        for round in range(0, rounds*2 - 1): 
            if len(time_series) > 0:
                if round % 2 == 0:
                    v = find_tuple_with_maximum_vl(group_u[last_row], time_series, last_row)
                    group_v.clear()
                    group_v[v] = time_series[v]
                    last_row = v
                else:
                    u = find_tuple_with_maximum_vl(group_v[last_row], time_series, last_row)
                    group_u.clear()
                    group_u[u] = time_series[u]
                    last_row = u

        # Now Assigned to group with lower uncertain penality
        index_keys_time_series = [index for (index, key) in enumerate(time_series) if key not in [u, v]]
        random.shuffle(index_keys_time_series)

        # add random row to group with lower NCP
        keys = [list(time_series.keys())[x] for x in index_keys_time_series]
        
        for key in keys:
            row_temp = time_series[key]
            group_u_values = list(group_u.values()); group_v_values = list(group_v.values())
            group_u_values.append(row_temp); group_v_values.append(row_temp)
            
            vl_u = compute_instant_value_loss(group_u_values)
            vl_v = compute_instant_value_loss(group_v_values)

            if vl_v < vl_u:
                group_v[key] = row_temp
            else:
                group_u[key] = row_temp
            del time_series[key]

        if len(group_u) > k_value:
            top_down_clustering(time_series=group_u, k_value=k_value, maximum_value=maximum_value, minimum_value=minimum_value, time_series_clustered=time_series_clustered, tree=tree, label=label+"a")
        else:
            time_series_clustered.append(group_u)
            tree.append(label)

        if len(group_v) > k_value:
            top_down_clustering(time_series=group_v, k_value=k_value, maximum_value=maximum_value, minimum_value=minimum_value, time_series_clustered=time_series_clustered, tree=tree, label=label+"b")
        else:
            time_series_clustered.append(group_v)
            tree.append(label)

def postprocessing(time_series_clustered=None, tree=None, k_value=None, maximum_value=None, minimum_value=None, time_series_postprocessed=None):
    newindex = list()
    newGroup = list()
    newTree = list()

    for index_group, g_group in enumerate(time_series_clustered):
        if len(g_group) < k_value:
            g_group_values = list(g_group.values())
            label = tree[index_group]
            index_neighbour = -1
            measure_neighbour = float('inf') 
            for index_label, label in enumerate(tree): 
                    if label[:-1] == label[:-1] and index_label != index_group and index_label not in newindex: 
                        index_neighbour = index_label
            
            if index_neighbour > 0:
                table_1 = g_group_values + list(time_series_clustered[index_neighbour].values())
                
                measure_neighbour = compute_instant_value_loss(table=table_1)

                group_merge_neighbour = dict()
                group_merge_neighbour.update(g_group)
                group_merge_neighbour.update(time_series_clustered[index_neighbour])

            measure_other_group = float('inf')   
            index_other_group = 0
            for index, other_group in enumerate(time_series_clustered): 
                if len(other_group) >= 2*k_value - len(g_group): #2k - |G|   
                    if index not in newindex:    
                        g_group_copy = g_group.copy()
                        for round in range(k_value - len(g_group)): #k - |G|         
                            round_measure = float('inf')
                            g_group_copy_values = list(g_group_copy.values())
                            for key, time_series in other_group.items(): 
                                if key not in g_group_copy.keys(): 
                                    temp_measure = compute_instant_value_loss(table=g_group_copy_values + [time_series])

                                    if temp_measure < round_measure:
                                        round_measure = temp_measure
                                        dict_to_add = { key : time_series }
                            
                            g_group_copy.update(dict_to_add)

                        if round_measure < measure_other_group:
                            measure_other_group = round_measure 
                            group_merge_other_group = g_group_copy
                            group_merge_remain = {key: value for (key, value) in other_group.items() if key not in g_group_copy.keys()} 
                            index_other_group = index

            if measure_neighbour < measure_other_group: 
                newindex.append(index_neighbour)
                newGroup.append(group_merge_neighbour)
                newTree.append(tree[index_neighbour][:-1]) 

            else:
                newindex.append(index_other_group)
                newGroup.append(group_merge_other_group)
                newGroup.append(group_merge_remain)
                newTree.append("")

            newindex.append(index_group)
    
    time_series_clustered = [group for (index, group) in enumerate(time_series_clustered) if index not in newindex ]
    time_series_clustered += newGroup 

    tree = [label for (index, label) in enumerate(tree) if index not in newindex]
    tree += newTree

    bad_group_count = 0
    for index, group in enumerate(time_series_clustered):
        if len(group) < k_value:
            bad_group_count +=1

    time_series_postprocessed += time_series_clustered
    
    if bad_group_count > 0:
        postprocessing(time_series_clustered=time_series_postprocessed, tree=tree, k_value=k_value, maximum_value=maximum_value, minimum_value=minimum_value)

def main_KAPRA(k_value=None, p_value=None, paa_value=None, dataset_path=None):
    if os.path.isfile(dataset_path):
        # read time_series_from_file
        time_series = pd.read_csv(dataset_path)

        # get columns name
        columns = list(time_series.columns)
        time_series_index = columns.pop(0)

        time_series_dict = dict()
        
        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row[time_series_index]] = list(row[columns])

        # create-tree phase
        good_leaf_nodes = list()
        bad_leaf_nodes = list()

        # creation root and start splitting node
        node = Node(level=1, group=time_series_dict, paa_value=paa_value)
        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)

        # recycle bad-leaves phase
        suppressed_nodes = list()
        if(len(bad_leaf_nodes) > 0):
            bad_leaf_nodes_dict = dict()
            for node in bad_leaf_nodes:
                if node.level in bad_leaf_nodes_dict.keys():
                    bad_leaf_nodes_dict[node.level].append(node)
                else:
                    bad_leaf_nodes_dict[node.level] = [node]

            bad_leaf_nodes_size = sum([node.size for node in bad_leaf_nodes])
            
            if bad_leaf_nodes_size >= p_value:
                current_level = max(bad_leaf_nodes_dict.keys())

                while bad_leaf_nodes_size >= p_value:
                    if current_level in bad_leaf_nodes_dict.keys():
                        merge_dict = dict()
                        keys_to_be_removed = list()
                        merge = False
                        for current_level_node in bad_leaf_nodes_dict[current_level]:
                            pr_node = current_level_node.pattern_representation
                            if pr_node in merge_dict.keys():
                                merge = True
                                merge_dict[pr_node].append(current_level_node)
                                if pr_node in keys_to_be_removed:
                                    keys_to_be_removed.remove(pr_node)
                            else:
                                merge_dict[pr_node] = [current_level_node]
                                keys_to_be_removed.append(pr_node)
                        
                        if merge:
                            for k in keys_to_be_removed:
                                del merge_dict[k]

                            for pr, node_list in merge_dict.items():
                                group = dict()
                                for node in node_list:
                                    bad_leaf_nodes_dict[current_level].remove(node)
                                    group.update(node.group)
                                if current_level > 1:
                                    level = current_level
                                else:
                                    level = 1
                                leaf_merge = Node(level=level, pattern_representation=pr,
                                    group=group, paa_value=paa_value)

                                if leaf_merge.size >= p_value:
                                    leaf_merge.label = "good-leaf"
                                    good_leaf_nodes.append(leaf_merge)
                                    bad_leaf_nodes_size -= leaf_merge.size
                                else: 
                                    leaf_merge.label = "bad-leaf"
                                    bad_leaf_nodes_dict[current_level].append(leaf_merge)

                    temp_level = current_level-1
                    for node in bad_leaf_nodes_dict[current_level]:
                        if temp_level > 1:
                            values_group = list(node.group.values())
                            data = np.array(values_group[0])
                            data_znorm = znorm(data)
                            data_paa = paa(data_znorm, paa_value)
                            pr = ts_to_string(data_paa, cuts_for_asize(temp_level))
                        else:
                            pr = "a"*paa_value
                        node.level = temp_level
                        node.pattern_representation = pr

                    if current_level > 0:
                        if temp_level not in bad_leaf_nodes_dict.keys():
                            bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict.pop(current_level)
                        else:
                            bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict[temp_level] + bad_leaf_nodes_dict.pop(current_level) 
                        current_level -= 1
                    else:
                        break 

            remaining_bad_leaf_nodes = list(bad_leaf_nodes_dict.values())[0]
            for node in remaining_bad_leaf_nodes:
                suppressed_nodes.append(node)
        
        # group formation phase
        # preprocessing
        pattern_representation_dict = dict() 
        p_group_list = list()

        for node in good_leaf_nodes: 
            p_group_list.append(node.group)
            pr = node.pattern_representation
            for key in node.group:
                pattern_representation_dict[key] = pr

        p_group_to_add = list()
        index_to_remove = list()

        for index, p_group in enumerate(p_group_list): 
            if len(p_group) >= 2*p_value:
                tree = list()
                p_group_splitted = list()
                p_group_to_split = p_group
                
                # start top down clustering
                top_down_clustering(time_series=p_group_to_split, k_value=p_value, time_series_clustered=p_group_splitted, tree=tree)
                
                # Postprocessing
                time_series_k_anonymized_postprocessed = list()
                postprocessing(time_series_clustered=p_group_splitted, tree=tree, k_value=p_value, time_series_postprocessed=time_series_k_anonymized_postprocessed)

                p_group_to_add += time_series_k_anonymized_postprocessed
                index_to_remove.append(index)

        p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove]
        p_group_list += p_group_to_add

        k_group_list = list()
        index_to_remove = list() 
        
        for index, group in enumerate(p_group_list):
            if len(group) >= k_value:
                index_to_remove.append(index)
                k_group_list.append(group)
        
        p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove]

        index_to_remove = list()
        p_group_list_size = sum([len(group) for group in p_group_list])
        
        while p_group_list_size >= k_value:
            k_group, index_min = minValueLossGroup(group_to_search=p_group_list, index_ignored=index_to_remove)
            index_to_remove.append(index_min)
            p_group_list_size -= len(k_group)

            while len(k_group) < k_value:
                group_to_add, index_group_to_add = minValueLossGroup(group_to_search=p_group_list, group_to_merge=k_group, index_ignored=index_to_remove)
                index_to_remove.append(index_group_to_add)
                k_group.update(group_to_add) 
                p_group_list_size -= len(group_to_add)
            k_group_list.append(k_group)   
        
        p_group_remaining = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove]
        
        for p_group in p_group_remaining:
            k_group, index_k_group = minValueLossGroup(group_to_search=k_group_list, group_to_merge=p_group)
            k_group_list.pop(index_k_group)
            k_group.update(p_group)
            k_group_list.append(k_group)

        dataset_anonymized = DatasetAnonymized()
        for group in k_group_list:
            # append group to anonymzed_data
            dataset_anonymized.anonymized_data.append(group)

            # good leaf nodes
            good_leaf_nodes = list()
            bad_leaf_nodes = list()

            # creation root and start splitting node
            node = Node(level=1, group=group, paa_value=paa_value)
            node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)

            if len(bad_leaf_nodes) > 0:
                Node.postprocessing(good_leaf_nodes, bad_leaf_nodes)

            dataset_anonymized.pattern_anonymized_data.append(good_leaf_nodes)

        dataset_anonymized.compute_anonymized_data()
        dataset_anonymized.save_on_file("./Output/output_kapra.csv")


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