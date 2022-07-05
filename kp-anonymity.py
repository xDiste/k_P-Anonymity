import os
import numpy as np
import pandas as pd
import sys
from loguru import logger
import random
from node import Node
from dataset_anonymized import DatasetAnonymized

max_level = 4

def instantValueLoss(table=None):
    maxRow = list(); minRow = list()
    for index in range(0, len(table[0])):
        maxRowTemp = 0; minRowTemp = float('inf')
        for row in table:
            if row[index] < minRowTemp:
                minRowTemp = row[index]
            if row[index] > maxRowTemp:
                maxRowTemp = row[index]
        if minRowTemp != float('inf'):
            maxRow.append(maxRowTemp); minRow.append(minRowTemp)
    valueLossSum = 0
    for index in range(0, len(maxRow)):
        valueLossSum += pow((maxRow[index] - minRow[index]), 2)
    return np.sqrt(valueLossSum/len(table[0])) * len(table)


# Find the tuple with maximum value loss
def maxValueLossTuple(fixed_tuple=None, time_series=None, key_fixed_tuple=None):
    maxValue = 0
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            valueLoss = instantValueLoss([fixed_tuple, time_series[key]])
            if valueLoss >= maxValue:
                maxValueLossTuple = key
    return maxValueLossTuple


# Find the group with minimum value loss
def minValueLossGroup(group_to_find=None, group_to_merge=dict(), index_ignored=list()):
    minGroup = dict()
    minGroupIndex = None
    minValueLoss = float("inf")
    for count, group in enumerate(group_to_find):
        if count not in index_ignored: 
            tmpValueLoss = instantValueLoss(list(group.values()) + list(group_to_merge.values()))
            if minValueLoss > tmpValueLoss:
                minGroupIndex = count; minGroup = group; minValueLoss = tmpValueLoss
    return minGroup, minGroupIndex


def top_down_clustering(time_series=None, k_value=None, time_series_clustered=None, algorithm=None, tree=None, label=''):
    """
    k-anonymity based on work of Xu et al. 2006,
    Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    """
    if len(time_series) < 2*k_value:
        time_series_clustered.append(time_series)
        tree.append(label)
        return
    else:
        keys = list(time_series.keys())
        rounds = 3

        # Pick a random row
        randomRow = keys[random.randint(0, len(keys) - 1)]

        group_u = dict(); group_v = dict()
        group_u[randomRow] = time_series[randomRow]

        for round in range(0, rounds*2 - 1): 
            if len(time_series) > 0:
                if round % 2 == 0:
                    v = maxValueLossTuple(group_u[randomRow], time_series, randomRow)
                    group_v.clear()
                    group_v[v] = time_series[v]
                    randomRow = v
                else:
                    u = maxValueLossTuple(group_v[randomRow], time_series, randomRow)
                    group_u.clear()
                    group_u[u] = time_series[u]
                    randomRow = u

        time_series_keyIndex = list()
        for (index, key) in enumerate(time_series):
            if key not in [u, v]:
                time_series_keyIndex.append(index)

        random.shuffle(time_series_keyIndex)    # I shuffle the indexes so that I don't always have the same result

        keys = list()
        for i in time_series_keyIndex:
            keys.append(list(time_series.keys())[i])
        
        for key in keys:
            tempRow = time_series[key]
            group_u_values = list(group_u.values()); group_v_values = list(group_v.values())
            group_u_values.append(tempRow); group_v_values.append(tempRow)
            
            valueLoss_group_u = instantValueLoss(group_u_values); valueLoss_group_v = instantValueLoss(group_v_values)

            if valueLoss_group_v < valueLoss_group_u:
                group_v[key] = tempRow
            else:
                group_u[key] = tempRow
            del time_series[key]

        if len(group_u) > k_value:
            top_down_clustering(time_series=group_u, k_value=k_value, time_series_clustered=time_series_clustered, tree=tree, label=label+'a')
        else:
            time_series_clustered.append(group_u)
            tree.append(label)

        if len(group_v) > k_value:
            top_down_clustering(time_series=group_v, k_value=k_value, time_series_clustered=time_series_clustered, tree=tree, label=label+'b')
        else:
            time_series_clustered.append(group_v)
            tree.append(label)


def postprocessing(time_series_clustered=None, k_value=None, time_series_postprocessed=None, tree=None):
    indexes = list(); newGroup = list(); newTree = list()
    for i, group in enumerate(time_series_clustered):
        if len(group) < k_value:
            groupValues = list(group.values())
            label = tree[i]
            neighbourIndex = -1
            neighbourValueLoss = float('inf') 
            for count, label in enumerate(tree): 
                    if label[:-1] == label[:-1] and count != i and count not in indexes: 
                        neighbourIndex = count
            
            neighbourGroup = dict()
            if neighbourIndex > 0:
                t = groupValues + list(time_series_clustered[neighbourIndex].values())
                neighbourValueLoss = instantValueLoss(t)
                neighbourGroup.update(group)
                neighbourGroup.update(time_series_clustered[neighbourIndex])

            otherGroupValueLoss = float('inf')   
            for j, otherGroup in enumerate(time_series_clustered): 
                # 2k - |G|
                if len(otherGroup) >= (2*k_value - len(group)):
                    if j not in indexes:
                        groupCopy = group.copy()
                        # k - |G|
                        for round in range(k_value - len(group)):
                            roundValueLoss = float('inf')
                            groupCopyValues = list(groupCopy.values())
                            newDict = {}
                            for key, time_series in otherGroup.items():
                                if key not in groupCopy.keys():
                                    tmpValueLoss = instantValueLoss(groupCopyValues + [time_series])
                                    if tmpValueLoss < roundValueLoss:
                                        roundValueLoss = tmpValueLoss
                                        newDict = {key: time_series}

                            if len(newDict) != 0:
                                groupCopy.update(newDict)

                        otherGroupIndex = -1
                        if roundValueLoss < otherGroupValueLoss:
                            otherGroupValueLoss = roundValueLoss
                            remainGroup = dict()
                            for (key, value) in otherGroup.items():
                                if key not in groupCopy.keys():
                                    remainGroup.update({key: value})
                            otherGroupIndex = j

            if neighbourValueLoss < otherGroupValueLoss:
                indexes.append(neighbourIndex)
                newGroup.append(neighbourGroup)
                newTree.append(tree[neighbourIndex][:-1])
            elif otherGroupIndex != -1:
                indexes.append(otherGroupIndex)
                newGroup.append(groupCopy)
                newGroup.append(remainGroup)
                newTree.append('')
            indexes.append(i)
    
    new_time_series_clustered = list()
    for (index, group) in enumerate(time_series_clustered):
        if index not in indexes:
            new_time_series_clustered.append(group)
    new_time_series_clustered += newGroup 

    tmpTree = list()
    for (index, label) in enumerate(tree):
        if index not in indexes:
            tmpTree.append(label)
    tmpTree += newTree

    bad_group_count = 0
    for index, group in enumerate(new_time_series_clustered):
        if len(group) < k_value:
            bad_group_count += 1
    time_series_postprocessed += new_time_series_clustered
    
    if bad_group_count > 0:
        postprocessing(time_series_clustered=time_series_postprocessed, k_value=k_value, tree=tmpTree)


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
                                    leaf_merge = Node(level=level, pattern_representation=pr, group=group, paa_value=paa_value)

                                if leaf_merge.size >= p_value:
                                    leaf_merge.label = "G" # Good Leaf
                                    good_leaf_nodes.append(leaf_merge)
                                    bad_leaf_nodes_size -= leaf_merge.size
                                else: 
                                    leaf_merge.label = "B" # Bad Leaf
                                    bad_leaf_nodes_dict[current_level].append(leaf_merge)

                    temp_level = current_level-1
                    for node in bad_leaf_nodes_dict[current_level]:
                        if temp_level > 1:
                            values_group = list(node.group.values())
                            # To reduce dimensionality
                            data = np.array(values_group[0])
                            data_znorm = znorm(data)
                            data_paa = paa(data_znorm, paa_value)
                            pr = ts_to_string(data_paa, cuts_for_asize(temp_level))
                        else:
                            pr = 'a' * paa_value
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
                postprocessing(time_series_clustered=p_group_splitted, k_value=p_value, time_series_postprocessed=time_series_k_anonymized_postprocessed, tree=tree)

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
            k_group, index_min = minValueLossGroup(group_to_find=p_group_list, index_ignored=index_to_remove)
            index_to_remove.append(index_min)
            p_group_list_size -= len(k_group)

            while len(k_group) < k_value:
                group_to_add, index_group_to_add = minValueLossGroup(group_to_find=p_group_list, group_to_merge=k_group, index_ignored=index_to_remove)
                index_to_remove.append(index_group_to_add)
                k_group.update(group_to_add) 
                p_group_list_size -= len(group_to_add)
            k_group_list.append(k_group)   
        
        p_group_remaining = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove]
        
        for p_group in p_group_remaining:
            k_group, index_k_group = minValueLossGroup(group_to_find=k_group_list, group_to_merge=p_group)
            k_group_list.pop(index_k_group)
            k_group.update(p_group)
            k_group_list.append(k_group)

        VL_TOT = 0
        for d in k_group_list:
            VL_TOT += instantValueLoss(list(d.values()))
        print("IVL", VL_TOT)
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
