import os
import numpy as np
import pandas as pd
import sys
from loguru import logger
import random
from node import Node
from dataset_anonymized import DatasetAnonymized
max_level = 4

def clean_data(dataset_path_to_clean):
    """
    Print on file the dataset cleaned, in this case remove all columns normalized
    :param dataset_path_to_clean:
    :return:
    """
    time_series = pd.read_csv(dataset_path)
    time_series = time_series.loc[0:len(time_series), "Product_Code":"W51"]
    time_series.to_csv(dataset_path_to_clean.replace(".csv", "_Final.csv"), index=False)


def find_tuple_with_maximum_ncp(fixed_tuple, time_series, key_fixed_tuple, maximum_value, minimum_value):
    """
    By scanning all tuples once, we can find tuple t1 that maximizes NCP(fixed_tuple, t1)
    :param fixed_tuple:
    :param time_series:
    :param key_fixed_tuple:
    :return:
    """
    max_value = 0
    tuple_with_max_ncp = None
    for key, value in time_series.items():
        if key != key_fixed_tuple:
            ncp = compute_normalized_certainty_penalty_on_ai([fixed_tuple, time_series[key]], maximum_value, minimum_value)
            if ncp >= max_value:
                tuple_with_max_ncp = key
    return tuple_with_max_ncp


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


def compute_normalized_certainty_penalty_on_ai(table=None, maximum_value=None, minimum_value=None):
    """
    Compute NCP(T)
    :param table:
    :return:
    """
    z_1 = list()
    y_1 = list()
    a = list()
    for index_attribute in range(0, len(table[0])):
        temp_z1 = 0
        temp_y1 = float('inf')
        for row in table:
            if row[index_attribute] > temp_z1:
                temp_z1 = row[index_attribute]
            if row[index_attribute] < temp_y1:
                temp_y1 = row[index_attribute]
        z_1.append(temp_z1)
        y_1.append(temp_y1)
        a.append(abs(maximum_value[index_attribute] - minimum_value[index_attribute]))
    ncp_t = 0
    for index in range(0, len(z_1)):
        try:
            ncp_t += (z_1[index] - y_1[index]) / a[index]
        except ZeroDivisionError:
            ncp_t += 0
    ncp_T = len(table)*ncp_t
    return ncp_T


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


def get_list_min_and_max_from_table(table):
    """
    From a table get a list of maximum and minimum value of each attribut
    :param table:
    :return: list_of_minimum_value, list_of_maximum_value
    """
    attributes_maximum_value = table[0]
    attributes_minimum_value = table[0]

    for row in range(0, len(table)):
        for index_attribute in range(0, len(table[row])):
            if table[row][index_attribute] > attributes_maximum_value[index_attribute]:
                attributes_maximum_value[index_attribute] = table[row][index_attribute]
            if table[row][index_attribute] < attributes_minimum_value[index_attribute]:
                attributes_minimum_value[index_attribute] = table[row][index_attribute]

    return attributes_minimum_value, attributes_maximum_value


def minValueLossGroup(group_to_search, group_to_merge, index_ignored):
    p_group_min = {"index" : None, "group" : dict(), "vl" : float("inf")} 
    for index, group in enumerate(group_to_search):
        if index not in index_ignored: 
            vl = compute_instant_value_loss(list(group.values()) + list(group_to_merge.values()))
            if p_group_min["vl"] > vl:
                p_group_min["index"] = index; p_group_min["group"] = group; p_group_min["vl"] = vl
    return p_group_min["index"], p_group_min["group"]


def top_down_clustering(time_series=None, k_value=None, columns_list=None, maximum_value=None, minimum_value=None, time_series_k_anonymized=None, algorithm=None, tree=None, label='r'):
    """
    top down clustering similar to k-anonymity based on work of Xu et al. 2006,
    Utility-Based Anonymization for Privacy Preservation with Less Information Loss
    :param time_series:
    :param k_value:
    :return:
    """
    if len(time_series) < 2*k_value:
        time_series_k_anonymized.append(time_series)
        return
    else:
        keys = list(time_series.keys())
        rounds = 3

        # pick random tuple
        random_tuple = keys[random.randint(0, len(keys) - 1)]
        group_u = dict()
        group_v = dict()
        group_u[random_tuple] = time_series[random_tuple]
        del time_series[random_tuple]
        lastRow = random_tuple

        for round in range(0, rounds*2 - 1): 
            if len(time_series) > 0:
                if round % 2 != 0:
                    if algorithm == "naive":
                        u = find_tuple_with_maximum_ncp(group_v[last_row], time_series, last_row, maximum_value, minimum_value)
                    if algorithm == "kapra":
                        u = find_tuple_with_maximum_vl(group_v[last_row], time_series, last_row)
                    group_u.clear()
                    group_u[u] = time_series[u]
                    last_row = u
                    del time_series[u]
                else:
                    if algorithm == "naive":
                        v = find_tuple_with_maximum_ncp(group_u[last_row], time_series, last_row, maximum_value, minimum_value)
                    if algorithm == "kapra":
                        v = find_tuple_with_maximum_vl(group_u[last_row], time_series, last_row)
                    group_v.clear()
                    group_v[v] = time_series[v]
                    last_row = v
                    del time_series[v]

        # Now Assigned to group with lower uncertain penality
        index_keys_time_series = [x for x in range(0, len(list(time_series.keys())))]
        random.shuffle(index_keys_time_series)

        # add random row to group with lower NCP
        keys = [list(time_series.keys())[x] for x in index_keys_time_series]

        for key in keys:
            row_temp = time_series[key]
            group_u_values = list(group_u.values())
            group_v_values = list(group_v.values())
            group_u_values.append(row_temp)
            group_v_values.append(row_temp)
            
            if algorithm == 'kapra':
                vl_u = compute_instant_value_loss(group_u_values)
                vl_v = compute_instant_value_loss(group_v_values)

                if vl_v < vl_u:
                    group_v[key] = row_temp
                else:
                    group_u[key] = row_temp
                del time_series[key]

            elif algorithm == 'naive':
                ncp_u = compute_normalized_certainty_penalty_on_ai(group_u_values, maximum_value, minimum_value)
                ncp_v = compute_normalized_certainty_penalty_on_ai(group_v_values, maximum_value, minimum_value)

                if ncp_v < ncp_u:
                    group_v[key] = row_temp
                else:
                    group_u[key] = row_temp
                del time_series[key]

        if len(group_u) > k_value:
            # recursive partition group_u
            top_down_clustering(time_series=group_u, k_value=k_value, columns_list=columns_list,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_k_anonymized=time_series_k_anonymized, algorithm=algorithm, tree=tree, label=label+'a')
        else:
            time_series_k_anonymized.append(group_u)
            tree.append(label)

        if len(group_v) > k_value:
            # recursive partition group_v
            top_down_clustering(time_series=group_v, k_value=k_value, columns_list=columns_list,
                                          maximum_value=maximum_value, minimum_value=minimum_value,
                                          time_series_k_anonymized=time_series_k_anonymized, algorithm=algorithm, tree=tree, label=label+'b')
        else:
            time_series_k_anonymized.append(group_v)
            tree.append(label)

# POST-PROCESSING DA FARE
def postprocessing(time_series=None, k_value=None, columns_list=None, maximum_value=None, minimum_value=None, time_series_k_anonymized_postprocessed=None, algorithm=None, tree=None):


# DA RIVEDERE
def main_KAPRA(k_value=None, p_value=None, paa_value=None, dataset_path=None):
    if dataset_path.is_file():
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

        # creation root and start splitting node
        node = Node(level=1, group=time_series_dict, paa_value=paa_value)
        node.start_splitting(p_value, max_level, good_leaf_nodes, bad_leaf_nodes)

        # recycle bad-leaves phase
        suppressed_nodes = list()
        if(len(bad_leaf_nodes) > 0):
            Node.recycle_bad_leaves(p_value, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value)

        suppressed_nodes_list = list()
        for node in suppressed_nodes:
            suppressed_nodes_list.append(node.group) # suppressed nodes
        
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
                tree_structure = list()
                p_group_splitted = list()
                p_group_to_split = p_group
                
                # start top down clustering
                top_down_clustering(time_series=time_series_dict_copy, k_value=k_value, columns_list=columns, maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value, time_series_k_anonymized=time_series_k_anonymized, algorithm="kapra")

                # Postprocessing
                postprocessing(time_series=time_series_k_anonymized, k_value=k_value, columns_list=column, maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value, time_series_k_anonymized_postprocessed=time_series_postprocessed, algorithm='naive')
                time_series_k_anonymized = time_series_postprocessed                               

                p_group_to_add += time_series_postprocessed
                index_to_remove.append(index)

        p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]
        p_group_list += p_group_to_add

        k_group_list = list()
        index_to_remove = list() 
        
        # step 1
        for index, group in enumerate(p_group_list):
            if len(group) >= k_value:
                index_to_remove.append(index)
                k_group_list.append(group)
        
        p_group_list = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]

        # step 2 - 3 - 4
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
        
        # step 5
        p_group_remaining = [group for (index, group) in enumerate(p_group_list) if index not in index_to_remove ]
        
        for p_group in p_group_remaining:
            k_group, index_k_group = minValueLossGroup(group_to_search=k_group_list, group_to_merge=p_group)
            k_group_list.pop(index_k_group)
            k_group.update(p_group)
            k_group_list.append(k_group)

        #Finish group formation phase
        dataset_anonymized = DatasetAnonymized(pattern_anonymized_data=pattern_representation_dict, anonymized_data=k_group_list, suppressed_data=suppressed_nodes_list)
        dataset_anonymized.compute_anonymized_data()
        dataset_anonymized.save_on_file("./output_kapra.csv")


def main_Naive(k_value=None, p_value=None, paa_value=None, dataset_path=None):
    """
    :param k_value:
    :param p_value:
    :param dataset_path:
    :return:
    """
    if os.path.isfile(dataset_path):
        # read time_series_from_file
        time_series = pd.read_csv(dataset_path)

        # get columns name
        columns = list(time_series.columns)
        columns.pop(0)  # remove product code

        time_series_dict = dict()

        # save dict file instead pandas
        for index, row in time_series.iterrows():
            time_series_dict[row["Product_Code"]] = list(row["W0":"W51"])

        # save all maximum value for each attribute
        attributes_maximum_value, attributes_minimum_value = get_list_min_and_max_from_table(time_series_dict)

        # start k_anonymity_top_down
        time_series_k_anonymized = list()
        tree = list()

        top_down_clustering(time_series=time_series_dict, k_value=k_value, columns_list=columns, maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value, time_series_k_anonymized=time_series_k_anonymized, algorithm="naive", tree=tree)

        # start kp anonymity
        time_series_postprocessed = list()

        # Postprocessing
        postprocessing(time_series=time_series_k_anonymized, k_value=k_value, columns_list=column, maximum_value=attributes_maximum_value, minimum_value=attributes_minimum_value, time_series_k_anonymized_postprocessed=time_series_postprocessed, algorithm='naive', tree=tree)
        time_series_k_anonymized = time_series_postprocessed

        dataset_anonymized = DatasetAnonymized()
        for group in time_series_k_anonymized:
            # append group to anonymzed_data (after we will create a complete dataset anonymized)
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
        dataset_anonymized.save_on_file("./output_naive.csv")


if __name__ == "__main__":
    if len(sys.argv) == 5:
        algorithm = sys.argv[1]
        k_value = int(sys.argv[2])
        p_value = int(sys.argv[3])
        paa_value = int(sys.argv[4])
        dataset_path = sys.argv[5]
        if k_value > p_value:
            if algorithm == 'naive':
                main_Naive(k_value=k_value, p_value=p_value, paa_value=paa_value, dataset_path=dataset_path)
            elif algorithm == 'kapra':
                main_KAPRA(k_value=k_value, p_value=p_value, paa_value=paa_value, dataset_path=dataset_path)
        else:
            print("[*] Usage: python kp-anonymity.py k_value p_value paa_value dataset.csv")
            print("[*] k_value should be greater than p_value")
    else:
        print("[*] Usage: python kp-anonymity.py algorithm k_value p_value paa_value dataset.csv")