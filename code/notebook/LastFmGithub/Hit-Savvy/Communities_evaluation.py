import math
import sys
import collections
import seaborn as sns
import json
import scipy
from scipy.stats import skew
from scipy.stats import binned_statistic
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
from pylab import *
import collections
import sqlite3
import networkx as nx
import tqdm

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


def get_communities_info():
    louvain_file = "communities/louvain.json"
    infomap_file = "communities/infomap.json"
    label_propagation_file = "communities/labelpropagation.json"
    angel_file = "communities/angel.json"

    with open(louvain_file, 'r', encoding='utf-8') as infile:
        louvain = json.load(infile)
        infile.close()

    with open(infomap_file, 'r', encoding='utf-8') as infile:
        infomap = json.load(infile)
        infile.close()

    with open(label_propagation_file, 'r', encoding='utf-8') as infile:
        label_propagation = json.load(infile)
        infile.close()

    with open(angel_file, 'r', encoding='utf-8') as infile:
        angel = json.load(infile)
        infile.close()
    return louvain, infomap, label_propagation, angel


def print_communities_general_info():
    louvain, infomap, label_propagation, angel = get_communities_info()
    print("Genereal info:")
    str_to_print = "louvain: num_comm. = " + str(len(louvain["communities"])) + " , overlap = " +\
                   str(louvain["overlap"]) + ", coverage = " + str(louvain["coverage"]) + "largest community =" + \
                   str(len(max(louvain["communities"], key=len)))
    print(str_to_print)

    str_to_print = "infomap: num_comm. = " + str(len(infomap["communities"])) + " , overlap = " + \
                   str(infomap["overlap"]) + ", coverage = " + str(infomap["coverage"]) + "largest community =" + \
                   str(len(max(infomap["communities"], key=len)))
    print(str_to_print)

    str_to_print = "label_propagation: num_comm. = " + str(len(label_propagation["communities"])) + " , overlap = " + \
                   str(label_propagation["overlap"]) + ", coverage = " + str(label_propagation["coverage"]) + "largest community =" + \
                   str(len(max(label_propagation["communities"], key=len)))
    print(str_to_print)

    str_to_print = "angel: num_comm. = " + str(len(angel["communities"])) + " , overlap = " + \
                   str(angel["overlap"]) + ", coverage = " + str(angel["coverage"]) + "largest community =" + \
                   str(len(max(angel["communities"], key=len)))
    print(str_to_print)


def get_score(str_with_score):
    str_array = str_with_score.split(",")
    score_split = str_array[2].split("=")
    return float(score_split[1])


def get_community_internal_edge_density(algo_coms):
    """
        internal_edge_density = e_c / (n_c * (n_c - 1))
        with: e_c = # edges inside community C
              n_c = # nodes inside community C
    """
    communities = algo_coms["communities"]
    communities_internal_edge_density = []

    # I have to count the number of edges between the internal nodes inside the community c_list:
    # a) I load the graph
    # b) I check if exist a node between i, j in C, i != j
    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        n_c = len(c_list)
        denominator = n_c * (n_c - 1) / 2
        e_c = 0

        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])
        H = G.subgraph(c_list)
        internal_edge_density = nx.density(H)
        communities_internal_edge_density.append(internal_edge_density)


        """e_c = H.number_of_edges()
        try:
            internal_edge_density = e_c / denominator
        except ZeroDivisionError:
            internal_edge_density = 0.0
        communities_internal_edge_density.append(internal_edge_density)"""

    a = np.array(communities_internal_edge_density)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std,  communities_internal_edge_density


def get_community_in_degree(algo_coms):
    communities = algo_coms["communities"]
    communities_edges_inside = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])
        H = G.subgraph(c_list)
        in_degree_list = list(H.in_degree)
        for elem in in_degree_list:
            node = elem[0]
            in_deg = elem[1]
            communities_edges_inside.append(in_deg)

        # e_c = H.number_of_edges()
        # communities_edges_inside.append(e_c)

    print(communities_edges_inside)
    for elem in communities_edges_inside:
        if int(elem) > 1000:
            print("elem = " + str(elem))
    a = np.array(communities_edges_inside)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, communities_edges_inside


def get_community_out_degree(algo_coms):
    communities = algo_coms["communities"]
    communities_edges_inside = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])
        H = G.subgraph(c_list)
        out_degree_list = list(H.out_degree)
        for elem in out_degree_list:
            node = elem[0]
            out_deg = elem[1]
            communities_edges_inside.append(out_deg)

        # e_c = H.number_of_edges()
        # communities_edges_inside.append(e_c)

    print(communities_edges_inside)
    for elem in communities_edges_inside:
        if int(elem) > 1000:
            print("elem = " + str(elem))
    a = np.array(communities_edges_inside)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, communities_edges_inside


def get_community_edges_inside(algo_coms):
    communities = algo_coms["communities"]
    communities_edges_inside = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])

        H = G.subgraph(c_list)
        in_degree_list = list(H.in_degree)
        out_degree_list = list(H.out_degree)
        for i in range(0, len(in_degree_list)):
            communities_edges_inside.append((in_degree_list[i] + out_degree_list[i]))

    a = np.array(communities_edges_inside)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, communities_edges_inside


def get_community_clustering(algo_coms):
    communities = algo_coms["communities"]
    communities_edges_inside = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])
        H = G.subgraph(c_list)
        out_degree_list = list(H.out_degree)
        clust = nx.clustering(H)  # IS A DICT
        for key, value in clust.items():
            communities_edges_inside.append(value)

    a = np.array(communities_edges_inside)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, communities_edges_inside


def get_edges_that_points_from_community(algo_coms):
    """
        edges that points to community = b_c
        with: b_c = # edges that points outside community C
    """
    communities = algo_coms["communities"]
    communities_edges_outside = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        b_c = 0
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])

        for node in c_list:
            # get nodes's neighbours
            neighbors = [n for n in G[node]]

            for friend in neighbors:
                if int(friend) not in c_list:
                    # check edge direction
                    if G.has_edge(int(node), int(friend)):
                        b_c += 1

        communities_edges_outside.append(b_c)

    a = np.array(communities_edges_outside)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, communities_edges_outside


def get_edges_that_points_to_community(algo_coms):
    """
        edges that points to community = bb_c
        with: b_c = # edges outide community C
    """
    communities = algo_coms["communities"]
    communities_edges_outside = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        b_c = 0
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])

        for node in c_list:
            # get nodes's neighbours
            neighbors = [n for n in G[node]]

            for friend in neighbors:
                if int(friend) not in c_list:
                    # check edge direction
                    if G.has_edge(int(friend), int(node)):
                        b_c += 1

        communities_edges_outside.append(b_c)

    a = np.array(communities_edges_outside)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, communities_edges_outside


def get_community_conductance(algo_coms):
    """
        conductance = 2*|E_out_comm| / 2*|E_in_comm| + |E_out_comm|
    """
    communities = algo_coms["communities"]
    conductance = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        # convert nodes into integers
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])

        # get edges inside community
        H = G.subgraph(c_list)
        in_degree_list = list(H.in_degree)
        curr_comm_in_degree = 0
        for elem in in_degree_list:
            node = elem[0]
            in_deg = elem[1]
            curr_comm_in_degree += in_deg

        # get edges out of current community
        b_c = 0
        for node in c_list:
            # get nodes's neighbours
            neighbors = [n for n in G[node]]

            for friend in neighbors:
                if int(friend) not in c_list:
                    # check edge direction
                    if G.has_edge(int(friend), int(node)):
                        b_c += 1
        try:
            curr_comm_conductance = (2*(b_c)) / (2*(curr_comm_in_degree) + b_c)
        except ZeroDivisionError:
            curr_comm_conductance = 0.0
        conductance.append(curr_comm_conductance)

    a = np.array(conductance)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, conductance


def get_community_modularity(algo_coms):
    """
        modularity = sum_{degree_i} with i in C / |nodes_in_C|
    """
    communities = algo_coms["communities"]
    modularity = []

    G = nx.DiGraph()
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        G.add_edge(u, v)

    for c_list in tqdm.tqdm(communities):
        # convert nodes into integers
        for i in range(0, len(c_list)):
            c_list[i] = int(c_list[i])

        # get edges inside community
        H = G.subgraph(c_list)
        in_degree_list = list(H.in_degree)
        out_degree_list = list(H.out_degree)
        degree_sum = 0
        for elem in in_degree_list:
            node = elem[0]
            in_deg = elem[1]
            degree_sum += in_deg

        for elem in out_degree_list:
            node = elem[0]
            out_deg = elem[1]
            degree_sum += out_deg

        curr_comm_modularity = degree_sum / len(c_list)
        modularity.append(curr_comm_modularity)

    a = np.array(modularity)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, modularity


def get_communities_size(algo_coms):
    communities = algo_coms["communities"]
    communities_size = []
    for c_list in communities:
        communities_size.append(len(c_list))

    a = np.array(communities_size)
    min = np.min(a)
    max = np.max(a)
    score = np.mean(a)
    std = np.std(a)
    return min, max, score, std, communities_size


def violin_plot():
    louvain, infomap, label_propagation, angel = get_communities_info()
    # proprieties = ["size", "internal_edge_density", "in_degree", "out_degree", "edges_to_community",
    # "edges_from_community", "clustering", "conductance", "edges_inside", "edges_outside", "conductance", "modularity"]
    proprieties = ["edges_inside"]

    algorithms = ["louvain", "infomap", "label_propagation", "angel"]
    labels = [algorithms[0], algorithms[1], algorithms[2], algorithms[3]]
    colors = ["red", "green", "yellow", "blue"]
    x = []
    for p in proprieties:
        if p == "size":
            _, _, _, _, l_size = get_communities_size(louvain)
            _, _, _, _,  i_size = get_communities_size(infomap)
            _, _, _, _, lp_size = get_communities_size(label_propagation)
            min, max, score, std, a_size = get_communities_size(angel)

            print(p)
            print(get_score(louvain[p]))
            print(get_score(infomap[p]))
            print(get_score(label_propagation[p]))
            print(score)
        elif p == "internal_edge_density":
            min_l, max_l, score_l, std_l, l_size = get_community_internal_edge_density(louvain)
            min_i, max_i, score_i, std_i, i_size = get_community_internal_edge_density(infomap)
            min_lp, max_lp, score_lp, std_lp, lp_size = get_community_internal_edge_density(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_community_internal_edge_density(angel)

            print(p)
            print(score_l)
            print(score_i)
            print(score_lp)
            print(score_a)
        elif p == "in_degree":
            min_l, max_l, score_l, std_l, l_size = get_community_in_degree(louvain)
            min_i, max_i, score_i, std_i, i_size = get_community_in_degree(infomap)
            min_lp, max_lp, score_lp, std_lp, lp_size = get_community_in_degree(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_community_in_degree(angel)

            for i in range(0, len(l_size)):
               try:
                   l_size[i] = math.log10(l_size[i])
               except ValueError:
                   continue

            for i in range(0, len(lp_size)):
               try:
                   lp_size[i] = math.log10(lp_size[i])
               except ValueError:
                   continue
            for i in range(0, len(i_size)):
               try:
                   i_size[i] = math.log10(i_size[i])
               except ValueError:
                   continue

            for i in range(0, len(a_size)):
               try:
                   a_size[i] = math.log10(a_size[i])
               except ValueError:
                   continue

            print(p)
            print(score_l)
            print(score_i)
            print(score_lp)
            print(score_a)
        elif p == "out_degree":
            min_l, max_l, score_l, std_l, l_size = get_community_out_degree(louvain)
            min_i, max_i, score_i, std_i, i_size = get_community_out_degree(infomap)
            min_lp, max_lp, score_lp, std_lp, lp_size = get_community_out_degree(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_community_out_degree(angel)

            for i in range(0, len(l_size)):
               try:
                   l_size[i] = math.log10(l_size[i])
               except ValueError:
                   continue

            for i in range(0, len(lp_size)):
               try:
                   lp_size[i] = math.log10(lp_size[i])
               except ValueError:
                   continue
            for i in range(0, len(i_size)):
               try:
                   i_size[i] = math.log10(i_size[i])
               except ValueError:
                   continue

            for i in range(0, len(a_size)):
               try:
                   a_size[i] = math.log10(a_size[i])
               except ValueError:
                   continue

            print(p)
            print(score_l)
            print(score_i)
            print(score_lp)
            print(score_a)
        elif p == "clustering":
            min_l, max_l, score_l, std_l, l_size = get_community_clustering(louvain)
            min_i, max_i, score_i, std_i, i_size = get_community_clustering(infomap)
            min_lp, max_lp, score_lp, std_lp, lp_size = get_community_clustering(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_community_clustering(angel)

            print(p)
            print(score_l)
            print(score_i)
            print(score_lp)
            print(score_a)
        elif p == "edges_to_community":
            min_l, max_l, score_l, std_l, l_size = get_edges_that_points_to_community(louvain)
            min_i, max_i, score_i, std_i, i_size = get_edges_that_points_to_community(infomap)
            min_lp, max_lp, score_lp, std_lp, lp_size = get_edges_that_points_to_community(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_edges_that_points_to_community(angel)

            for i in range(0, len(l_size)):
                try:
                    l_size[i] = math.log10(l_size[i])
                except ValueError:
                    continue

            for i in range(0, len(lp_size)):
                try:
                    lp_size[i] = math.log10(lp_size[i])
                except ValueError:
                    continue
            for i in range(0, len(i_size)):
                try:
                    i_size[i] = math.log10(i_size[i])
                except ValueError:
                    continue

            for i in range(0, len(a_size)):
                try:
                    a_size[i] = math.log10(a_size[i])
                except ValueError:
                    continue

            print(p)
            print(score_l)
            print(score_i)
            print(score_lp)
            print(score_a)
        elif p == "edges_from_community":
            min_l, max_l, score_l, std_l, l_size = get_edges_that_points_from_community(louvain)
            min_i, max_i, score_i, std_i, i_size = get_edges_that_points_from_community(infomap)
            min_lp, max_lp, score_lp, std_lp, lp_size = get_edges_that_points_from_community(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_edges_that_points_from_community(angel)

            for i in range(0, len(l_size)):
                try:
                    l_size[i] = math.log10(l_size[i])
                except ValueError:
                    continue

            for i in range(0, len(lp_size)):
                try:
                    lp_size[i] = math.log10(lp_size[i])
                except ValueError:
                    continue
            for i in range(0, len(i_size)):
                try:
                    i_size[i] = math.log10(i_size[i])
                except ValueError:
                    continue

            for i in range(0, len(a_size)):
                try:
                    a_size[i] = math.log10(a_size[i])
                except ValueError:
                    continue

            print(p)
            print(score_l)
            print(score_i)
            print(score_lp)
            print(score_a)
        elif p == "edges_inside":
            _, _, _, _, l_size = get_community_edges_inside(louvain)
            _, _, _, _, i_size = get_community_edges_inside(infomap)
            _, _, _, _, lp_size = get_community_edges_inside(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_community_edges_inside(angel)

            print(p)
            print(get_score(louvain["edges_inside"]))
            print(get_score(infomap["edges_inside"]))
            print(get_score(label_propagation["edges_inside"]))
            print(score_a)
        elif p == "edges_outside":
            min_l, max_l, score_l, std_l, l_size = get_community_edges_outside(louvain)
            min_i, max_i, score_i, std_i, i_size = get_community_edges_outside(infomap)
            min_lp, max_lp, score_lp, std_lp, lp_size = get_community_edges_outside(label_propagation)
            min_a, max_a, score_a, std_a, a_size = get_community_edges_outside(angel)

            for i in range(0, len(i_size)):
               try:
                   i_size[i] = math.log10(i_size[i])
               except ValueError:
                   continue

            for i in range(0, len(lp_size)):
               try:
                   lp_size[i] = math.log10(lp_size[i])
               except ValueError:
                   continue

            print(p)
            print(score_l)
            print(score_i)
            print(score_lp)
            print(score_a)

        elif p == "conductance":
            _, _, _, _, l_size = get_community_conductance(louvain)
            _, _, _, _,  i_size = get_community_conductance(infomap)
            _, _, _, _, lp_size = get_community_conductance(label_propagation)
            min, max, score, std, a_size = get_community_conductance(angel)

            print(p)
            print(get_score(louvain[p]))
            print(get_score(infomap[p]))
            print(get_score(label_propagation[p]))
            print(score)

        elif p == "modularity":
            _, _, _, l_score, l_size = get_community_modularity(louvain)
            _, _, _, i_score,  i_size = get_community_modularity(infomap)
            _, _, _, lp_score, lp_size = get_community_modularity(label_propagation)
            min, max, score, std, a_size = get_community_modularity(angel)

            print(p)
            print(l_score)
            print(i_score)
            print(lp_score)
            print(score)


        fig, ax = plt.subplots()
        # ax = sns.violinplot(x="algorithms", y=p, data=df_x)
        violin_parts = ax.violinplot([l_size, i_size, lp_size, a_size], showmeans=False, showextrema=False, showmedians=False,)

        # plt.yscale('log')

        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)

        i = 0
        for pc in violin_parts['bodies']:
            pc.set_facecolor(colors[i])
            i += 1
            pc.set_edgecolor('black')

        plt.xlabel("communities")
        plt.ylabel(p)
        plt.show()


def get_largest_community(community_lists):
    return len(max(community_lists["communities"], key=len))

def retrieve_ht_community_purity_and_size(adopters, clust_res_file, cd_algo):

    with open(clust_res_file, 'r', encoding='utf-8') as infile:
        clus_dict = json.load(infile)
        infile.close()

    cd_algo_largest_community = get_largest_community(cd_algo)
    global_cd_algo_purities = []
    global_cd_algo_size = []

    # retrieve each community
    for c_list in cd_algo["communities"]:

        current_cd_list_purities = {}
        for key in list(clus_dict["cluster_members"].keys()):
            current_cd_list_purities[str(key)] = 0

        how_many_reps = 0
        # iterate over each communitie's member
        for ad in c_list:
            if str(ad) in adopters:  # check if ad is a Hit-Savvie
                how_many_reps += 1
            try:
                cluster_membership = clus_dict["new_labels"][str(ad)]
            except KeyError:
                # default
                cluster_membership = 0
            current_cd_list_purities[str(cluster_membership)] += 1

        # sort dict by decreasing value
        sorted_current_cd_list_purities = {k: v for k, v in
                                           sorted(current_cd_list_purities.items(), key=lambda x: x[1],
                                                  reverse=True)}
        great_clust = list(sorted_current_cd_list_purities.keys())[0]
        great_counter = list(sorted_current_cd_list_purities.values())[0]

        purity = float(great_counter / len(c_list))

        while how_many_reps > 0:
            global_cd_algo_purities.append(purity)
            global_cd_algo_size.append(len(c_list))
            how_many_reps -= 1

    return global_cd_algo_purities, global_cd_algo_size



def get_hitsavvies_purity_in_communities(clust_res_file, dir_name, hit_or_flop):
    louvain, infomap, label_propagation, angel = get_communities_info()

    cd_algorithms = []
    cd_algorithms.append(louvain)
    cd_algorithms.append(infomap)
    cd_algorithms.append(label_propagation)
    cd_algorithms.append(angel)

    for i in range(0, len(cd_algorithms)):
        cd_algo = cd_algorithms[i]

        multi_purity = []
        multi_size = []

        # retrieve each success definiton HitSavvies
        for success_def in range(1, 8):
            complete_dir_name = "Def" + str(success_def) + dir_name

            if hit_or_flop == "hit":
                file_name = complete_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
            else:
                file_name = complete_dir_name + "flop_adopters_success_def" + str(success_def) + ".json"

            with open(file_name, 'r', encoding='utf-8') as infile:
                info_dict = json.load(infile)

            adopters = [str(x) for x in list(info_dict.keys())]
            purity, size = retrieve_ht_community_purity_and_size(adopters, clust_res_file, cd_algo)

            multi_purity.append(purity)
            multi_size.append(size)

        labels = ["seed-user listeners", "seed-user playcounts", "google trends", "last.fm listeners", "last.fm playcounts",
                  "AT", "actions"]
        if i == 0:
            x_label = "Louvain purity"
            x2_label = "Louvain size"
        elif  i == 1:
            x_label = "Infopap purity"
            x2_label = "Infomap size"
        elif i == 2:
            x_label = "Label Propagation purity"
            x2_label = "Label Propagation size"
        elif i == 3:
            x_label = "Angel purity"
            x2_label = "Angel size"
        draw_multi_histogram(multi_purity, x_label, "# Hit-Savvies", labels)
        draw_multi_histogram(multi_size, x2_label, "# Hit-Savvies", labels)



def draw_multi_histogram(multi_x, x_label, y_label, labels):
    fig, ax1 = plt.subplots()
    # y axis is logarithmic
    n, bins_edges, patches = ax1.hist(multi_x, log= True, label=labels)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # add extra y tick
    y_axis_values = list(plt.yticks()[0])
    del y_axis_values[0]
    del y_axis_values[0]
    plt.yticks(y_axis_values)

    plt.legend()
    plt.show()
    plt.close("all")


def draw_purity_scatter(global_cd_algo_size, global_cd_algo_purities, x2_label, y_label, color):
    fig, ax1 = plt.subplots()
    ax1.scatter(global_cd_algo_size, global_cd_algo_purities, color=color)

    plt.xlabel(x2_label)
    plt.ylabel(y_label)

    # add extra y tick
    y_axis_values = list(plt.yticks()[0])
    del y_axis_values[0]
    # del y_axis_values[0]
    plt.yticks(y_axis_values)

    plt.show()
    plt.close("all")


def compute_purity_using_profile_clustering(clust_res_file):
    louvain, infomap, label_propagation, angel = get_communities_info()

    cd_algorithms = []
    cd_algorithms.append(louvain)
    cd_algorithms.append(infomap)
    cd_algorithms.append(label_propagation)
    cd_algorithms.append(angel)

    with open(clust_res_file, 'r', encoding='utf-8') as infile:
        clus_dict = json.load(infile)
        infile.close()

    for i in range(0, len(cd_algorithms)):
        cd_algo = cd_algorithms[i]
        cd_algo_largest_community = get_largest_community(cd_algo)
        global_cd_algo_purities = []
        global_cd_algo_size = []

        # retrieve each community
        for c_list in cd_algo["communities"]:

            current_cd_list_purities = {}
            for key in list(clus_dict["cluster_members"].keys()):
                current_cd_list_purities[str(key)] = 0

            # iterate over each communitie's member
            for ad in c_list:
                try:
                    cluster_membership = clus_dict["new_labels"][str(ad)]
                except KeyError:
                    # default
                    cluster_membership = 0

                current_cd_list_purities[str(cluster_membership)] += 1

            # sort dict by decreasing value
            sorted_current_cd_list_purities = {k: v for k,v in sorted(current_cd_list_purities.items(), key=lambda x: x[1], reverse=True)}
            great_clust = list(sorted_current_cd_list_purities.keys())[0]
            great_counter = list(sorted_current_cd_list_purities.values())[0]

            purity = float(great_counter / len(c_list))
            global_cd_algo_purities.append(purity)
            global_cd_algo_size.append(len(c_list))

            if len(c_list) == cd_algo_largest_community:
                print("larger community is = " + str(purity) + " pure")
                print()

        if i == 0:
            x_label = "%louvain_communities"
            x2_label = "louvain_sizes"
            color = "fuchsia"
        elif i == 1:
            x_label = "%infomap_communities" + str()
            x2_label = "infomap_sizes"
            color = "green"
        elif i == 2:
            x_label = "%label_propagation_communities" + str()
            x2_label = "label_propagation_sizes"
            color = "blue"
        elif i == 3:
            x_label = "%angel_communities" + str()
            x2_label = "angel_sizes"
            color = "red"
        # draw_purity_hist(global_cd_algo_purities, x_label, "purity", color)
        draw_purity_scatter(global_cd_algo_size, global_cd_algo_purities, x2_label, "purity", color)


print_communities_general_info()
violin_plot()
get_hitsavvies_purity_in_communities("LPDM_clust_res.json", "/part2/", "hit")
compute_purity_using_profile_clustering()
compute_purity_using_profile_clustering("LPDM_clust_res.json")

