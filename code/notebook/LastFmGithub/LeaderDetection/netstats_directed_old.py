import os
import sys, time
import zipfile
import math
import sys
import collections
from array import array

import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import skew
from scipy.stats import binned_statistic
import seaborn as sns
import json
import pandas as pd
from collections import Counter
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from Polynomial import Polynomial
import scipy.sparse
import scipy.sparse.csgraph

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')

import networkx as nx

__author__ = 'Giulio Rossetti', 'Alexandra Bradan'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com", "alexandrabradan@gmail.com"


def create_directory(path):
    """
   Create a new directory in the specified path
   :param path: path where to create the new directory
   """
    os.makedirs(path)


def check_if_directory_exist(path):
    """
       Check if a directory exist
       :param path: directory's path to check existance
       :return: 0 if the directory exist and refers to a directory path
                -1 otherwiise

    """
    check = os.path.exists(path) and os.path.isdir(path)
    if check is True:
        return 0
    else:
        return -1


def init():
    """
        Function which build the graph from the  "network_filtered_int1.edgelist" file.
        The function also create the following files:
        1. "graph_stats_global" => file which contains the global statistics of the graph
        2. "graph_stats_local" => file which contains the local statistics of the graph (relevant of every node)
    """
    netstats_directory = "netstats_directed_old/"
    check0 = check_if_directory_exist(netstats_directory)

    if check0 == -1:
        create_directory("netstats_directed_old/")

    G = nx.DiGraph()  # directed graph
    start = time.time()  # starting time

    print("Loading the social graphload")

    # populate graph with "networks.zip/network_filtered_int1.edgelist" file's lines
    with zipfile.ZipFile('networks.zip') as z:
        for filename in z.namelist():
            if filename == "network_filtered_int1.edgelist":
                with z.open(filename, 'r') as f:

                    for line in f:
                        fields = line.strip().decode('utf-8').split()
                        G.add_edge(int(fields[0]), int(fields[1]))
                    f.close()

    sys.stderr.write("Data load! Runtime: %s\n" % (time.time() - start))

    # G.remove_edges_from(G.selfloop_edges())  # remove self-loops

    # collect all the leaders founded
    final_leaders = get_final_leaders_founded()

    # get node's degree, in_degree, out_degree and relative global frequency
    # degree_study(G, start, final_leaders)

    # get social graph's macroscopic measurements
    # macroscopic_measurements(G, start, final_leaders)

    # get social graph's microscopic measurements
    # microscopic_measurements(G, start, final_leaders)

    avg_clusterings = nx.clustering(G)  # IS A DICT
    x = avg_clusterings.values()
    draw_in_degree(x, "Clustering", "# Nodes", [])


def get_final_leaders_founded():
    """
        Function which retrieve the unicoal leaders found and present in the
        "univocal_final_leaders" file
    """
    final_leaders = []

    univocal_final_leaders_file = open("univocal_final_leaders", "r")
    for line in univocal_final_leaders_file:
        user_encoding = int(line.replace("\n", ""))
        final_leaders.append(user_encoding)
    univocal_final_leaders_file.close()

    return final_leaders


def get_doane_num_bin_and_binwidth(x):
    """
        Function which computes the number of bins and the binwidth of an histogram,
         calculated with the Doane formula
    :param x: data array of values to plot
    :return: num_bins, binwidth
    """
    N = len(x)  # number of observations
    min_x = min(x)  # min observation value
    max_x = max(x)  # max observation value

    # skewness: mean((x - mu) / psi)**3
    g = skew(x)
    # skewness's standard deviation: math.sqrt((6*(N - 2)) / ((N + 1)*(N+3)))
    psi_g = math.sqrt((6*(N - 2) / ((N + 1)*(N + 3))))
    # number of bins = 1 + log2(N) + log2(1 + g / psi_g)  [Doane formula]
    num_bins = np.round(np.ceil(1 + math.log2(N) + math.log2(1 + (g + abs(g) / psi_g))))
    # binwidth = ceil(max(x) - min(x) / num_bins)
    range = max_x - min_x
    binwidth = np.round(np.ceil(range / num_bins))

    return num_bins, binwidth

def third_polinomial(x, a, b, c, d):
    return a*x**3 +b*x**2 + c*x + d

def powlaw(x, a, b, c):
    """Fit data to a power law with weights according to a y log scale"""
    return a * np.exp(-b * x) + c

def draw_in_degree(x, x_label, y_label, global_bin_edges):

    # draw out-degree
    fig, ax1 = plt.subplots()

    # print bin's numbers and bins's size
    # num_bins, binwidth = get_doane_num_bin_and_binwidth(x)
    # print("num_bins = " + str(num_bins))
    # print("binwidth = " + str(binwidth))

    if len(global_bin_edges) != 0:
        n, bin_edges, patches = ax1.hist(x, log=True, bins=global_bin_edges, color='red', width=binwidth / 2)
        print("num_bins = " + str(len(n)))
    else:
        n, bin_edges, patches = ax1.hist(x, log=True, bins='doane', color='red', width=binwidth / 2)
        print("num_bins = " + str(len(n)))


    fig, ax2 = plt.subplots(sharex=True, sharey=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    ax2.errorbar(bin_centres, n, fmt='ro')
    ax2.set_yscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    # I have to extract bin's y values in order to fit the scatter data
    c = Counter(x)
    c_sorted = sorted(c.items())
    print(c_sorted)

    xdata = []
    ydata = []
    for key, value in c_sorted:
        xdata.append(key)
        ydata.append(value)
    print(xdata)
    print(ydata)

    # I have to sum all y values falling in the same bin
    new_ydata = []
    for j in range(0, len(bin_edges) - 1):
        tmp_sum = 0
        found_bin_of_appartenence = False
        for i in range(0, len(xdata)):
            if float(xdata[i]) >= bin_edges[j] and float(xdata[i]) < bin_edges[j + 1]:
                tmp_sum += ydata[i]
                found_bin_of_appartenence = True
            else:
                if found_bin_of_appartenence is True:
                    new_ydata.append(tmp_sum)
                    break
        if found_bin_of_appartenence is False:
            new_ydata.append(0)

    if len(global_bin_edges) == 0:
        # I have to add last bin edge, to compute last bin center and to calcilate last bin's y
        last_bin_edge = float(bin_edges[len(bin_edges) - 1]) + float((bin_edges[1] - bin_edges[0]))
        last_bin_center = (bin_edges[len(bin_edges) - 1] + last_bin_edge) / 2
        bin_edges = np.append(bin_edges, last_bin_edge)
        bin_centres = np.append(bin_centres, last_bin_center)

    tmp_sum = 0
    for i in range(0, len(xdata)):
        if float(xdata[i]) >= bin_edges[len(bin_edges) - 2]:
            tmp_sum += ydata[i]
    new_ydata.append(tmp_sum)

    print("bins_edges = " + str(bin_edges))
    print("new_ydata = " + str(new_ydata))

    # I convert my y values array into a np.array (N-multidimensional array)
    np_y = np.array(new_ydata)

    # I prune the non strictly positive y values
    y_pruned = np.where(np_y < 1, 1, np_y)

    print("bin_centres = " + str(bin_centres))
    print("np_y = " + str(np_y))
    print("y_pruned = " + str(y_pruned))

    # p0 = (200000, 0.001, 0)
    p0 = (200000, 0.003, 1)  # a=max(y) b=steep dacay between first and second data point  c=min(y)

    # I try to fit my scattered data with a third_polinomial function, with logarithmic y values
    popt, pcov = curve_fit(powlaw, bin_centres, np.log(y_pruned), p0=p0)
    print("popt = " + str(popt))
    print("pcov = " + str(pcov))

    fig, ax3 = plt.subplots()

    p = '%5.2f^{%5.3fx} + %5.2f' % tuple(popt)
    ax3.plot(bin_centres, powlaw(bin_centres, *popt), 'g--', label="$" + "y = " + str(p) + "$")
    ax3.plot(bin_centres, np.log(y_pruned), 'ro', label='data')

    ax3.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label + " (In(log(y)))")
    plt.show()

    return bin_edges, new_ydata


def draw_out_degree(x, x_label, y_label, global_bin_edges):

    # draw out-degree
    fig, ax1 = plt.subplots()

    # print bin's numbers and bins's size
    num_bins, binwidth = get_doane_num_bin_and_binwidth(x)
    print("num_bins = " + str(num_bins))
    print("binwidth = " + str(binwidth))

    if len(global_bin_edges) != 0:
        n, bin_edges, patches = ax1.hist(x, log=True, bins=global_bin_edges, color='red', width=binwidth / 2)
    else:
        n, bin_edges, patches = ax1.hist(x, log=True, bins='doane', color='red', width=binwidth / 2)

    fig, ax2 = plt.subplots(sharex=True, sharey=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    ax2.errorbar(bin_centres, n, fmt='ro')
    ax2.set_yscale('log')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    # I have to extract bin's y values in order to fit the scatter data
    c = Counter(x)
    c_sorted = sorted(c.items())
    print(c_sorted)

    xdata = []
    ydata = []
    for key, value in c_sorted:
        xdata.append(key)
        ydata.append(value)
    print(xdata)
    print(ydata)

    # I have to sum all y values falling in the same bin
    new_ydata = []
    for j in range(0, len(bin_edges) - 1):
        tmp_sum = 0
        found_bin_of_appartenence = False
        for i in range(0, len(xdata)):
            if float(xdata[i]) >= bin_edges[j] and float(xdata[i]) < bin_edges[j + 1]:
                tmp_sum += ydata[i]
                found_bin_of_appartenence = True
            else:
                if found_bin_of_appartenence is True:
                    new_ydata.append(tmp_sum)
                    break
        if found_bin_of_appartenence is False:
            new_ydata.append(0)

    if len(global_bin_edges) == 0:
        # I have to add last bin edge, to compute last bin center and to calcilate last bin's y
        last_bin_edge = float(bin_edges[len(bin_edges) - 1]) + float((bin_edges[1] - bin_edges[0]))
        last_bin_center = (bin_edges[len(bin_edges) - 1] + last_bin_edge) / 2
        bin_edges = np.append(bin_edges, last_bin_edge)
        bin_centres = np.append(bin_centres, last_bin_center)

    tmp_sum = 0
    for i in range(0, len(xdata)):
        if float(xdata[i]) >= bin_edges[len(bin_edges) - 2]:
            tmp_sum += ydata[i]
    new_ydata.append(tmp_sum)

    print("bins_edges = " + str(bin_edges))
    print("new_ydata = " + str(new_ydata))

    # I convert my y values array into a np.array (N-multidimensional array)
    np_y = np.array(new_ydata)

    # I prune the non strictly positive y values
    y_pruned = np.where(np_y < 1, 1, np_y)

    print("bin_centres = " + str(bin_centres))
    print("np_y = " + str(np_y))
    print("y_pruned = " + str(y_pruned))

    # I try to fit my scattered data with a third_polinomial function, with logarithmic y values
    popt, pcov = curve_fit(third_polinomial, bin_centres, np.log(y_pruned))
    print("popt = " + str(popt))
    print("pcov = " + str(pcov))

    fig, ax3 = plt.subplots()
    p = Polynomial(round(popt[0], 7), round(popt[1], 4), round(popt[2], 2), round(popt[3], 2))
    ax3.plot(bin_centres, third_polinomial(bin_centres, *popt), 'g--', label="$" + "y = " + str(p) + "$")
    ax3.plot(bin_centres, np.log(y_pruned), 'ro', label='data')

    ax3.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label + " (In(log(y)))")
    plt.show()

    return bin_edges, new_ydata


def degree_study_function_of_support(degree, final_leaders, outputfile, x_label, y_label):

    f = open(outputfile, "a")
    degree_all_nodes = []
    degree_leaders = []
    for node, deg in degree:
        str_to_write = str(node) + "::" + str(deg) + "\n"
        f.write(str_to_write)
        degree_all_nodes.append(deg)

        if int(node) in final_leaders:
             degree_leaders.append(deg)

    f.close()

    if x_label == "In-Degree":
        global_bin_edges, global_new_ydata = draw_in_degree(degree_all_nodes, x_label, y_label, [])
        bin_edges, new_ydata = draw_in_degree(degree_leaders, x_label, y_label, global_bin_edges)
    elif x_label == "Out-Degree":
        global_bin_edges, global_new_ydata = draw_out_degree(degree_all_nodes, x_label, y_label, [])
        bin_edges, new_ydata = draw_out_degree(degree_leaders, x_label, y_label, global_bin_edges)
    else:
        global_bin_edges, global_new_ydata = draw_out_degree(degree_all_nodes, x_label, y_label, [])

    if x_label == "In-Degree" or x_label == "Out-Degree":
        res = []
        for i in range(0, len(new_ydata)):
            diff = global_new_ydata[i] - new_ydata[i]
            res.append(diff)

        print()
        print("global_new_ydata = " + str(global_new_ydata))
        print("new_ydata = " + str(new_ydata))
        print("res = " + str(res))


def degree_study(G, start, final_leaders):

    in_degree = list(G.in_degree)
    degree_study_function_of_support(in_degree, final_leaders, "netstats_directed_old/directed_graph_in_degree", "In-Degree", "# Nodes")

    out_degree = list(G.out_degree)
    degree_study_function_of_support(in_degree, final_leaders, "netstats_directed_old/directed_graph_out_degree", "Out-Degree", "# Nodes")

    sys.stderr.write("In-Degree and Out-Degree calculated! Runtime: %s\n" % (time.time() - start))


def macroscopic_measurements(G, start, final_leaders):
    """
         Function which computes the macroscopic attributes of the newtork:
         1. total nodes
         2. total edges
         3. graph's density
         4. number of isolated nodes
         5. number of self loops
         6. average_shortest_path
     """

    number_of_total_nodes = len(G)
    number_of_total_edges = len(G.edges)
    sys.stderr.write("Nodes and edges' number calculated! Runtime: %s\n" % (time.time() - start))

    # fraction of existing edges out of all potentially possiblie edges => number between 0 e 1
    graph_density = nx.density(G)
    sys.stderr.write("Graph's density calculated! Runtime: %s\n" % (time.time() - start))

    # check if there are isolated nodes [a node with no neighbors (that is, with degree zero)]
    isolated_nodes = nx.number_of_isolates(G)
    sys.stderr.write("Isolates nodes calculated! Runtime: %s\n" % (time.time() - start))

    # check if there are self-loops (edge that has the same node at both ends)
    self_loops = G.number_of_selfloops()
    sys.stderr.write("Self loops calculated! Runtime: %s\n" % (time.time() - start))

    G.remove_edges_from(G.selfloop_edges())  # remove self-loops

    f = open("netstats_directed_old/directed_graph_macro_stats", 'a')
    f.write("Number nodes = %s \n" % number_of_total_nodes)
    f.write("\n")
    f.write("Number edges = %s \n" % number_of_total_edges)
    f.write("\n")
    f.write("Density = %s \n" % graph_density)
    f.write("\n")
    f.write("Isolated nodes = %s \n" % isolated_nodes)
    f.write("\n")
    f.write("Self-loops = %s \n" % self_loops)
    f.write("\n")
    f.close()


def microscopic_measurements(G, start, final_leaders):
    """
        Function which computes the microscopic attributes of the newtork:
        1. degree centrality
        2. average_neighbor_degree
        3. clustering coefficent
        4. betweeness centrality
        5. closeness centrality
        6. eigenvalue centrality
    """

    in_degree_centr = nx.in_degree_centrality(G)  # IS A DICT
    sys.stderr.write("In-Degree centrality calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_in_degree_centrality", 'a')
    for i in G:
        if int(i) in final_leaders:
            f.write("%d::%s\n" % (i, in_degree_centr[i]))
    f.close()

    in_degree_centr.clear()  # free the memory of the hashset used in the dict

    out_degree_centr = nx.out_degree_centrality(G)  # IS A DICT
    sys.stderr.write("Out-Degree centrality calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_out_degree_centrality", 'a')
    for i in G:
        if int(i) in final_leaders:
            f.write("%d::%s\n" % (i, out_degree_centr[i]))
    f.close()

    out_degree_centr.clear()  # free the memory of the hashset used in the dict

    neigh_degree = nx.average_neighbor_degree(G, source='in', target='in', nodes=final_leaders)  # IS A DICT
    sys.stderr.write("AVG Neighbor in-degree calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_avg_neighbor_in_degree_centrality", 'a')
    for key, value in neigh_degree.items():
        if int(key) in final_leaders:
            f.write("%s::%s\n" % (key, value))
    f.close()

    neigh_degree.clear()  # free the memory of the hashset used in the dict

    neigh_degree = nx.average_neighbor_degree(G, source='out', target='out', nodes=final_leaders)  # IS A DICT
    sys.stderr.write("AVG Neighbor out-degree calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_avg_neighbor_out_degree_centrality", 'a')
    for key, value in neigh_degree.items():
        if int(key) in final_leaders:
            f.write("%s::%s\n" % (key, value))
    f.close()

    neigh_degree.clear()  # free the memory of the hashset used in the dict

    avg_clusterings = nx.clustering(G, nodes=final_leaders)  # IS A DICT
    sys.stderr.write("Clusering calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_clustering", 'a')
    f.write("\n")
    f.write("Micro Stats:\n")
    for key, value in avg_clusterings.items():
        if int(key) in final_leaders:
            f.write("%s::%s\n" % (key, value))
    f.close()

    avg_clusterings.clear()  # free the memory of the hashset used in the dict

    clo_centr = {}
    for node in final_leaders:
        u_dict = (nx.closeness_centrality(G, u=node))  # IS A DICT
        for key, value in u_dict.items():
            clo_centr[str(key)] = value
    sys.stderr.write("Closeness centrality calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_closeness", 'a')
    for key, value in clo_centr.items():
        if int(key) in final_leaders:
            f.write("%s::%s\n" % (key, value))
    f.close()

    clo_centr.clear()  # free the memory of the hashset used in the dict

    eigenvector_centr = nx.eigenvector_centrality_numpy(G)
    sys.stderr.write("Eigenvector centrality calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_eigenvector", 'a')
    for i in G:
        if int(i) in final_leaders:
            f.write("%d::%s\n" % (i, eigenvector_centr[i]))
    f.close()

    eigenvector_centr.clear()  # free the memory of the hashset used in the dict

    bet_centr = compute_betweenness_centrality(G)
    sys.stderr.write("Betweenness centrality calculated! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_betweenness", 'a')
    for key, value in bet_centr.items():
        if int(key) in final_leaders:
            f.write("%s::%s\n" % (key, value))
    f.close()

    bet_centr.clear()  # free the memory of the hashset used in the dict

    # set end of the analysis
    analysis_time = time.time() - start

    sys.stderr.write("Done! Runtime: %s\n" % (time.time() - start))

    f = open("netstats_directed_old/directed_graph_macro_stats", 'a')
    f.write("Analysis time = %d \n" % analysis_time)
    f.close()


def compute_betweenness_centrality(G):
    """
        Function found on "https://medium.com/@pasdan/closeness-centrality-via-networkx-is-taking-too-long-1a58e648f5ce"
    """
    # Leverage the adjacency matrix and compute the shortest paths using the Floyd — Warshall Method.
    A = nx.adjacency_matrix(G).tolil()
    D = scipy.sparse.csgraph.floyd_warshall(A, directed=False, unweighted=False)

    # Use the shortest path Matrix to calculate the closeness metric for each node.
    n = D.shape[0]
    closeness_centrality = {}
    for r in range(0, n):

        cc = 0.0

        possible_paths = list(enumerate(D[r, :]))
        shortest_paths = dict(filter(lambda x: not x[1] == np.inf, possible_paths))

        total = sum(shortest_paths.values())
        n_shortest_paths = len(shortest_paths) - 1.0
        if total > 0.0 and n > 1:
            s = n_shortest_paths / (n - 1)
            cc = (n_shortest_paths / total) * s
            closeness_centrality[r] = cc

    return closeness_centrality


if __name__ == "__main__":

    # construct graph and get its macroscopic and microscopic measurements
    init()




