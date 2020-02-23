import networkx as nx
from igraph import *
import pandas as pd
import json
import sys
from matplotlib import pyplot as plt
import seaborn
import pandas as pd

from cdlib import algorithms
from cdlib import evaluation
from cdlib import ensemble
from cdlib import viz
from cdlib import readwrite
import infomap as imp

import warnings
warnings.filterwarnings('ignore')


def load_the_social_graph():

    g = nx.DiGraph()
    node_added = []

    print("Loading the social graphload")
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # print(data)

    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)
        if u not in node_added:
            node_added.append(u)

        if v not in node_added:
            node_added.append(v)

        g.add_edge(str(u), str(v))

    polling_and_optimisation()
    community_discoverying_algorithms(g)


def write_on_file(communities, filename):

    dict = json.loads(communities.to_json())

    dict["node_clustering_obj"] = str(communities)
    dict["map"] = str(communities.to_node_community_map())
    dict["min_max_mean_std"] = str(communities.average_internal_degree())  # Fitness obj
    dict["fraction_over_median_degree"] = str(communities.fraction_over_median_degree())
    dict["internal_edge_density"] = str(communities.internal_edge_density())
    dict["conductance"] = str(communities.conductance())
    dict["cut_ratio"] = str(communities.cut_ratio())
    dict["edges_inside"] = str(communities.edges_inside())
    dict["expansion"] = str(communities.expansion())
    dict["size"] = str(communities.size())
    # dict["triangle_participation_ratio"] = str(communities.triangle_participation_ratio())
    # dict["modularity density"] = str(communities.modularity_density())
    # dict["purity"] = str(communities.purity())

    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(dict, outfile, indent=4)


def draw_community_graph(g, coms, filename):
    pos = nx.spring_layout(g)
    viz.plot_network_clusters(g, coms, pos)
    plt.savefig(filename)


def polling_and_optimisation():
    threshold = ensemble.Parameter(name="threshold", start=0.25, end=1.0, step=0.25) # numeric range
    angel_conf = [threshold]

    resolution = ensemble.Parameter(name="resolution", start=0.1, end=1, step=0.1)  # numeric range
    louvain_conf = [resolution]

    methods = [algorithms.angel, algorithms.louvain]

    for coms, scoring in ensemble.pool_grid_filter(g, methods, [louvain_conf, angel_conf],
                                                   quality_score=evaluation.erdos_renyi_modularity, aggregate=max):
        print("%s\nCommunities:\n %s \nConfiguration: %s \nScoring: %s\n" % (
        coms.method_name, coms.communities, coms.method_parameters, scoring))


def draw_cluster_violin_map(list_of_communities):
    viz.plot_com_stat(list_of_communities, evaluation.size)
    plt.savefig("communities/size.png")

    viz.plot_com_stat(list_of_communities, evaluation.average_internal_degree)
    plt.savefig("communities/average_internal_degree.png")

    viz.plot_com_stat(list_of_communities, evaluation.edges_inside)
    plt.savefig("communities/edges_inside.png")

    viz.plot_com_stat(list_of_communities, evaluation.cut_ratio)
    plt.savefig("communities/cut_ratio.png")


def draw_plot_map(list_of_communities, number):
    lmplot = viz.plot_com_properties_relation(list_of_communities, evaluation.size, evaluation.internal_edge_density)
    filename = "communities/size_vs_internal_density" + str(number) + ".png"

    # lmplot = viz.plot_com_properties_relation(list_of_communities, evaluation.size, evaluation.triangle_participation_ratio)
    # filename = "communities/size_vs_triangle_participation_ratio" + str(number) + ".png"
    # plt.savefig(filename)

    lmplot = viz.plot_com_properties_relation(list_of_communities, evaluation.average_internal_degree,
                                              evaluation.fraction_over_median_degree)
    filename = "communities/internal_degree_vs_fraction_over_median_degree" + str(number) + ".png"
    plt.savefig(filename)

    lmplot = viz.plot_com_properties_relation(list_of_communities, evaluation.conductance, evaluation.expansion)
    filename = "communities/outside_vs_inside_edges" + str(number) + ".png"
    plt.savefig(filename)

    lmplot = viz.plot_com_properties_relation(list_of_communities, evaluation.edges_inside, evaluation.cut_ratio)
    filename = "communities/edges_inside_vs_cut_ratio" + str(number) + ".png"
    plt.savefig(filename)

    # lmplot = viz.plot_com_properties_relation(list_of_communities, evaluation.internal_edge_density, evaluation.modularity_density)
    # filename = "communities/internal_edge_density_vs_modularity_density" + str(number) + ".png"
    # plt.savefig(filename)


def draw_cluster_heatmap(list_of_communities):
    clustermap = viz.plot_sim_matrix(list_of_communities, evaluation.adjusted_mutual_information)
    plt.savefig("communities/clustermap.png")


def community_discoverying_algorithms(g):
    """
    All Community Discovery algorithms generate as result a NodeClustering object, allowing
    also for the generation of a JSON representation of the results. Then evaluate the clusters with fitness
    functions (ex. synthetic representation of its min/max/mean/std values ORD communitiy-wise value)
    """

    print("Starting computing angel_coms")
    angel_coms = algorithms.angel(g.to_undirected(), threshold=0.25)
    write_on_file(angel_coms, "communities/angel.json")
    draw_community_graph(g, angel_coms, "communities/angel.png")
    print("END")

    print("Starting computing infomap_coms")
    infomap_coms = algorithms.infomap(g.to_undirected())
    write_on_file(infomap_coms, "communities/infomap.json")
    draw_community_graph(g, infomap_coms, "communities/infomap.png")
    print("END")

    print("Starting computing louvain_coms")
    louvain_coms = algorithms.louvain(g.to_undirected())
    write_on_file(louvain_coms, "communities/louvain.json")
    draw_community_graph(g, louvain_coms, "communities/louvain.png")
    print("END")

    print("Starting computing labelpropagation_coms")
    labelpropagation_coms = algorithms.label_propagation(g.to_undirected())
    write_on_file(labelpropagation_coms, "communities/labelpropagation.json")
    draw_community_graph(g, labelpropagation_coms, "communities/labelpropagation.png")
    print("END")

    draw_cluster_violin_map([angel_coms, infomap_coms, louvain_coms, labelpropagation_coms])
    draw_cluster_heatmap([angel_coms, infomap_coms, louvain_coms, labelpropagation_coms])

    draw_plot_map([angel_coms, infomap_coms], 1)
    draw_plot_map([angel_coms, louvain_coms], 2)
    draw_plot_map([angel_coms, labelpropagation_coms], 3)
    draw_plot_map([infomap_coms, louvain_coms], 4)
    draw_plot_map([infomap_coms, labelpropagation_coms], 5)
    draw_plot_map([louvain_coms, labelpropagation_coms], 6)

    # evaluate the clusterings
    # print(evaluation.normalized_mutual_information(louvain_coms, infomap_coms))
    # print(evaluation.normalized_mutual_information(louvain_coms, labelpropagation_coms))
    # print(evaluation.normalized_mutual_information(louvain_coms, angel_coms))
    # print(evaluation.normalized_mutual_information(infomap_coms, labelpropagation_coms))
    # print(evaluation.normalized_mutual_information(infomap_coms, angel_coms))
    # print(evaluation.normalized_mutual_information(labelpropagation_coms, angel_coms))


load_the_social_graph()