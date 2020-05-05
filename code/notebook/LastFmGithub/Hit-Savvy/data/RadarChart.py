import math
import sys
import collections
import seaborn as sns
import json
import scipy
import tqdm
import csv
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
from pylab import *
import collections

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


def collect_profile_data():
    """
        Function which collect the data present in the "profle_mu_and_sigma_global.json" file
        and return a list of lists, where each list present in the main list corresponds
        to the PLDM of each user
    """
    with open("profile_mu_and_sigma.json", 'r', encoding='utf-8') as infile:
        profile_dict = json.load(infile)

    data = {}
    for ad in list(profile_dict.keys()):
        ad_x = []

        L_u = int(profile_dict[str(ad)]["nlistening"])
        ad_x.append(L_u)
        A_u = int(profile_dict[str(ad)]["nartists"])
        ad_x.append(A_u)
        G_u = int(profile_dict[str(ad)]["ngenres"])
        ad_x.append(G_u)
        S_u = int(profile_dict[str(ad)]["nslots"])
        ad_x.append(S_u)
        Q_u = int(profile_dict[str(ad)]["nquantities"])
        ad_x.append(Q_u)

        e_au = float(profile_dict[str(ad)]["e_au"])
        ad_x.append(e_au)
        e_gu = float(profile_dict[str(ad)]["e_gu"])
        ad_x.append(e_gu)
        e_su = float(profile_dict[str(ad)]["e_su"])
        ad_x.append(e_su)
        e_qu = float(profile_dict[str(ad)]["e_qu"])
        ad_x.append(e_qu)

        """hat_au = int(profile_dict[str(ad)]["hat_au"])
        ad_x.append(hat_au)
        hat_gu = int(profile_dict[str(ad)]["hat_gu"])
        ad_x.append(hat_gu)
        hat_su = int(profile_dict[str(ad)]["hat_su"])
        ad_x.append(hat_su)
        hat_qu = int(profile_dict[str(ad)]["hat_qu"])
        ad_x.append(hat_qu)

        tilde_au = [int(x) for x in list(profile_dict[str(ad)]["tilde_au"].keys())]
        ad_x.append(tilde_au)
        tilde_gu = [int(x) for x in list(profile_dict[str(ad)]["tilde_gu"].keys())]
        ad_x.append(tilde_gu)
        tilde_su = [int(x) for x in list(profile_dict[str(ad)]["tilde_su"].keys())]
        ad_x.append(tilde_su)
        tilde_qu = [int(x) for x in list(profile_dict[str(ad)]["tilde_qu"].keys())]
        ad_x.append(tilde_qu)"""

        followings = [int(x) for x in profile_dict[str(ad)]["followings"]]
        followers = [int(x) for x in profile_dict[str(ad)]["followers"]]
        if len(followings) == 0 or len(followers) == 0:
            continue

        mu_au = float(profile_dict[str(ad)]["mu_au_followings"])
        ad_x.append(mu_au)
        mu_gu = float(profile_dict[str(ad)]["mu_gu_followings"])
        ad_x.append(mu_gu)
        mu_su = float(profile_dict[str(ad)]["mu_su_followings"])
        ad_x.append(mu_su)
        mu_qu = float(profile_dict[str(ad)]["mu_qu_followings"])
        ad_x.append(mu_qu)
        sigma_au = float(profile_dict[str(ad)]["sigma_au_followings"])
        ad_x.append(sigma_au)
        sigma_gu = float(profile_dict[str(ad)]["sigma_gu_followings"])
        ad_x.append(sigma_gu)
        sigma_su = float(profile_dict[str(ad)]["sigma_su_followings"])
        ad_x.append(sigma_su)
        sigma_qu = float(profile_dict[str(ad)]["sigma_qu_followings"])
        ad_x.append(sigma_qu)

        mu_au = float(profile_dict[str(ad)]["mu_au_followers"])
        ad_x.append(mu_au)
        mu_gu = float(profile_dict[str(ad)]["mu_gu_followers"])
        ad_x.append(mu_gu)
        mu_su = float(profile_dict[str(ad)]["mu_su_followers"])
        ad_x.append(mu_su)
        mu_qu = float(profile_dict[str(ad)]["mu_qu_followers"])
        ad_x.append(mu_qu)
        sigma_au = float(profile_dict[str(ad)]["sigma_au_followers"])
        ad_x.append(sigma_au)
        sigma_gu = float(profile_dict[str(ad)]["sigma_gu_followers"])
        ad_x.append(sigma_gu)
        sigma_su = float(profile_dict[str(ad)]["sigma_su_followers"])
        ad_x.append(sigma_su)
        sigma_qu = float(profile_dict[str(ad)]["sigma_qu_followers"])
        ad_x.append(sigma_qu)

        """global_mu_au = float(profile_dict[str(ad)]["global_mu_au"])
        ad_x.append(global_mu_au)
        global_mu_gu = float(profile_dict[str(ad)]["global_mu_gu"])
        ad_x.append(global_mu_gu)
        global_mu_su = float(profile_dict[str(ad)]["global_mu_su"])
        ad_x.append(global_mu_su)
        global_mu_qu = float(profile_dict[str(ad)]["global_mu_qu"])
        ad_x.append(global_mu_qu)
        global_sigma_au = float(profile_dict[str(ad)]["global_sigma_au"])
        ad_x.append(global_sigma_au)
        global_sigma_gu = float(profile_dict[str(ad)]["global_sigma_gu"])
        ad_x.append(global_sigma_gu)
        global_sigma_su = float(profile_dict[str(ad)]["global_sigma_su"])
        ad_x.append(global_sigma_su)
        global_sigma_qu = float(profile_dict[str(ad)]["global_sigma_qu"])
        ad_x.append(global_sigma_qu)"""

        data[str(ad)] = ad_x
    return data

def collect_leaders_mean_data():
    """
        Function which collect the data present in the "final_leaders_DIRECTED_mean_values.json" file
        and return a list of lists, where each list present in the mean values associated with each
        leader
    """
    with open("final_leaders_DIRECTED_mean_values.json", 'r', encoding='utf-8') as infile:
        mean_values_dict = json.load(infile)

    data = {}
    for l in list(mean_values_dict.keys()):
        ad_x = []
        for key, value in mean_values_dict[str(l)].items():
            ad_x.append(value)
        data[str(l)] = ad_x
    return data


def elbow_criterion_for_optimal_k_for_kmeans(clustering_range, data):
    data = list(data.values())

    sse = {}
    for k in range(1, clustering_range):
        print("sse for k = " + str(k))
        kmeans = KMeans(n_clusters=k, max_iter=100000000).fit(data)
        # data["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    # plt.xticks([x for x in range(1, clustering_range + 1, 10)])
    plt.xticks([x for x in range(1, clustering_range + 1, 1)])
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()


def silhoutte_coefficent_for__optimal_k_for_kmeans(clustering_range, data):
    data = list(data.values())

    sc = {}
    for n_cluster in range(2, clustering_range):
        print("silhoutte coefficent for n_cluster = " + str(n_cluster))
        kmeans = KMeans(n_clusters=n_cluster).fit(data)
        label = kmeans.labels_
        sil_coeff = silhouette_score(data, label, metric='euclidean')
        sc[n_cluster] = sil_coeff
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

    plt.figure()
    # plt.xticks([x for x in range(1, clustering_range + 1, 10)])
    plt.xticks([x for x in range(1, clustering_range + 1, 1)])
    plt.plot(list(sc.keys()), list(sc.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhouette Coefficient")
    plt.show()


def cluster_with_KMeans(num_clusters, data, res_file):
    adopters = list(data.keys())
    data = list(data.values())
    clustering_res = {}
    clustering_res["cluster_members"] = {}
    clustering_res["new_labels"] = {}
    clustering_res["centers"] = {}

    km = KMeans(n_clusters=num_clusters, n_init=42, max_iter=100000000)
    km.fit(data)

    new_labels = km.labels_
    # print("new_labels = " + str(new_labels))
    # print("new_labels = " + str(len(new_labels)))

    for i in range(0, len(adopters)):
        clustering_res["new_labels"][str(adopters[i])] = int(new_labels[i])

    centers = km.cluster_centers_
    # print(centers)
    for i in range(0, len(centers)):
        key = "cluster " + str(i)
        print(centers[i])
        clustering_res["centers"][key] = [float(x) for x in centers[i]]

    counter_dict = collections.Counter(list(new_labels))
    # sort dict by key
    sorted_counter_dict = {k: v for k, v in sorted(counter_dict.items(), key=lambda x: x[0])}
    for key, value in sorted_counter_dict.items():
        print("Cluster " + str(key) + "'s members = " + str(value))
        clustering_res["cluster_members"][str(key)] = int(value)

    # write results on file
    with open(res_file, "w", encoding="utf-8") as outfile:
        json.dump(clustering_res, outfile, indent=4)


def plot_profile_statistics():
    """
        Function which makes some plots of the ProfileBuilder result in order to
        make some insight for the informations retrieved
    """


def get_cluster_data(clust_res_file, num_cluster, index_list, tot_adopters):
    with open(clust_res_file, 'r', encoding='utf-8') as infile:
        clust_res_dict = json.load(infile)

    stats = []
    key = "cluster " + str(num_cluster)
    c_list_values = clust_res_dict["centers"][key]
    perc_adopters = '%.2f' % (clust_res_dict["cluster_members"][str(num_cluster)] / tot_adopters)
    # access with the index list passed by argument the values to display
    for i in index_list:
        stats.append(c_list_values[i])
    return stats, perc_adopters


def get_name(num_clust, perc_adopters):
    name_array = ["|A|", "|B|", "|C|", "|D|", "|E|", "|F|","|G|","|H|", "|I|", "|L|", "|M|", "|N|", "|O|",
                  "|P|" "|Q|", "|R|", "|S|", "|T|", "|U|", "|V|", "|Z|"]
    name = name_array[num_clust] + " = " + str(perc_adopters)
    return name


def make_radar_chart(data, clust_res_file, dir_to_save, file_to_save, tot_clust, attribute_labels, index_list, color_list):

    plot_markers = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # markers
    plot_str_markers = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    labels = np.array(attribute_labels)

    tot_adopters = len(data)
    for i in range(0, tot_clust):
        stats, perc_adopters = get_cluster_data(clust_res_file, i, index_list, tot_adopters)
        color = color_list[i]
        name = get_name(i, perc_adopters)

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        stats = np.concatenate((stats,[stats[0]]))
        angles = np.concatenate((angles,[angles[0]]))

        fig= plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, stats, 'o-', linewidth=2, alpha=0.25, color=color)
        ax.fill(angles, stats, alpha=0.25, color=color)
        # ax.set_thetagrids(angles * 180/np.pi, labels)
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        plt.yticks(plot_markers)
        ax.set_title(name, fontweight="bold", loc='right')
        ax.grid(True)

        fig.savefig(dir_to_save + file_to_save +  "%s.png" % name)
        # plt.show()


def get_most_frequent_element_in_list(test_list):
    max = 0
    res = test_list[0]
    for i in test_list:
        freq = test_list.count(i)
        if freq > max:
            max = freq
            res = i
    return res


def mean_leaders_attributes():
    """
        Function which performs for each leader a mean among its Width, Depth, Streng and music tag
        (retriving the most representative among the items he spreads) and saves the results on file
    """
    data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t",
                    usecols=["action", "leader", "tribe", "depth", "mean_depth", "width" , "strength", "music_tag"])

    leaders = {}
    # iterate over dataframe's rows
    for row in tqdm.tqdm(data.itertuples()):
       l = int(row.leader)
       g = int(row.action)
       g_genre = int(row.music_tag)
       tribe = int(row.tribe)
       depth = float(row.depth)
       mean_depth = float(row.mean_depth)
       width = float(row.width)
       strength = float(row.strength)

       try:
           l_info = leaders[str(l)]
           l_info["tot_num_trees"] += 1
           l_info["genre"].append(g_genre)
           l_info["tribe"] += tribe
           l_info["depth"] += depth
           l_info["mean_depth"] += mean_depth
           l_info["width"] += width
           l_info["strength"] += strength
           leaders[str(l)] = l_info
       except KeyError:
           leaders[str(l)] = {}
           leaders[str(l)]["tot_num_trees"] = 1
           leaders[str(l)]["genre"] = [g_genre]
           leaders[str(l)]["tribe"] = tribe
           leaders[str(l)]["depth"] = depth
           leaders[str(l)]["mean_depth"] = mean_depth
           leaders[str(l)]["width"] = width
           leaders[str(l)]["strength"] = strength

    # computer for every leader its mean values
    leaders_mean_values = {}
    for l in list(leaders.keys()):
        leaders_mean_values[str(l)] = {}
        leaders_mean_values[str(l)]["actions"] = leaders[str(l)]["tot_num_trees"]
        leaders_mean_values[str(l)]["genre"] = get_most_frequent_element_in_list(leaders[str(l)]["genre"])
        leaders_mean_values[str(l)]["tribe"] =  leaders[str(l)]["tribe"] / leaders[str(l)]["tot_num_trees"]
        leaders_mean_values[str(l)]["depth"] = leaders[str(l)]["depth"] / leaders[str(l)]["tot_num_trees"]
        leaders_mean_values[str(l)]["mean_depth"] =  leaders[str(l)]["mean_depth"] / leaders[str(l)]["tot_num_trees"]
        leaders_mean_values[str(l)]["width"] = leaders[str(l)]["width"] / leaders[str(l)]["tot_num_trees"]
        leaders_mean_values[str(l)]["strength"] = leaders[str(l)]["strength"] / leaders[str(l)]["tot_num_trees"]

    # write results on file
    with open("final_leaders_DIRECTED_mean_values.json", "w", encoding="utf-8") as outfile:
        json.dump(leaders_mean_values, outfile, indent=4)


plot_profile_statistics()
data = collect_profile_data()
elbow_criterion_for_optimal_k_for_kmeans(31, data)
silhoutte_coefficent_for__optimal_k_for_kmeans(31, data)
num_clust = 3
cluster_with_KMeans(num_clust, data, "LPDM_clust_res.json")
read clusters' centroids and plot them on a radar/spider plot

attribute_labels = ["sigma_au_followers", "sigma_gu_followers", "sigma_su_followers", "sigma_qu_followers"]
index_list = [21, 22, 23, 24]
color_list = ["red", "blue", "green", "purple", "orange", "black", "brown", "cyan"]
make_radar_chart(data, "LPDM_clust_res.json", "Plotting/PLDM/", "sigma_followers", num_clust,  attribute_labels, index_list, color_list)

attribute_labels = ["mu_au_followers", "mu_gu_followers", "mu_su_followers", "mu_qu_followers"]
index_list = [17, 18, 19, 20]
color_list = ["red", "blue", "green", "purple", "orange", "black", "brown", "cyan"]
make_radar_chart(data, "LPDM_clust_res.json", "Plotting/PLDM/", "mu_followers", num_clust,  attribute_labels, index_list, color_list)

attribute_labels = ["sigma_au_followings", "sigma_gu_followings", "sigma_su_followings", "sigma_qu_followings"]
index_list = [13, 14, 15, 16]
color_list = ["red", "blue", "green", "purple", "orange", "black", "brown", "cyan"]
make_radar_chart(data, "LPDM_clust_res.json", "Plotting/PLDM/", "sigma_followings", num_clust,  attribute_labels, index_list, color_list)

attribute_labels = ["mu_au_followings", "mu_gu_followings", "mu_su_followings", "mu_qu_followings"]
index_list = [9, 10, 11, 12]
color_list = ["red", "blue", "green", "purple", "orange", "black", "brown", "cyan"]
make_radar_chart(data, "LPDM_clust_res.json", "Plotting/PLDM/", "mu_followings", num_clust,  attribute_labels, index_list, color_list)

attribute_labels = ["e_au", "e_gu", "e_su", "e_qu"]
index_list = [5, 6, 7, 8]
color_list = ["red", "blue", "green", "purple", "orange", "black", "brown", "cyan"]
make_radar_chart(data, "LPDM_clust_res.json", "Plotting/PLDM/", "e", num_clust,  attribute_labels, index_list, color_list)


mean_leaders_attributes()
data = collect_leaders_mean_data()
elbow_criterion_for_optimal_k_for_kmeans(31, data)
silhoutte_coefficent_for__optimal_k_for_kmeans(10, data)
cluster_with_KMeans(5, data, "leaders_mean_values_clust_res.json")
