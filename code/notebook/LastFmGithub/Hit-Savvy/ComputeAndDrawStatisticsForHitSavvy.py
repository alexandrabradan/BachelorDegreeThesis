import math
import sys
import collections
import seaborn as sns
import json
import scipy
import csv
import networkx as nx
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
import ast
import tqdm
import random
from collections import Counter
import matplotlib.ticker
import math

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


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


def draw_histogram(x, x_label, y_label):
    fig, ax1 = plt.subplots()
    # hist, bins, _ = ax1.hist(np.log10(x), color="#red", ec="white", log=True)
    hist, bins, _ = ax1.hist(np.log10(x), color="#ff737d", ec="white", log=True)

    # SET CUSTORM TICK FORMATTING (in my case log10)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, y: '{}'.format(float(math.ceil(math.log(x, 10))))))

    # get mean
    np_x = np.array(x)
    plt.axvline(np.log10(np_x.mean()*1.1), color='black', linestyle='dashed', linewidth=3,
                label='Mean: {:.2f}'.format(np.log10(np_x.mean())))

    handles, labels = ax1.get_legend_handles_labels()
    binwidth = math.floor(bins[1] - bins[0])
    mylabel = "Binwidth: {}".format(binwidth) + ", Bins: {}".format(len(hist))
    red_patch = mpatches.Patch(color='red', label=mylabel)
    handles = [red_patch] + handles
    labels = [mylabel] + labels
    plt.legend(handles, labels)

    y_labels = list(plt.yticks()[0])
    print(y_labels)
    del y_labels[0]
    del y_labels[-1]
    plt.yticks(y_labels)

    plt.xlabel(x_label + " (log10)")
    plt.ylabel(y_label + " (log10)")
    plt.legend()
    plt.show()


def distribuition_adopters_per_same_artists_adopted():
    """
        Function which draws the distribuition of how many seed
        users adopted the same artists (and as a conseguence are his listeners)
    """
    x = []

    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct good from adoptions""")
    goods = cur.fetchall()

    for good in goods:
        good = good[0]

        cur = conn.cursor()
        cur.execute("""SELECT count(distinct adopter) from adoptions where good='%s'""" % good)
        res = cur.fetchall()  # [(total_good_unique_adopters, )]

        x.append(int(res[0][0]))
        print("good " + str(good) + ", seed_listeners = " + str(res[0][0]))

    # draw histogram
    draw_histogram(x, "# Unique adopters", "# Artists")


def distribuiotion_artists_per_total_playcounts():
    """
        Function which draws the distribuition of artists versus the total number of playcounts that the seed users
        have made
    """
    x = []

    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct good from adoptions""")
    goods = cur.fetchall()

    for good in goods:
        good = good[0]

        cur = conn.cursor()
        cur.execute("""SELECT sum(quantity) from adoptions where good='%s'""" % good)
        res = cur.fetchall()  # [(total_playcounts_seed_users )]

        x.append(int(res[0][0]))
        print("good " + str(good) + ", seed_playcounts = " + str(res[0][0]))

    # draw histogram
    draw_histogram(x, "# Playcounts", "# Artists")


def distribuition_adopters_per_artists_adopted():
    """
        Function which draws the distribuition of seed users versus how many artists they adopted
    """
    x = []

    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct adopter from adoptions""")
    adopters = cur.fetchall()

    for adopter in adopters:
        adopter = adopter[0]
        print("adopter = " + str(adopter))

        cur = conn.cursor()
        cur.execute("""SELECT count(distinct good) from adoptions where adopter='%s'""" % adopter)
        res = cur.fetchall()  # [(total_good_adopted,  )]
        print(res[0][0])

        x.append(res[0][0])

        # draw histogram
    draw_histogram(x, "# Items adopted", "# Adopters")


def distribuition_adopters_per_playcounts():
    """
        Function which draws the distribuition of seed users versus how many playcounts they made
    """
    x = []

    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct adopter from adoptions""")
    adopters = cur.fetchall()

    for adopter in adopters:
        adopter = adopter[0]
        print("adopter = " + str(adopter))

        cur = conn.cursor()
        cur.execute("""SELECT sum(quantity) from adoptions where adopter='%s'""" % adopter)
        res = cur.fetchall()  # [(total_plc,  )]
        print(res)

        x.append(res[0][0])

        # draw histogram
    draw_histogram(x, "# Playcounts", "# Adopters")


def distribuition_leaders_per_artists_adopted():
    """
        Function which draws the distribuition of leaders versus how many artists they adopted
    """
    x = []

    leaders = []
    data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["leader"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.leader)
        if u not in leaders:
            leaders.append(u)

    for adopter in leaders:
        print("adopter = " + str(adopter))

        conn = sqlite3.connect("lastfm.db")
        cur = conn.cursor()
        cur.execute("""SELECT count(distinct good) from adoptions where adopter='%s'""" % str(adopter))
        res = cur.fetchall()  # [(total_good_adopted,  )]
        print(res[0][0])

        x.append(res[0][0])

        # draw histogram
    draw_histogram(x, "# Items adopted", "# Leaders")


def distribuition_leaders_per_playcounts():
    """
        Function which draws the distribuition of leaders versus how many playcounts they made
    """
    x = []

    leaders = []
    data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["leader"])
    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.leader)
        if u not in leaders:
            leaders.append(u)

    for adopter in leaders:
        print("adopter = " + str(adopter))

        conn = sqlite3.connect("lastfm.db")
        cur = conn.cursor()
        cur.execute("""SELECT sum(quantity) from adoptions where adopter='%s'""" % str(adopter))
        res = cur.fetchall()  # [(total_plc,  )]
        print(res)

        x.append(res[0][0])

        # draw histogram
    draw_histogram(x, "# Playcounts", "# Leaders")


def draw_boxplot():
    collection_0 = []  # increasing trends
    collection_1 = []  # decreasing treands

    goods_in_collection_0 = []
    goods_in_collection_1 = []

    continuos_AT_file = "data/global_cotinous_at.json"
    with open(continuos_AT_file, 'r', encoding='utf-8') as infile:
        at_dict = json.load(infile)
        infile.close()

    goods = list(at_dict.keys())
    for i in range(0, len(goods)):
        goods[i] = int(goods[i])
    goods_at = list(at_dict.values())

    ground_truth_file = "data/global_AT_ground_truth_file5"
    g = open(ground_truth_file, "r", encoding="utf-8")
    for line in g:
        line_array = line.split(",")
        good = int(line_array[0])
        is_hit = int(line_array[1].replace("\n", ""))

        if good in goods:
            if is_hit == 1:
                goods_in_collection_0.append(good)  # increasing AT
            else:
                goods_in_collection_1.append(good)  # decreasing AT
    g.close()

    for i in range(0, len(goods_at)):
        tot_good_unique_adopters = 0
        for elem in goods_at[i]:
            elem = int(elem)
            tot_good_unique_adopters += elem
        if goods[i] in goods_in_collection_0:
            collection_0.append(tot_good_unique_adopters)
        elif goods[i] in goods_in_collection_1:
            collection_1.append(tot_good_unique_adopters)
        else:
            sys.exit(-1)

    data_to_plot = [collection_0, collection_1]
    # vert=0
    collection_0_label = "Increasing AT"
    collection_1_label = "Decreasing AT"
    box = plt.boxplot(data_to_plot, patch_artist=True, labels=[collection_0_label, collection_1_label],
                      showfliers=False)
    colors = ['green', 'red']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xscale('log')
    plt.yscale('log')
    y_axis_values = list(plt.yticks()[0])
    del y_axis_values[0]
    plt.yticks(y_axis_values)

    plt.ylabel("Unique adopters")
    plt.show()


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

distribuition_adopters_per_same_artists_adopted()
distribuiotion_artists_per_total_playcounts()
distribuition_adopters_per_artists_adopted()
distribuition_adopters_per_playcounts()
distribuition_leaders_per_artists_adopted()
distribuition_leaders_per_playcounts()

draw_boxplot()
