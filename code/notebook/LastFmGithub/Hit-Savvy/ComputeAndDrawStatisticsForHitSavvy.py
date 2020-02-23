import math
import sys
import collections
import seaborn as sns
import json
import scipy
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
import matplotlib.cm as cm
import matplotlib.colors as colors
from pylab import *
import collections
import sqlite3
import ast

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

    # print bin's numbers and bins's size
    num_bins, binwidth = get_doane_num_bin_and_binwidth(x)
    print("num_bins = " + str(num_bins))
    print("binwidth = " + str(binwidth))

    # binning done with Doane formula
    # y axis is logarithmic
    n, bins_edges, patches = ax1.hist(x, log=True, bins='doane', color='red', width=binwidth / 1.2)
    print("number of elements in each bin = " + str(n))
    print("number of bins = " + str(len(n)))
    print("bin's edges = " + str(bins_edges))

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    np_x = np.array(x)
    print("np_x.mean() = " + str(np_x.mean()))
    plt.axvline(np_x.mean() * 1.1, color='lime', linestyle='dashed', linewidth=3,
                label='Mean: {:.2f}'.format(np_x.mean()))

    handles, labels = ax1.get_legend_handles_labels()
    binwidth = math.floor(bins_edges[1] - bins_edges[0])
    mylabel = "Binwidth: {}".format(binwidth) + ", Bins: {}".format(len(n))
    red_patch = mpatches.Patch(color='red', label=mylabel)
    handles = [red_patch] + handles
    labels = [mylabel] + labels
    plt.legend(handles, labels)

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
    draw_histogram(x, "# Items adopted", "# Unique adopters")


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

    plt.yscale('log')
    y_axis_values = list(plt.yticks()[0])
    del y_axis_values[0]
    plt.yticks(y_axis_values)

    plt.ylabel("Unique adopters")
    plt.show()


def draw_predictive_values():
    PPV = [0]
    TPR = [0]
    NPV = [0]
    TNR = [0]
    for i in range(6, 14):
        predictive_file = "prova" + str(i) + "_db"
        f = open(predictive_file, "r", encoding="utf-8")
        dictionary = ast.literal_eval(f.read())
        f.close()
        ppv = dictionary["precision"]
        tpr = dictionary["recall (without unclassified)"]
        npv = dictionary["specificity (with unclassified)"]
        print(npv)
        tnr = dictionary["NPV"]
        print(tnr)
        PPV.append(ppv)
        TPR.append(tpr)
        NPV.append(npv)
        TNR.append(tnr)

    fig, ax = plt.subplots()
    fig.canvas.draw()

    x = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ax.plot(x, PPV, label="PPV", color="red")
    ax.plot(x, TPR, 'r--', label="TPR")
    plt.legend()
    plt.xlabel("Months observed")
    plt.ylabel("Score")
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, NPV, label="NPV", color="blue")
    ax.plot(x, TNR, 'b--', label="TNR")
    plt.legend()
    plt.xlabel("Months observed")
    plt.ylabel("Score")
    plt.show()

def vertical_mean_line_survived(x, **kwargs):
    ls = {"0":"-","1":"--"}
    plt.axvline(x.mean(), linestyle =ls[kwargs.get("label","0")],
                color = kwargs.get("color", "g"))
    txkw = dict(size=12, color = kwargs.get("color", "g"), rotation=90)
    tx = "mean: {:.2f}, std: {:.2f}".format(x.mean(),x.std())
    plt.text(x.mean()+1, 0.052, tx, **txkw)


def draw_m_threshold():
    conn = sqlite3.connect("prova_GLOBAL.db")
    curr = conn.cursor()
    curr.execute("""SELECT * from model""")
    model = curr.fetchall()
    print(model)

    curr.execute("""SELECT * from '%s' """ % (model[0][1]))
    res = curr.fetchall()
    print(res)
    print(len(res))

    goods_table = model[0][1] + "_goods"
    curr.execute("""SELECT * from '%s' """ % (goods_table))
    res = curr.fetchall()
    print(res)
    print(len(res))

    x = []
    for elem in res:
        adoption = elem[1]
        x.append(adoption)

    fig, ax1 = plt.subplots()
    p = sns.kdeplot(x, shade=True, color="green")
    tx = "m: {:.2f}".format(model[0][2])
    plt.text(model[0][2] + 0.001, 80, tx, color="blue")

    plt.axvline(model[0][2], color="blue")

    plt.show()

def draw_n_threshold():
    conn = sqlite3.connect("prova_GLOBAL.db")
    curr = conn.cursor()
    curr.execute("""SELECT * from model""")
    model = curr.fetchall()
    print(model)

    curr.execute("""SELECT * from '%s' """ % (model[1][1]))
    res = curr.fetchall()
    print(res)
    print(len(res))

    goods_table = model[1][1] + "_goods"
    curr.execute("""SELECT * from '%s' """ % (goods_table))
    res = curr.fetchall()
    print(res)
    print(len(res))

    x = []
    for elem in res:
        adoption = elem[1]
        x.append(adoption)

    print(x)

    fig, ax1 = plt.subplots()
    sns.kdeplot(x, shade=True, color="red")
    """tx = "n: {:.2f}".format(model[1][2])
    plt.text(model[1][2] - 0.025, 200, tx, color="fuchsia")

    plt.axvline(model[1][2], color="fuchsia")"""
    plt.yscale('log')
    plt.show()

# distribuition_adopters_per_same_artists_adopted()
# distribuiotion_artists_per_total_playcounts()
# distribuition_adopters_per_artists_adopted()
# distribuition_adopters_per_playcounts()
# distribuition_leaders_per_artists_adopted()
# distribuition_leaders_per_playcounts()

# draw_boxplot()
# draw_predictive_values()
# draw_m_threshold()
# draw_n_threshold()