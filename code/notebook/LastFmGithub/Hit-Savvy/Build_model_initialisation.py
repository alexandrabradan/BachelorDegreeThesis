import networkx as nx
import math
import sqlite3
import numpy as np
import operator
import tqdm
import sys
import os
import json
from itertools import (takewhile, repeat)

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
from pylab import *
import collections
from sklearn.metrics import classification_report
from random import sample

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


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


def create_directory(path):
    """
   Create a new directory in the specified path
   :param path: path where to create the new directory
   """
    os.makedirs(path)


def __build_db(database_name):
    """
    Function which builds a database (the main database or the ones relative to the 4 training sets)
    and populates the database with the table "adoptions"
    :param database_name: the name of the database to create
    """
    conn = sqlite3.connect(database_name)
    conn.execute("""DROP TABLE IF EXISTS adoptions;""")
    conn.commit()
    conn.execute("""CREATE TABLE IF NOT EXISTS adoptions
                   (good TEXT  NOT NULL,
                   adopter TEXT NOT NULL,
                   slot      INTEGER NOT NULL,
                   quantity  INTEGER
                   );""")

    conn.close()


def __build_friendship_db(friendship_db):
    """
        Function which builds the friendship db
    :param friendship_db: friendship db name
    """
    conn = sqlite3.connect(friendship_db)
    conn.execute("""DROP TABLE IF EXISTS friendship;""")
    conn.commit()
    conn.execute("""CREATE TABLE IF NOT EXISTS friendship
                   (source TEXT  NOT NULL,
                   target TEXT NOT NULL);""")
    conn.close()


def __build_table_for_test_set(database_name):
    """
    Function which constructs in the database of one of the 4 training sets the table "adoptions_test"
    to use as training set
    :param database_name: the name of the database in which to create the table
    """
    conn = sqlite3.connect(database_name)
    conn.execute("""DROP TABLE IF EXISTS adoptions_test;""")
    conn.commit()
    conn.execute("""CREATE TABLE  adoptions_test
                   (good TEXT  NOT NULL,
                   adopter TEXT NOT NULL,
                   slot      INTEGER NOT NULL,
                   quantity  INTEGER
                   );""")

    conn.close()


def load_main_db_data(adoption_log_file, main_db_name):
    """
        Function which constructs the main adoption database, from which to split the various training sets
        :param adoption_log_file: file which contains all the adoptions made
        :param main_db_name: name of the main database
    """

    # build main database
    __build_db(main_db_name)

    conn = sqlite3.connect(main_db_name)
    f = open(adoption_log_file)
    # f.next()
    count = 0
    for row in tqdm.tqdm(f):
        row = row.rstrip().split(",")
        conn.execute("""INSERT into adoptions (good, adopter, slot, quantity) VALUES ('%s', '%s', %d, %d)""" %
                     (row[0], row[1], int(row[2]), int(row[3])))
        count += 1
        if count % 10000 == 0:
            conn.commit()
    conn.commit()
    conn.execute('CREATE INDEX good_idx on adoptions(good)')
    conn.execute('CREATE INDEX adopter_idx on adoptions(adopter)')
    conn.execute('CREATE INDEX slot_idx on adoptions(slot)')
    conn.close()


def load_friendship_db_data(friendship_file, friendship_db, main_db):
    """
        Function which constructs the main adoption database, from which to split the various training sets
        :param adoption_log_file: file which contains all the adoptions made
        :param friendship_db: name of the friendship database
    """
    # retrieve all adopters
    adopters = []
    conn = sqlite3.connect("%s" % main_db)
    curr = conn.cursor()
    curr.execute("""SELECT distinct adopter from adoptions""")
    res = curr.fetchall()
    for ad in res:
        ad = ad[0]
        if ad not in adopters:
            adopters.append(ad)
    curr.close()
    conn.close()

    # build friendship database
    __build_friendship_db(friendship_db)

    count = 0
    conn = sqlite3.connect(friendship_db)
    data = pd.read_csv(friendship_file, delimiter="\t", usecols=['Source', 'Target'], encoding="utf-8")
    for row in tqdm.tqdm(data.itertuples()):
        u = row.Source
        v = row.Target

        if str(u) not in adopters:
            continue
        if str(v) not in adopters:
            continue

        conn.execute("""INSERT into friendship (source, target) VALUES ('%s', '%s')""" % (u, v))
        count += 1
        if count % 10000 == 0:
            conn.commit()
    conn.commit()
    conn.execute('CREATE INDEX source_idx on friendship(source)')
    conn.execute('CREATE INDEX target_idx on friendship(target)')
    conn.close()


def __rawincount(filename):
    f = open(filename, 'rb')
    buf_gen = takewhile(lambda x: x, (f.read(1024 * 1024) for _ in repeat(None)))
    return sum(buf.count(b'\n') for buf in buf_gen)


def check_if_at_is_continous(at, period_of_continuity):
    continous_week = 0
    for i in range(0, len(at)):
        if int(at[i]) > 0:
            continous_week += 1
        else:
            continous_week = 0

        if continous_week == period_of_continuity:
            return True

    if continous_week >= period_of_continuity:
        return True
    else:
        return False


def compute_GLOBAL_adoption_trand():
    f1 = "global_continous_goods"
    f2 = "global_not_continous_goods"

    artists = []
    f = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        g = int(line_array[0])
        if g not in artists:
            artists.append(g)
    f.close()
    print("TOTAL ARTIST = " + str(len(artists)))

    conn = sqlite3.connect("lastfm.db")
    curr = conn.cursor()
    curr.execute("""SELECT distinct slot from adoptions order by slot asc""")
    weeks = curr.fetchall()

    tmp_dict = {}
    tmp_dict_not_cont = {}

    for g in artists:
        # initialize the dict
        tmp_AT = {}
        for w in weeks:
            w = w[0]
            tmp_AT[str(w)] = 0

        period_of_continuity = 38  # half of 76 weeks
        print("period_of_continuity TRS = " + str(period_of_continuity))

        curr = conn.cursor()
        curr.execute("""SELECT distinct adopter, slot from adoptions where good='%s' order by slot asc""" % str(g))
        adopters = curr.fetchall()

        for elem in adopters:
            a = elem[0]
            s = elem[1]
            counter = tmp_AT[str(s)]
            counter += 1
            tmp_AT[str(s)] = counter

        at = list(tmp_AT.values())
        for elem in at:
            if int(elem) < 0:
                print("good's at problem = " + str(g))
                sys.exit(-1)
        check = check_if_at_is_continous(at, period_of_continuity)

        if check is True:
            print("good = " + str(g) + " is CONTINOUS")
            tmp_dict[str(g)] = at
        else:
            print("good = " + str(g) + " not continous")
            tmp_dict_not_cont[str(g)] = at

    curr.close()
    conn.close()

    with open(f1, 'w', encoding='utf-8') as outfile:
        json.dump(tmp_dict, outfile, indent=4)
        outfile.close()

    with open(f2, 'w', encoding='utf-8') as outfile:
        json.dump(tmp_dict_not_cont, outfile, indent=4)
        outfile.close()

def compute_GLOBAL_adoption_trand():
    f1 = "global_continous_goods"
    f2 = "global_not_continous_goods"

    artists = []
    f = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        g = int(line_array[0])
        if g not in artists:
            artists.append(g)
    f.close()
    print("TOTAL ARTIST = " + str(len(artists)))

    conn = sqlite3.connect("lastfm.db")
    curr = conn.cursor()
    curr.execute("""SELECT distinct slot from adoptions order by slot asc""")
    weeks = curr.fetchall()

    tmp_dict = {}
    tmp_dict_not_cont = {}

    for g in artists:
        # initialize the dict
        tmp_AT = {}
        for w in weeks:
            w = w[0]
            tmp_AT[str(w)] = 0

        period_of_continuity = 38  # half of 76 weeks
        print("period_of_continuity TRS = " + str(period_of_continuity))

        curr = conn.cursor()
        curr.execute("""SELECT distinct adopter, slot from adoptions where good='%s' order by slot asc""" % str(g))
        adopters = curr.fetchall()

        for elem in adopters:
            a = elem[0]
            s = elem[1]
            counter = tmp_AT[str(s)]
            counter += 1
            tmp_AT[str(s)] = counter

        at = list(tmp_AT.values())
        for elem in at:
            if int(elem) < 0:
                print("good's at problem = " + str(g))
                sys.exit(-1)
        check = check_if_at_is_continous(at, period_of_continuity)

        if check is True:
            print("good = " + str(g) + " is CONTINOUS")
            tmp_dict[str(g)] = at
        else:
            print("good = " + str(g) + " not continous")
            tmp_dict_not_cont[str(g)] = at

    curr.close()
    conn.close()

    with open(f1, 'w', encoding='utf-8') as outfile:
        json.dump(tmp_dict, outfile, indent=4)
        outfile.close()

    with open(f2, 'w', encoding='utf-8') as outfile:
        json.dump(tmp_dict_not_cont, outfile, indent=4)
        outfile.close()


def my_new_biased():
    f1 = "my_new_continous_AT_with_bias.json"

    artists = []
    f = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        g = int(line_array[0])
        if g not in artists:
            artists.append(g)
    f.close()
    print("TOTAL ARTIST = " + str(len(artists)))

    conn = sqlite3.connect("lastfm.db")
    curr = conn.cursor()
    curr.execute("""SELECT distinct slot from adoptions order by slot asc""")
    weeks = curr.fetchall()

    tmp_dict = {}
    tmp_dict_not_cont = {}

    for g in artists:
        # initialize the dict
        tmp_AT = {}
        for w in weeks:
            w = w[0]
            tmp_AT[str(w)] = 0

        curr = conn.cursor()
        curr.execute("""SELECT distinct adopter, slot from adoptions where good='%s' order by slot asc""" % str(g))
        adopters = curr.fetchall()

        for elem in adopters:
            a = elem[0]
            s = elem[1]
            counter = tmp_AT[str(s)]
            counter += 1
            tmp_AT[str(s)] = counter

        at = list(tmp_AT.values())
        for elem in at:
            if int(elem) < 0:
                print("good's at problem = " + str(g))
                sys.exit(-1)

        check = True

        if check is True:
            print("good = " + str(g) + " is CONTINOUS")
            tmp_dict[str(g)] = at
        else:
            print("good = " + str(g) + " not continous")
            tmp_dict_not_cont[str(g)] = at

    curr.close()
    conn.close()

    with open(f1, 'w', encoding='utf-8') as outfile:
        json.dump(tmp_dict, outfile, indent=4)
        outfile.close()


def new_ground_truth_file():
    file_array = []
    ground_truth_file_directory = "info_data/"
    filename = ground_truth_file_directory + "rank_artists_with_top_listeners_made_by_seed_users"
    file_array.append(filename)
    filename = ground_truth_file_directory + "rank_artists_with_top_listeners"
    file_array.append(filename)
    filename = ground_truth_file_directory + "rank_artists_with_top_playcounts_made_by_seed_users"
    file_array.append(filename)
    filename = ground_truth_file_directory + "rank_artists_with_top_playcounts"
    file_array.append(filename)
    filename = ground_truth_file_directory + "rank_artists_with_top_google_searches"
    file_array.append(filename)
    filename = ground_truth_file_directory + "rank_artists_with_top_adoptions"
    file_array.append(filename)

    for filename in file_array:
        current_goods = []
        f = open(filename, "r", encoding="utf-8")
        for line in f:
            line_array = line.split("::")
            good = int(line_array[0])
            if good not in current_goods:
                current_goods.append(good)
        f.close()

        # percentage = 7
        percentage = 38
        num_lines = len(current_goods)
        v_i = int((num_lines * percentage) / 100)

        file_array = filename.split("rank_artists_with_top_")
        new_filename = ground_truth_file_directory + file_array[1] + "_ground_truth_file_v_" + str(percentage)
        g = open(new_filename, "a", encoding="utf-8")
        for good in current_goods:
            if v_i > 0:
                str_to_write = str(good) + "," + "1" + "\n"
                v_i = v_i - 1
            else:
                str_to_write = str(good) + "," + "-1" + "\n"
            g.write(str_to_write)
        g.close()


def evaluate_clustering_results(not_continous_file, clustering_res_file, ground_truth_file):
    with open(clustering_res_file, 'r', encoding='utf-8') as infile:
        clustering = json.load(infile)
        infile.close()

    hits = clustering["cluster_0"]
    flops = clustering["cluster_1"]
    for i in range(0, len(hits)):
        hits[i] = int(hits[i])

    for i in range(0, len(flops)):
        flops[i] = int(flops[i])

    with open(not_continous_file, 'r', encoding='utf-8') as infile:
        nc = json.load(infile)
        infile.close()
    not_cont_goods = list(nc.keys())
    for i in range(0, len(not_cont_goods)):
        not_cont_goods[i] = int(not_cont_goods[i])

    # CONSTRUCT_GROUND_TRUTH_FILE
    f = open(ground_truth_file, "a", encoding="utf-8")
    for g in hits:
        str_to_write = str(g) + ",1\n"
        f.write(str_to_write)
        f.flush()

    for g in flops:
        str_to_write = str(g) + ",-1\n"
        f.write(str_to_write)
        f.flush()

    for g in not_cont_goods:
        str_to_write = str(g) + ",-1\n"
        f.write(str_to_write)
        f.flush()
    f.close()


def collect_AT(continuos_AT_file):
    ddata = []
    AT_goods_map = {}

    with open(continuos_AT_file, 'r', encoding='utf-8') as infile:
        at_dict = json.load(infile)
        infile.close()

    goods = list(at_dict.keys())
    goods_at = list(at_dict.values())

    univocal = []
    for g in goods:
        if int(g) not in univocal:
            univocal.append(int(g))
    print(len(univocal))

    for i in range(0, len(goods)):
        total_adoptions = 0
        for elem in goods_at[i]:
            total_adoptions += int(elem)

        # compute adoption rate
        adoption_trand_rate = []
        for elem in goods_at[i]:
            l = int(elem) / total_adoptions
            adoption_trand_rate.append(float("{0:.2f}".format(l)))

        ddata.append(pd.Series(adoption_trand_rate))
        # N.B: multiple goods may have the same AT
        try:
            goods_list = AT_goods_map[str(pd.Series(adoption_trand_rate))]
            goods_list.append(int(goods[i]))
            AT_goods_map[str(pd.Series(adoption_trand_rate))] = goods_list
        except KeyError:
            AT_goods_map[str(pd.Series(adoption_trand_rate))] = [int(goods[i])]

    return ddata, AT_goods_map


def elbow_criterion_for_optimal_k_for_kmeans(data, x_label):
    sse = {}
    for k in range(1, 21):
        print("sse for k = " + str(k))
        kmeans = KMeans(n_clusters=k, max_iter=100000000).fit(data)
        # data["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel(x_label)
    plt.ylabel("SSE")
    plt.show()


def silhoutte_coefficent_for__optimal_k_for_kmeans(data, x_label):
    sc = {}
    for n_cluster in range(2, 21):
        print("silhoutte coefficent for n_cluster = " + str(n_cluster))
        kmeans = KMeans(n_clusters=n_cluster).fit(data)
        label = kmeans.labels_
        sil_coeff = silhouette_score(data, label, metric='euclidean')
        sc[n_cluster] = sil_coeff
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

    plt.figure()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    plt.plot(list(sc.keys()), list(sc.values()))
    plt.xlabel(x_label)
    plt.ylabel("Silhouette Coefficient")
    plt.show()


def DTWDistance(s1, s2):
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


def k_means_clust(data, num_clust, num_iter):
    # random.sample(population, k) => Return a k length list of unique elements chosen from the population
    centroids = sample(data, num_clust)
    counter = 0
    last_assignements = {}
    for n in range(num_iter):
        counter += 1
        print(counter)
        assignments = {}
        # assign data points to clusters, looping over each time series and having an automatic counter
        for ind, i in enumerate(data):
            min_dist = float('inf')
            closest_clust = None
            for c_ind, j in enumerate(centroids):
                cur_dist = DTWDistance(i, j)
                if cur_dist < min_dist:
                    min_dist = cur_dist
                    closest_clust = c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)  # append  time series counter
            else:
                assignments[closest_clust] = [ind]  # SELF/ADDES

        # recalculate centroids of clusters
        for key in assignments.keys():
            clust_sum = 0
            for k in assignments[key]:
                clust_sum = clust_sum + data[k]
            centroids[key] = [m / len(assignments[key]) for m in clust_sum]

        if n == num_iter - 1:
            last_assignements = assignments

    return centroids, last_assignements


def cluster_adoption_trends(ddata, AT_goods_map, num_clusters, res_clustering_file, x_label, swap):
    """
        Function which clusters the AT with k-means
    """

    centroids, last_assignements = k_means_clust(ddata, num_clusters, 10)  # num iterations
    print(centroids)

    if num_clusters == 2:
        goods_in_cluster_0 = []
        goods_in_cluster_1 = []
        ddata_labels = {}
        for ind, i in enumerate(ddata):
            ddata_labels[str(ind)] = i

        for cluster, cluster_elements in last_assignements.items():
            for elem in cluster_elements:
                # retrive corrisponding adoption trand
                AT = ddata_labels[str(elem)]

                # retrieve the good which corresponds to this AT (first element of goods which have this AT)
                good_list = AT_goods_map[str(AT)]

                good = good_list[0]
                del good_list[0]  # remove this element from list

                if int(cluster) == 0:
                    goods_in_cluster_0.append(good)
                else:
                    goods_in_cluster_1.append(good)

        tmp = {}
        if swap is True:
            tmp["cluster_0"] = goods_in_cluster_1
            tmp["cluster_1"] = goods_in_cluster_0
        else:
            tmp["cluster_0"] = goods_in_cluster_0
            tmp["cluster_1"] = goods_in_cluster_1

        with open(res_clustering_file, 'w', encoding='utf-8') as outfile:
            json.dump(tmp, outfile, indent=4)
            outfile.close()

        print("CLUSTER 0")
        print(goods_in_cluster_0)
        print(len(goods_in_cluster_0))
        print("CLLUSTER 1")
        print(goods_in_cluster_1)
        print(len(goods_in_cluster_1))

    fig, ax = plt.subplots()
    # We need to draw the canvas, otherwise the labels won't be positioned and
    # won't have values yet.
    fig.canvas.draw()

    print(len(centroids))

    if num_clusters == 2:
        first = True
        for i in centroids:
            if first is True:
                ax.plot(i, c="green", label="cluster 0")
                first = False
            else:
                ax.plot(i, c="red", label="cluster 1")
    elif num_clusters == 3:
        first = True
        second = True
        for i in centroids:
            if first is True:
                ax.plot(i, c="red", label="cluster 0")
                first = False
            elif second is True:
                ax.plot(i, c="green", label="cluster 1")
                second = False
            else:
                ax.plot(i, c="blue", label="cluster 2")
    else:
        first = True
        second = True
        third = True
        for i in centroids:
            if first is True:
                ax.plot(i, c="blue", label="cluster 2")
                first = False
            elif second is True:
                ax.plot(i, c="red", label="cluster 0")
                second = False
            elif third is True:
                ax.plot(i, c="orange", label="cluster 1")
                third = False
            else:
                ax.plot(i, c="green", label="cluster 3")

    # add extra y tick
    y_axis_values = list(plt.yticks()[0])
    del y_axis_values[0]
    plt.yticks(y_axis_values)

    # add extra x tick
    x_axis_values = list(plt.xticks()[0])
    del x_axis_values[0]
    plt.xticks(x_axis_values)

    plt.xlabel(x_label)
    plt.ylabel("Adoption rate")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    adoption_log_file = "adoption_log.csv"
    main_db_name = "lastfm.db"
    friendship_db_name = "friendship.db"
    ground_truth_file = "data/global_AT_ground_truth_file5"

    # Load Adoption log Data
    print( "Loading data")
    load_main_db_data(adoption_log_file, main_db_name)
    load_friendship_db_data("filtered_edge_list.csv", friendship_db_name, main_db_name)

    conn = sqlite3.connect("%s" % friendship_db_name)
    curr = conn.cursor()
    curr.execute("""SELECT * from friendship""")
    res = curr.fetchall()
    print("elem inserted = " + str(len(res)))

    compute_GLOBAL_adoption_trand()
    new_ground_truth_file()
    my_new_biased()


    # my_new_biased()
    ddata, AT_goods_map = collect_AT("my_new_continous_AT_with_bias.json")
    cluster_adoption_trends(ddata, AT_goods_map, 2, "my_new_continous_AT_with_bias_clustering_res.json", "Weeks", True)
    evaluate_clustering_results("empty.json", "my_new_continous_AT_with_bias_clustering_res.json", "my_new_continous_AT_with_bias_ground_truth_file")

    new_ground_truth_file()
