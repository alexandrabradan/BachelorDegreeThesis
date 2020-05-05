import os
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
from matplotlib.font_manager import FontProperties
import matplotlib.cm as cm
import matplotlib.colors as colors
from pylab import *
import collections
import sqlite3
import ast
import tqdm
from collections import Counter

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


def check_if_user_follow_another_user(first_user, second_user, friendship_db):
    """
        Function which checks if the first user follows the second user
    """
    conn = sqlite3.connect("%s" % friendship_db)
    curr = conn.cursor()
    curr.execute("""SELECT * from friendship where source='%s' and target='%s'""" % (str(first_user), str(second_user)))
    res = curr.fetchall()

    if res is None or len(res) == 0:
        return False
    else:
        return True


def get_diffusion_tree_directory_name(diffusion_tree_directory_number):
    """
        Function which decode the number passed by argument as the name of a directory
        :param diffusion_tree_directory_number: number assigned to one of the Diffusion tree's
        directories
    """

    if diffusion_tree_directory_number == 1:
        return "alternative/"
    elif diffusion_tree_directory_number == 2:
        return "blues/"
    elif diffusion_tree_directory_number == 3:
        return "classical/"
    elif diffusion_tree_directory_number == 4:
        return "country/"
    elif diffusion_tree_directory_number == 5:
        return "dance/"
    elif diffusion_tree_directory_number == 6:
        return "electronic/"
    elif diffusion_tree_directory_number == 7:
        return "hip-hop_rap/"
    elif diffusion_tree_directory_number == 8:
        return "jazz/"
    elif diffusion_tree_directory_number == 9:
        return "latin/"
    elif diffusion_tree_directory_number == 10:
        return "pop/"
    elif diffusion_tree_directory_number == 11:
        return "r&b_soul/"
    elif diffusion_tree_directory_number == 12:
        return "reggae/"
    elif diffusion_tree_directory_number == 13:
        return "rock/"
    else:
        print("No associated directory name with number = " + str(diffusion_tree_directory_number))
        sys.exit(-1)


def check_if_leader_followers_are_also_followings(friendship_db, reciprocal_rate_file_name):
    """
        Function which iterates over all leaders' Diffusion Trees
        (each one of them being stored in the directory with the name relative
        the main music genre of the item they're related), retrieve's
        Diffusion Trees's first level (leader's followers) and check
        if the leader follow them back
    """

    reciprocal_rate_file = open(reciprocal_rate_file_name, "a", encoding="utf-8")
    reciprocal_rate_file.write("leader\taction\tdiffusion_tree_followings\tdiffusion_tree_followers\treciprocal_rate\n")

    for i in range(1, 14):
        diffusion_tree_directory = get_diffusion_tree_directory_name(i)

        list_of_files = os.listdir(diffusion_tree_directory)
        for file in tqdm.tqdm(list_of_files):
            file_path = diffusion_tree_directory + file

            str_array = file.split("_for_action_")
            leader = int(str_array[0])
            str_array2 = str_array[1].split("_diffusion_tree")
            action = int(str_array2[0])
            diffusion_tree_followers = 0  # first level nodes
            diffusion_tree_followings = 0  # first level nodes that leader follows back

            f = open(file_path, "r", encoding="utf-8")
            for line in f:
                line_array = line.split("\t")
                u = int(line_array[0])
                v = int(line_array[1].replace("\n", ""))

                if u == leader:
                    # I'm iterating over Diffusion tree's first level
                    diffusion_tree_followers += 1

                    # check if the leader is following the follower back
                    following = check_if_user_follow_another_user(str(leader), str(v), friendship_db)
                    if following is True:
                        diffusion_tree_followings += 1
            f.close()

            if diffusion_tree_followers != 0:
                reciprocal_rate = float(diffusion_tree_followings / diffusion_tree_followers)
            else:
                reciprocal_rate = 0.0

            reciprocal_rate_file.write(f"{leader}\t{action}\t{diffusion_tree_followings}\t{diffusion_tree_followers}\t{reciprocal_rate}\n")

    reciprocal_rate_file.close()


def check_if_user_is_a_hit_savvy(user, hit_savvies_each_success_def):
    """
        Function which checks if the user passed by argument is present in the list of list relatetive
        to each set of Hit-savvies identified by the 7 different success definitions
    :param user:  user to check if it's a Hit-Savvy
    :param hit_savvies_each_success_def: list of lists, within where each list i correspond to the Hit-Savvy set
                                          identified by the success definition i+1
    """
    i = 0
    indexes = []
    for hit_savvies_list in hit_savvies_each_success_def:
        if user in hit_savvies_list:
            indexes.append(i)
        i += 1
    if len(indexes) == 0:
        return False, indexes
    else:
        return True, indexes


def get_who_influence_hit_savvies_who_are_leaders(diffusion_trees_dir, success_dir):
    """
        Function which for Hit-savvy found by each success definition, verify keep_tracks
        of whom the Hit-savvy influence as a leader or by whom leader the Hit-savvy is influenced
    """
    # list of lists, where each within list i == Hit-Savvies found with succ. def i + 1
    hit_savvies_each_success_def = []

    # what to write to file hit_savvies_not_leaders_file + 1 (each element is a tuple)
    hit_savvies_not_leaders = [[], [], [], [], [], [], []]

    # what to write to file hit_savvies_leaders_file + 1  (each element is a couple)
    hit_savvies_leaders = [[], [], [], [], [], [], []]

    # iterate over success definitions' Hit-Savvies
    for success_def in range(1, 8):
        complete_success_dir_name = success_dir + "Def" + str(success_def) + "/" + success_dir

        if diffusion_trees_dir == "leaders_diffusion_trees/":
            # create hit_savvies_not_leaders file
            hit_savvies_not_leaders_file_name = complete_success_dir_name + "hit_savvies_not_leaders" + str(success_def) \
                                                + ".csv"
            hit_savvies_not_leaders_file = open(hit_savvies_not_leaders_file_name, "w", encoding="utf-8")
            hit_savvies_not_leaders_file.write("hit_savvy_not_leader\tleader\taction\n")
            hit_savvies_not_leaders_file.close()

            # create hit_savvies_leaders path
            hit_savvies_leaders_file_name = complete_success_dir_name + "hit_savvies_leaders" + str(success_def) \
                                                + ".csv"
            hit_savvies_leaders_file = open(hit_savvies_leaders_file_name, "w", encoding="utf-8")
            hit_savvies_leaders_file.write("hit_savvy_as_leader\taction\n")
            hit_savvies_leaders_file.close()

        # retrieve current success definition's Hit-Savvies
        success_file_path = complete_success_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
        with open(success_file_path, 'r', encoding='utf-8') as infile:
            ht_info_dict = json.load(infile)

        adopters = [int(x) for x in list(ht_info_dict.keys())]
        hit_savvies_each_success_def.append(adopters)

    # iterate over the Diffusion Trees
    for i in tqdm.tqdm(range(1, 14)):
        diffusion_tree_directory = diffusion_trees_dir + get_diffusion_tree_directory_name(i)

        list_of_files = os.listdir(diffusion_tree_directory)
        for file in tqdm.tqdm(list_of_files):
            file_path = diffusion_tree_directory + file

            str_array = file.split("_for_action_")
            leader = int(str_array[0])
            str_array2 = str_array[1].split("_diffusion_tree")
            action = int(str_array2[0])

            # check if the leader is a Hit_savvy
            isHitSavvy, indexes = check_if_user_is_a_hit_savvy(leader, hit_savvies_each_success_def)
            if isHitSavvy is True:
                for ind in indexes:
                    if leader not in hit_savvies_leaders[ind]:
                        hit_savvies_leaders[ind].append((leader, action))

            # check if in the current Diffusion Trees some Hit-savvies for some success definition are present
            # (this means that they are influenced by the leader of this Diffusion Tree)
            f = open(file_path, "r", encoding="utf-8")
            for line in f:
                line_array = line.split("\t")
                u = int(line_array[0])
                v = int(line_array[1].replace("\n", ""))

                # check if user v is a Hit_Savvy
                isHitSavvy, indexes = check_if_user_is_a_hit_savvy(v, hit_savvies_each_success_def)
                if isHitSavvy is True:
                    for ind in indexes:
                        if v not in hit_savvies_not_leaders[ind]:
                            hit_savvies_not_leaders[ind].append((v, leader, action))
            f.close()

    # write info on file
    for success_def in range(0, 7):
        complete_success_dir_name = success_dir + "Def" + str(success_def + 1) + "/" + success_dir

        hit_savvies_not_leaders_file_name = complete_success_dir_name + "hit_savvies_not_leaders" + str(success_def + 1) \
                                            + ".csv"
        hit_savvies_not_leaders_file = open(hit_savvies_not_leaders_file_name, "a", encoding="utf-8")
        for tuplee in hit_savvies_not_leaders[success_def]:
            hit_savvies_not_leaders_file.write(f"{tuplee[0]}\t{tuplee[1]}\t{tuplee[2]}\n")
        hit_savvies_not_leaders_file.close()

        hit_savvies_leaders_file_name = complete_success_dir_name + "hit_savvies_leaders" + str(success_def + 1) \
                                        + ".csv"
        hit_savvies_leaders_file = open(hit_savvies_leaders_file_name, "a", encoding="utf-8")
        for pairr in hit_savvies_leaders[success_def]:
            hit_savvies_leaders_file.write(f"{pairr[0]}\t{pairr[1]}\n")
        hit_savvies_leaders_file.close()


def get_music_genre_from_encoding(music_encoding):
    # N.B. profile music tags are encoded starting from 1
    main_tags = {"1": "alternative", "2": "blues", "3": "classical", "4": "country", "5": "dance", "6": "electronic",
                 "7": "hip-hop/rap", "8": "jazz", "9": "latin", "10": "pop", "11": "r&b/soul", "12": "reggae", "13": "rock"}
    try:
        tag = main_tags[str(music_encoding)]
        return tag
    except KeyError:
        print("Music encoding = " + str(music_encoding) + " doesn't have a music tag associated")
        sys.exit(-1)


def retrieve_users_main_genre(users_list):
    """
        Function which retrives from "profile_mu_and_sigma.json" file the
        main music genre of the users present in the list passed by argument
    """
    with open("profile_mu_and_sigma.json", 'r', encoding='utf-8') as infile:
        users_profile = json.load(infile)

    users_genres = []
    for user in users_list:
        try:
            main_genre = users_profile[str(user)]["hat_gu"]
            users_genres.append(get_music_genre_from_encoding(main_genre))
        except KeyError:
            print("User = " + str(user) + " hasn't a profile")
            sys.exit(-1)
    return users_genres


def get_artist_main_genre(artist_encoding):
    f = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        a = int(line_array[0])
        genre_encoding = line_array[1].replace("\n", "")

        if a == int(artist_encoding):
            return genre_encoding
    f.close()
    return ""


def evaluate_hit_savvies_hits(success_dir):
    multi_x = []

    # iterate over success definitions' Hit-Savvies
    for success_def in range(1, 8):
        complete_success_dir_name = success_dir + "Def" + str(success_def) + "/" + success_dir

        # retrieve current success definition's Hit-Savvies
        success_file_path = complete_success_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
        with open(success_file_path, 'r', encoding='utf-8') as infile:
            ht_info_dict = json.load(infile)

        goods_genres = []
        for ad in list(ht_info_dict.keys()):
            g_list = ht_info_dict[str(ad)]["goods"]
            for g in g_list:
                genre = get_artist_main_genre(int(g))
                if genre == "":
                    print("artist = " + str(g) + ", doesn't have a main genre")
                    sys.exit(-1)
                else:
                    goods_genres.append(str(genre))
        multi_x.append(goods_genres)

    labels = ["seed-user listeners", "seed-user playcounts", "google trends", "last.fm listeners", "last.fm playcounts",
              "AT", "actions"]

    # draw hist main music genres vs HT_laeders
    draw_histogram_music_x_axis(multi_x, "Tags", "# Hit_Savvies", labels)


def evaluate_hit_savvies(success_dir):
    multi_x = []
    multi_x2 = []
    multi_x3 = []
    multi_x4 = []

    # iterate over success definitions' Hit-Savvies
    for success_def in range(1, 8):
        complete_success_dir_name = success_dir + "Def" + str(success_def) + "/" + success_dir

        # retrieve current success definition's Hit-Savvies
        success_file_path = complete_success_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
        with open(success_file_path, 'r', encoding='utf-8') as infile:
            ht_info_dict = json.load(infile)

        adopters = [int(x) for x in list(ht_info_dict.keys())]
        print("tot HT success def" + str(success_def) + " = " + str(len(adopters)))

        genres_spread_by_ht_leaders = []
        genres_influenced_ht = []

        # retrieve HT Leaders for current success definition
        hit_savvies_leaders_file_name = complete_success_dir_name + "hit_savvies_leaders" + str(success_def) \
                                        + ".csv"
        print(hit_savvies_leaders_file_name)
        data2 = pd.read_csv(hit_savvies_leaders_file_name, delimiter="\t",
                                                         usecols=["hit_savvy_as_leader", "action"], encoding="utf-8")
        unicoval_leaders = []
        actions = []  # a Leader HT may be leader for more  than one item
        for row in data2.itertuples():
            a = int(row.hit_savvy_as_leader)
            g = int(row.action)

            if a not in unicoval_leaders:
                unicoval_leaders.append(a)

            actions.append(g)
            genre = get_artist_main_genre(g)
            if genre == "":
                print("artist = " + str(g) + ", doesn't have a main genre")
                sys.exit(-1)
            genres_spread_by_ht_leaders.append(genre)

        print("HT Leaders = " + str(len(unicoval_leaders)))
        print("tot actions = " + str(len(actions)))

        # retrive Hit-Savvyes which are influenced by a Leader (latter I'll check by whom he is influenced)
        hit_savvies_not_leaders_file_name = complete_success_dir_name + "hit_savvies_not_leaders" + str(success_def) \
                                            + ".csv"
        print(hit_savvies_not_leaders_file_name)
        data = pd.read_csv(hit_savvies_not_leaders_file_name, delimiter="\t",
                           usecols=["hit_savvy_not_leader", "leader", "action"], encoding="utf-8")
        unicoval_not_leaders = []
        leaders_who_influence = []
        actions_influenced = []  # an influenced HT may be influenced by more than on item from the same Leader
        for row in data.itertuples():
            a = int(row.hit_savvy_not_leader)
            l = int(row.leader)
            g = int(row.action)

            if a not in unicoval_not_leaders:
                unicoval_not_leaders.append(a)

            if l not in leaders_who_influence:
                leaders_who_influence.append(l)

            actions_influenced.append(g)
            genre = get_artist_main_genre(g)
            if genre == "":
                print("artist = " + str(g) + ", doesn't have a main genre")
                sys.exit(-1)
            genres_influenced_ht.append(genre)

        print("influenced HT = " + str(len(unicoval_not_leaders)))
        print("HT NOT influeneced = " + str(len(adopters) - len(unicoval_not_leaders) - len(unicoval_leaders)))
        print("actions_influenced = " + str(len(actions_influenced)))
        print("Hit_Savvies Leaders who influenced HT NOT Leaders = " \
              + str(len(set(unicoval_leaders).intersection(set(leaders_who_influence)))))
        print("actions with which influence = " \
              + str(len(set(actions).intersection(set(actions_influenced)))))
        print("HT_l AND HT_i at same time = " + str(len(set(unicoval_leaders).intersection(set(unicoval_not_leaders)))))

        multi_x.append(genres_spread_by_ht_leaders)
        multi_x2.append(genres_influenced_ht)

        # (neutral Hit_Savvies = neutral_Hit_Savvies - HT_leaders - HT_influenced)
        tmp_set = set(adopters).difference(set(unicoval_leaders))
        neutral_ht = [int(x) for x in list(set(tmp_set).difference(set(unicoval_not_leaders)))]
        multi_x3.append(retrieve_users_main_genre(neutral_ht))

        if success_def == 1:
            color = "royalblue"
        elif success_def == 2:
            color = "orange"
        elif success_def == 3:
            color = "forestgreen"
        elif success_def == 4:
            color = "firebrick"
        elif success_def == 5:
            color = "blueviolet"
        elif success_def == 6:
            color = "sienna"
        elif success_def == 7:
            color = "hotpink"

    labels = ["seed-user listeners", "seed-user playcounts", "google trends", "last.fm listeners", "last.fm playcounts",
              "AT", "actions"]

    # draw hist main music genres vs HT_laeders
    draw_histogram_music_x_axis(multi_x, "Tags", "# Leader Hit-Savvies", labels)

    # draw hist main music genres vs HT_not_leaders
    draw_histogram_music_x_axis(multi_x2, "Tags", "# influenced Hit-Savvies", labels)

    # draw hist main music genres vs neutral HT
    draw_histogram_music_x_axis(multi_x3, "Tags", "# neutral Hit-Savvies", labels)


def draw_histogram_music_x_axis(multi_x, x_label, y_label, labels):

    fig, ax1 = plt.subplots()
    """i = 1
    sorted_multi_x = []
    sorted_multi_y = []
    for x in multi_x:
        frequency_dict = Counter(x)
        # sort dict by descendent value
        sorted_frequency_dict = {k: v for k, v in sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)}
        print("success def = " + str(i))
        i += 1
        print(frequency_dict)
        new_x = list(sorted_frequency_dict.keys())
        sorted_multi_x.append(new_x)
        print(new_x)
        y = list(sorted_frequency_dict.values())
        sorted_multi_y.append(y)
        print(y)
        print()"""

    # draw hystogram
    # ax1.bar(sorted_multi_x, sorted_multi_y, align='center', log=True, labels=labels)
    ax1.hist(multi_x, align='mid', label=labels, log=True)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)  # rotate x' labels in order to prevent crowdiness

    """y_axis_values = list(plt.yticks()[0])
    del y_axis_values[0]
    del y_axis_values[0]
    del y_axis_values[(len(y_axis_values)-1)]
    plt.yticks(y_axis_values)"""

    # Put a legend below current axis
    # ax1.legend(loc='upper center', fancybox=True, shadow=True, ncol=3)

    plt.legend()
    plt.show()
    plt.close("all")


def draw_histogram(x, x_label, y_label):
    fig, ax1 = plt.subplots()

    ax1.hist(x, log=True, color='red', rwidth=0.9)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    y_axis_values = list(plt.yticks()[0])
    print(y_axis_values)
    # del y_axis_values[0]
    # del y_axis_values[0]
    plt.yticks(y_axis_values)

    np_x = np.array(x)
    print("np_x.mean() = " + str(np_x.mean()))
    plt.axvline(np_x.mean() * 1.1, color='lime', linestyle='dashed', linewidth=3,
                label='Mean: {:.2f}'.format(np_x.mean()))

    plt.legend()
    plt.show()


def get_reciprocal_friendship_data(reciprocal_file_name1, reciprocal_file_name2):
    x = []
    data = pd.read_csv(reciprocal_file_name1, delimiter="\t", usecols=["reciprocal_rate"], encoding="utf-8")
    for row in data.itertuples():
        x.append(row.reciprocal_rate)

    data2 = pd.read_csv(reciprocal_file_name2, delimiter="\t", usecols=["reciprocal_rate"], encoding="utf-8")
    for row in data2.itertuples():
        x.append(row.reciprocal_rate)
    return x


friendship_db = "friendship.db"
reciprocal_file_name1 = "info_data/reciprocal_friendship_rate1.csv"  # more than 4 nodes diffusion trees
reciprocal_file_name2 = "info_data/reciprocal_friendship_rate1.csv"  # less than 4 nodes diffusion trees
check_if_leader_followers_are_also_followings(friendship_db, reciprocal_file_name1)
check_if_leader_followers_are_also_followings(friendship_db, reciprocal_file_name2)

x = get_reciprocal_friendship_data(reciprocal_file_name1, reciprocal_file_name2)
draw_histogram(x, "reciprocal friendship rate", "# Diffusion trees")


get_who_influence_hit_savvies_who_are_leaders("leaders_diffusion_trees/", "part2/")
get_who_influence_hit_savvies_who_are_leaders("less_then_4_nodes/", "part2/")
evaluate_hit_savvies("part2/")
evaluate_hit_savvies_hits("part2/")
