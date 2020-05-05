import os
import sys
import tqdm
import ast
import sqlite3
from scipy.stats import skew
from scipy.stats import binned_statistic
from sklearn.metrics import jaccard_similarity_score
from scipy.spatial.distance import hamming
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as colors
import seaborn as sns
import json

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


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


def draw_multi_bar(multi_x, multi_y,  x_label, y_label, labels):
    fig, ax1 = plt.subplots()

    for i in range(0, len(multi_x)):
        ax1.bar(multi_x[i], multi_y[i], label=labels[i])

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


def draw_split_and_predictive_period_scores(dir_multi_precision, dir_multi_npv, dir_multi_recall2,
                                                    dir_multi_specificity2, success_def, dir_name):
    labels = ["PPV", "NPV", "TPR", "TNR"]
    split = []
    x = [[], [], [], [], []]
    for i in range(0, 5):
        for j in range(1, (len(dir_multi_precision[i]) + 1)):
            x[i].append(j)

        if i == 0:
            split.append("a")
        elif i == 1:
            split.append("b")
        elif i == 2:
            split.append("c")
        elif i == 3:
            split.append("d")
        elif i == 4:
            split.append("e")

        if len(dir_multi_precision[i]) == 0:
            print("missing success_def" + str(success_def) + " split " + str(split[i]))
            return

    for i in range(0, 5):

        fig, ax1 = plt.subplots()
        ax1.plot(x[i], dir_multi_precision[i], 'r', label=labels[0])
        ax1.plot(x[i], dir_multi_npv[i], 'r--', label=labels[1])
        ax1.plot(x[i], dir_multi_recall2[i], 'b', label=labels[2])
        ax1.plot(x[i], dir_multi_specificity2[i], 'b--', label=labels[3])

        plt.xlabel("Months observed")
        plt.ylabel("Score")

        title = "Success def " + str(success_def) + "'s split " + str(split[i])
        plt.title(title)
        # Show the grid lines as dark grey lines
        plt.grid(b=True, which='major', color='#666666', linestyle='--')

        ax1.set_xlim(xmin=x[i][0])
        ax1.set_xlim(xmax=x[i][len(x[i])-1])
        # plt.xticks([1] + list(plt.xticks()[0]))

        plt.legend()
        figure_location = "Plotting" + dir_name + "prediction_res/" + str(dir_name).replace("/", "") \
                          + "_success_def_" + str(success_def) + "_" + str(split[i])
        plt.savefig(figure_location)
        plt.close("all")


def draw_unclassified_items(dir_multi_unclassified, success_def, dir_name):
    total_items = [11109.0, 6686.0,  4015.0, 2255.0, 1053.0]
    split = []
    x = [[], [], [], [], []]
    for i in range(0, 5):
        for j in range(1, (len(dir_multi_unclassified[i]) + 1)):
            x[i].append(j)

        if i == 0:
            split.append("a")
        elif i == 1:
            split.append("b")
        elif i == 2:
            split.append("c")
        elif i == 3:
            split.append("d")
        elif i == 4:
            split.append("e")

        if len(dir_multi_unclassified[i]) == 0:
            print("missing success_def" + str(success_def) + " split " + str(split[i]))
            return

    for i in range(0, 5):
        fig, ax1 = plt.subplots()
        label = "tot items = " + str(total_items[i])
        ax1.plot(x[i], dir_multi_unclassified[i], 'ko', label=label)

        plt.xlabel("Months observed")
        plt.ylabel("Unclassified items")

        title = "Success def " + str(success_def) + "'s split " + str(split[i]) + ", unclassified items"
        plt.title(title)
        # Show the grid lines as dark grey lines
        plt.grid(b=True, which='major', color='#666666', linestyle='--')

        plt.legend()
        ax1.set_xlim(xmin=x[i][0])
        ax1.set_xlim(xmax=x[i][len(x[i]) - 1])
        # plt.xticks([1] + list(plt.xticks()[0]))

        figure_location = "Plotting" + dir_name + "prediction_res/" + str(dir_name).replace("/", "") \
                          + "_success_def_" + str(success_def) + "_" + str(split[i]) + "_unclassified"
        plt.savefig(figure_location)
        plt.close("all")

def get_total_adopters_and_goods():
    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct adopter from adoptions""")
    res = cur.fetchall()

    total_adopters = []
    for a in tqdm.tqdm(res):
        a = a[0]
        if a not in total_adopters:
            total_adopters.append(a)
    print("tot adopters = " + str(len(total_adopters)))

    cur = conn.cursor()
    cur.execute("""SELECT distinct good from adoptions""")
    res = cur.fetchall()

    total_goods = []
    for g in tqdm.tqdm(res):
        g = g[0]
        if g not in total_goods:
            total_goods.append(g)
    print("tot goods = " + str(len(total_goods)))
    cur.close()
    conn.close()

    return total_adopters, total_goods


def compare_success_definitions(dir_name, hit_or_flop):
    """
        Function which compare HitSavvies' |HT|, |H| and HF-propensity of the different success
        definitions
    """
    total_adopters, total_goods = get_total_adopters_and_goods()

    multi_hfpropensities = []
    multi_hits = []
    for success_def in range(1, 8):
        complete_dir_name = "Def" + str(success_def) + dir_name
        if hit_or_flop == "hit":
            file_name = complete_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
        else:
            file_name = complete_dir_name + "flop_adopters_success_def" + str(success_def) + ".json"

        with open(file_name, 'r', encoding='utf-8') as infile:
            info_dict = json.load(infile)

        adopters = list(info_dict.keys())
        hf_prop = []
        hits_covered = []
        num_hits_for_hitsavvy = []
        for ad in list(info_dict.keys()):
            hf_prop.append(info_dict[str(ad)]["value"])
            for g in info_dict[str(ad)]["goods"]:
                if g not in hits_covered:
                    hits_covered.append(g)
            num_hits_for_hitsavvy.append(len(info_dict[str(ad)]["goods"]))
        multi_hfpropensities.append(hf_prop)
        multi_hits.append(num_hits_for_hitsavvy)

        # retrive total number of hits identified by the current success definition
        ground_truth_file = ""
        if success_def == 1:
            ground_truth_file = "info_data/listeners_made_by_seed_users_ground_truth_file_v_38"
        elif success_def == 2:
            ground_truth_file = "info_data/playcounts_made_by_seed_users_ground_truth_file_v_38"
        elif success_def == 3:
            ground_truth_file = "info_data/google_searches_ground_truth_file_v_38"
        elif success_def == 4:
            ground_truth_file = "info_data/listeners_ground_truth_file_v_38"
        elif success_def == 5:
            ground_truth_file = "info_data/playcounts_ground_truth_file_v_38"
        elif success_def == 6:
            ground_truth_file = "info_data/my_new_continous_AT_with_bias_ground_truth_file5"
        elif success_def == 7:
            ground_truth_file = "info_data/adoptions_ground_truth_file_v_38"

        f = open(ground_truth_file)
        hits_train, flops_train = [], []
        for l in f:
            l = l.rstrip().split(",")
            if int(l[1]) == -1:
                if l[0] not in flops_train:
                    flops_train.append(l[0])
            else:
                if l[0] not in hits_train:
                    hits_train.append(l[0])
        f.close()

        print("SUCCESS DEF " + str(success_def))
        print("HT = " + str(len(adopters)) + ", HT/tot ad = " + str((len(adopters) / len(total_adopters))*100))
        print("H = " + str(len(hits_covered)) + ", H/tot g = " + str((len(hits_covered) / len(total_goods))*100))
        if hit_or_flop == "hit":
            print("Coverage = " + str((len(hits_covered) / len(hits_train))*100))
        else:
            pass
            print("Coverage = " + str((len(hits_covered) / len(flops_train)) * 100))

    labels = ["seed-user listeners", "seed-user playcounts", "google trends", "last.fm listeners", "last.fm playcounts",
              "AT", "actions"]
    if hit_or_flop == "hit":
        y_label = "# HitSavvy"
    else:
        y_label = "# Flop-adopter"
    draw_multi_histogram(multi_hfpropensities, "HF-propensity", y_label, labels)

    #  Check how many successful items each HitSavvy has spotted (he can be made just one adoption and this adoption
    #  could be successful and I want to verify how many times this happens)
    if hit_or_flop == "hit":
        x_label = "# Hits"
        y_label = "# HitSavvy"
    else:
        x_label = "# Flops"
        y_label = "# Flop-adopter"
    draw_multi_histogram(multi_hits, x_label, y_label, labels)


def compute_jaccard(user1_vals, user2_vals):
    """
        Method to compute Jaccard similarity index between two sets,
        defined as the size of the intersection divided by the size of the union of 2 lists
    """
    intersection = user1_vals.intersection(user2_vals)
    union = user1_vals.union(user2_vals)
    jaccard = float(len(intersection)/(len(user1_vals) + len(user2_vals) - len(union)))
    return jaccard


def compare_success_definitions_hitsavvies(dir_name, hit_or_flop):
    """
        Function which compare HitSavvies' HT and H among the different
        success definitions
    """
    success_def_hitsavvies = []
    success_def_hits = []

    # retrieve each success definiton HitSavvies and hits
    for success_def in range(1, 8):
        complete_dir_name = "Def" + str(success_def) + dir_name
        if hit_or_flop == "hit":
            file_name = complete_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
        else:
            file_name = complete_dir_name + "flop_adopters_success_def" + str(success_def) + ".json"

        with open(file_name, 'r', encoding='utf-8') as infile:
            info_dict = json.load(infile)

        adopters = list(info_dict.keys())
        success_def_hitsavvies.append(adopters)
        hits_covered = []
        for ad in list(info_dict.keys()):
            for g in info_dict[str(ad)]["goods"]:
                if g not in hits_covered:
                    hits_covered.append(g)
        success_def_hits.append(hits_covered)

    # compare HitSavvies and hits of each success definition with all the others, checking how many HitSavvies (hits)
    # found are the same and which differ among them
    comparison_dict = {}
    comparison_dict2 = {}
    for success_def in tqdm.tqdm(range(0, 7)):
        current_hitsavvies_list = success_def_hitsavvies[success_def]
        current_hits_list = success_def_hits[success_def]
        for j in range(success_def + 1, 6):
            next_hitsavvies_list = success_def_hitsavvies[j]
            next_hits_list = success_def_hits[j]

            same_ht = []
            only_in_first_ht = []
            only_in_second_ht = []
            for ad in next_hitsavvies_list:
                if ad not in current_hitsavvies_list:
                    only_in_second_ht.append(ad)
                else:
                    same_ht.append(ad)
            for ad in current_hitsavvies_list:
                if ad not in next_hitsavvies_list:
                    only_in_first_ht.append(ad)
            jaccard_ht_score = compute_jaccard(set(current_hitsavvies_list), set(next_hitsavvies_list))
            key = "success_" + str((success_def + 1)) + "&" + "success_" + str((j + 1))
            comparison_dict[key] = {}
            comparison_dict[key]["jaccard_similarity"] = jaccard_ht_score
            comparison_dict[key]["|same_HT|"] = len(same_ht)
            comparison_dict[key]["|HT_only_in_first_def|"] = len(only_in_first_ht)
            comparison_dict[key]["|HT_only_in_second|"] = len(only_in_second_ht)
            comparison_dict[key]["same_HT"] = same_ht
            comparison_dict[key]["HT_only_in_first_def"] = only_in_first_ht
            comparison_dict[key]["HT_only_in_second"] = only_in_second_ht

            same_hit = []
            only_in_first_hit = []
            only_in_second_hit = []
            for ad in next_hits_list:
                if ad not in current_hits_list:
                    only_in_second_hit.append(ad)
                else:
                    same_hit.append(ad)
            for ad in current_hits_list:
                if ad not in next_hits_list:
                    only_in_first_hit.append(ad)
            jaccard_h_score = compute_jaccard(set(current_hits_list), set(next_hits_list))
            key = "success_" + str((success_def + 1)) + "&" + "success_" + str((j + 1))
            comparison_dict2[key] = {}
            comparison_dict2[key]["jaccard_similarity"] = jaccard_h_score
            comparison_dict2[key]["|same_H|"] = len(same_hit)
            comparison_dict2[key]["|H_only_in_first_def|"] = len(only_in_first_hit)
            comparison_dict2[key]["|H_only_in_second|"] = len(only_in_second_hit)
            comparison_dict2[key]["same_H"] = same_hit
            comparison_dict2[key]["H_only_in_first_def"] = only_in_first_hit
            comparison_dict2[key]["H_only_in_second"] = only_in_second_hit

    if hit_or_flop == "hit":
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_hit_savvies_comparison.json"
    else:
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_flop_adopters_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as out:
        json.dump(comparison_dict, out, indent=4)

    if hit_or_flop == "hit":
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_hits_comparison.json"
    else:
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_flops_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as out:
        json.dump(comparison_dict2, out, indent=4)


def compare_active_period(dir_name, hit_or_flop):
    """
    Function which return Hit-Savvy’s active period, which is defined as the average number of consecutive
     weeks in HF(a) having HF-propensity greater than 0
    """
    multi_x = []
    for success_def in range(1, 8):
        complete_dir_name = "Def" + str(success_def) + dir_name

        if hit_or_flop == "hit":
            active_file = complete_dir_name + "active_period_hit_savvies_success_def" + str(success_def) + ".json"
        else:
            active_file = complete_dir_name + "active_period_flop_adopters_success_def" + str(success_def) + ".json"
        with open(active_file, 'r', encoding='utf-8') as infile:
            active_dict = json.load(infile)

        max_active_period = max(list(active_dict.values()))
        print("max_active_period " + str(max_active_period))

        multi_x.append([int(x) for x in list(active_dict.values())])

    labels = ["seed-user listeners", "seed-user playcounts", "google trends", "last.fm listeners", "last.fm playcounts",
              "AT", "actions"]
    if hit_or_flop == "hit":
        y_label = "# HitSavvy"
    else:
        y_label = "# Flop-adopter"
    draw_multi_histogram(multi_x, "Active period (weeks)", y_label, labels)


def compare_percentage_same_hitters(dir_name, hit_or_flop):
    """
        Function which compares the HitSavvies and hits found by each success definition with all the others
    """
    success_def_ht_list = []
    success_def_hits_list = []
    for success_def in range(1, 8):
        complete_dir_name = "Def" + str(success_def) + dir_name

        current_goods = []
        if hit_or_flop == "hit":
            active_file = complete_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
        else:
            active_file = complete_dir_name + "flop_adopters_success_def" + str(success_def) + ".json"
        with open(active_file, 'r', encoding='utf-8') as infile:
            ht_dict = json.load(infile)

        # retrive current success definition HitSavvies
        current_ht_list = list(ht_dict.keys())
        success_def_ht_list.append(current_ht_list)

        # retrive current success definition goods
        for ad in list(ht_dict.keys()):
            for g in ht_dict[str(ad)]["goods"]:
                if g not in current_goods:
                    current_goods.append(g)
        success_def_hits_list.append(current_goods)

    # compare HitSavvies and hits of each success definition with all the others, checking how many HitSavvies (hits)
    # found are the same and which differ among them
    comparison_dict = {}
    comparison_dict2 = {}
    for success_def in tqdm.tqdm(range(0, 7)):
        current_ht_list = success_def_ht_list[success_def]
        current_hits_list = success_def_hits_list[success_def]
        for j in range(success_def+1, 7):
            next_ht_list = success_def_ht_list[j]
            next_hits_list = success_def_hits_list[j]

            same_ht = []
            only_in_first_ht = []
            only_in_second_ht = []
            for ad in next_ht_list:
                if ad not in current_ht_list:
                    only_in_second_ht.append(ad)
                else:
                    same_ht.append(ad)
            for ad in current_ht_list:
                if ad not in next_ht_list:
                    only_in_first_ht.append(ad)
            key = "success_" + str((success_def + 1)) + "&" + "success_" + str((j + 1))
            comparison_dict[key] = {}
            comparison_dict[key]["jaccard_similarity"] = compute_jaccard(set(current_ht_list), set(next_ht_list))
            comparison_dict[key]["same_HT"] = same_ht
            comparison_dict[key]["HT_only_in_first_def"] = only_in_first_ht
            comparison_dict[key]["HT_only_in_second"] = only_in_second_ht

            same_hit = []
            only_in_first_hit = []
            only_in_second_hit = []
            for ad in next_hits_list:
                if ad not in current_hits_list:
                    only_in_second_hit.append(ad)
                else:
                    same_hit.append(ad)
            for ad in current_hits_list:
                if ad not in next_hits_list:
                    only_in_first_hit.append(ad)
            key = "success_" + str((success_def + 1)) + "&" + "success_" + str((j + 1))
            comparison_dict2[key] = {}
            comparison_dict2[key]["jaccard_similarity"] = compute_jaccard(set(current_hits_list), set(next_hits_list))
            comparison_dict2[key]["same_H"] = same_hit
            comparison_dict2[key]["H_only_in_first_def"] = only_in_first_hit
            comparison_dict2[key]["H_only_in_second"] = only_in_second_hit

    if hit_or_flop == "hit":
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_hit_savvies_comparison.json"
    else:
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_flop_adopters_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as out:
        json.dump(comparison_dict, out, indent=4)

    if hit_or_flop == "hit":
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_hits_comparison.json"
    else:
        comparison_file = "Plotting" + dir_name + dir_name.replace("/", "") + "_flops_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as out:
        json.dump(comparison_dict2, out, indent=4)


def draw_m_threshold(dir_name, hit_or_flop):
    """
        Function which draws the threshold of the probability distribuition Π and Δ
    """
    for success_def in range(1, 8):
        complete_dir_name = "Def" + str(success_def) + dir_name
        for i in range(1, 6):
            if i == 1:
                split = "a"
            elif i == 2:
                split = "b"
            elif i == 3:
                split = "c"
            elif i == 4:
                split = "d"
            elif i == 5:
                split = "e"
            if hit_or_flop == "hit":
                hit_model_file = complete_dir_name + "hit_model/hit_model_info_success_def" + str(success_def) \
                                 + "_" + str(split) + ".json"
                hit_success_file = complete_dir_name + "hit_model/hit_model_success_def" + str(success_def) \
                                   + "_" + str(split) + ".json"
            else:
                hit_model_file = complete_dir_name + "flop_model/flop_model_info_success_def" + str(success_def) \
                                 + "_" + str(split) + ".json"
                hit_success_file = complete_dir_name + "flop_model/flop_model_success_def" + str(success_def) \
                                   + "_" + str(split) + ".json"
            with open(hit_model_file, 'r', encoding='utf-8') as infile:
                hit_model_dict = json.load(infile)
            threshold = hit_model_dict["threshold"]
            adoptions = hit_model_dict["adoptions"]

            fig, ax1 = plt.subplots()
            if hit_or_flop == "hit":
                ax = sns.kdeplot(adoptions, shade=True, color="red")
                print("threshold = " + str(threshold))

                line = ax.get_lines()[-1]
                x, y = line.get_data()
                mask = x >= (threshold - 0.01)
                x, y = x[mask], y[mask]
                ax.fill_between(x, y1=y, alpha=0.5, facecolor='red')

                plt.axvline(threshold, color='gray', linestyle='dashed', linewidth=1,
                            label='Threshold: {:.2f}'.format(threshold))

                plt.ylabel("π")
                plt.xlabel("% ^HT")

            else:
                ax = sns.kdeplot(adoptions, shade=True, color="blue")
                print("threshold = " + str(threshold))

                line = ax.get_lines()[-1]
                x, y = line.get_data()
                mask = x >= (threshold - 0.01)
                x, y = x[mask], y[mask]
                ax.fill_between(x, y1=y, alpha=0.5, facecolor='blue')

                plt.axvline(threshold, color='gray', linestyle='dashed', linewidth=1,
                            label='Threshold: {:.2f}'.format(threshold))
                plt.ylabel("δ")
                plt.xlabel("% ^FT")

            ax.set_xlim(xmax=1.0)
            ax.set_ylim(ymin=0)
            ax.set_xlim(xmin=0.0)

            plt.legend()
            title = "Success def " + str(success_def) + "'s split " + str(split)
            plt.title(title)
            threshold_file_res = "Plotting" + str(dir_name) + "prediction_res/" + dir_name.replace("/", "") \
                                 + "_success_def" + str(success_def) + "_" + hit_or_flop + "_threshold_" + str(split)
            plt.savefig(threshold_file_res)
            plt.close("all")


def print_best_split_and_predictive_period(dir_name, best_split_and_predictive_period):
    file_path = dir_name + best_split_and_predictive_period
    with open(file_path, 'r', encoding='utf-8') as infile:
        prediction_res = json.load(infile)

    precision = prediction_res["precision"]
    NPV = prediction_res["NPV"]
    recall1 = prediction_res["recall (with unclassified)"]
    recall2 = prediction_res["recall (without unclassified)"]
    specificity2 = prediction_res["specificity (with unclassified)"]  # in reality is without unclassified

    TP = prediction_res["TP"]
    TN = prediction_res["TN"]
    FP = prediction_res["FP"]
    FN = prediction_res["FN"]
    unclassified = prediction_res["unclassified"]
    specificity1 = TN / (TN + FP + unclassified)  # with unclassified

    accuracy = prediction_res["accuracy"]
    F1 = prediction_res["F1"]

    print("precision = " + str(precision))
    print("NPV = " + str(NPV))
    print("recall1 = " + str(recall1))
    print("recall2 = " + str(recall2))
    print("specificity1 = " + str(specificity1))
    print("specificity2 = " + str(specificity2))
    print("accuracy = " + str(accuracy))
    print("F1 = " + str(F1))


def get_best_split_and_predictiveperiod_success_def(dir_name):

    for success_def in range(1, 8):
        complete_dir_name = "Def" + str(success_def) + dir_name + "/prediction/"
        list_of_files = os.listdir(complete_dir_name)

        tmp_dir_multi_precision = [{}, {}, {}, {}, {}]
        tmp_dir_multi_npv = [{}, {}, {}, {}, {}]
        tmp_dir_multi_recall2 = [{}, {}, {}, {}, {}]
        tmp_dir_multi_specificity2 = [{}, {}, {}, {}, {}]
        tmp_dir_multi_unclassified = [{}, {}, {}, {}, {}]

        dir_multi_precision = [[], [], [], [], []]
        dir_multi_npv = [[], [], [], [], []]
        dir_multi_recall2 = [[], [], [], [], []]
        dir_multi_specificity2 = [[], [], [], [], []]
        dir_multi_unclassified = [[], [], [], [], []]

        best_split_and_predictive_period = ""
        max_precision = -1
        max_npv = -1
        max_recall1 = -1
        max_recall2 = -1
        max_specificity1 = -1
        max_specificity2 = -1
        max_accuracy = -1
        max_f1 = -1

        for file in list_of_files:
            file_path = complete_dir_name + file

            # get current success definition's split
            file_array = file.split("_")
            current_split = str(file_array[3])
            current_predictive_period = int(file_array[4].replace(".json", ""))

            with open(file_path, 'r', encoding='utf-8') as infile:
                prediction_res = json.load(infile)

            precision = prediction_res["precision"]
            NPV = prediction_res["NPV"]
            recall1 = prediction_res["recall (with unclassified)"]
            recall2 = prediction_res["recall (without unclassified)"]
            specificity2 = prediction_res["specificity (with unclassified)"]  # in reality is without unclassified

            TP = prediction_res["TP"]
            TN = prediction_res["TN"]
            FP = prediction_res["FP"]
            FN = prediction_res["FN"]
            unclassified = prediction_res["unclassified"]
            specificity1 = TN / (TN + FP + unclassified)  # with unclassified

            accuracy = prediction_res["accuracy"]
            F1 = prediction_res["F1"]

            if current_split == "a":
                i = 0
            elif current_split == "b":
                i = 1
            elif current_split == "c":
                i = 2
            elif current_split == "d":
                i = 3
            elif current_split == "e":
                i = 4
            tmp_dir_multi_precision[i][str(current_predictive_period)] = precision
            tmp_dir_multi_npv[i][str(current_predictive_period)] = NPV
            tmp_dir_multi_recall2[i][str(current_predictive_period)] = recall2
            tmp_dir_multi_specificity2[i][str(current_predictive_period)] = specificity2
            tmp_dir_multi_unclassified[i][str(current_predictive_period)] = unclassified

            """if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_split_and_predictive_period = file"""

            """if precision > max_precision:
                max_precision = precision
                best_split_and_predictive_period = file"""

            """if F1 > max_f1:
                max_f1 = F1
                best_split_and_predictive_period = file"""

            if (precision + NPV) > max_precision and (recall2 + specificity2) > max_npv:
                max_precision = precision + NPV
                max_npv = recall2 + specificity2
                best_split_and_predictive_period = file

        # I have to rearange array positions (not alphabetical order)
        for i in range(0, 5):
            sorted_tmp_dir_multi_precision = sorted([int(x) for x in tmp_dir_multi_precision[i].keys()])
            for key in sorted_tmp_dir_multi_precision:
                dir_multi_precision[i].append(tmp_dir_multi_precision[i][str(key)])

            sorted_tmp_dir_multi_npv = sorted([int(x) for x in tmp_dir_multi_npv[i].keys()])
            for key in sorted_tmp_dir_multi_npv:
                dir_multi_npv[i].append(tmp_dir_multi_npv[i][str(key)])

            sorted_dir_multi_recall2 = sorted([int(x) for x in tmp_dir_multi_recall2[i].keys()])
            for key in sorted_dir_multi_recall2:
                dir_multi_recall2[i].append(tmp_dir_multi_recall2[i][str(key)])

            sorted_tmp_dir_multi_specificity2 = sorted([int(x) for x in tmp_dir_multi_specificity2[i].keys()])
            for key in sorted_tmp_dir_multi_specificity2:
                dir_multi_specificity2[i].append(tmp_dir_multi_specificity2[i][str(key)])

            sorted_tmp_dir_multi_unclassified = sorted([int(x) for x in tmp_dir_multi_unclassified[i].keys()])
            for key in sorted_tmp_dir_multi_unclassified:
                dir_multi_unclassified[i].append(tmp_dir_multi_unclassified[i][str(key)])

        # plot current success definition's split and predictive period scores
        draw_split_and_predictive_period_scores(dir_multi_precision, dir_multi_npv, dir_multi_recall2,
                                                dir_multi_specificity2, success_def, dir_name)

        draw_unclassified_items(dir_multi_unclassified, success_def, dir_name)

        # plot current success definition's split and predictive period scores unclassiied items

        print(best_split_and_predictive_period)
        print_best_split_and_predictive_period(complete_dir_name, best_split_and_predictive_period)
        print()


def compare_hitsavvies_flops_adopted(dir_name, hit_or_flop):
    """
        Function which compare HitSavvies' HT and F among the different
        success definitions
    """
    success_def_hits_covered = []
    success_def_flops_covered = []

    # retrieve each success definiton HitSavvies and hits
    for success_def in range(1, 8):
        complete_dir_name = "Def" + str(success_def) + dir_name
        if hit_or_flop == "hit":
            file_name = complete_dir_name + "hit_savvies_success_def" + str(success_def) + ".json"
        else:
            file_name = complete_dir_name + "flop_adopters_success_def" + str(success_def) + ".json"

        with open(file_name, 'r', encoding='utf-8') as infile:
            info_dict = json.load(infile)

        hits_covered = []
        flops_covered = []
        adopters = list(info_dict.keys())
        for ad in list(info_dict.keys()):
            if info_dict[str(ad)]["hits"] > 0:  # REMOVE
                hits_covered.append(info_dict[str(ad)]["hits"])
            if info_dict[str(ad)]["flops"] > 0:  # REMOVE
                flops_covered.append(info_dict[str(ad)]["flops"])
        success_def_hits_covered.append(hits_covered)
        success_def_flops_covered.append(flops_covered)

        print("success def = " + str(hits_covered))
        print(len(hits_covered))
        print(len(flops_covered))
        print()

    """labels = ["seed-user listeners", "seed-user playcounts", "google trends", "last.fm listeners", "last.fm playcounts",
              "AT", "actions"]
    if hit_or_flop == "hit":
        draw_multi_histogram(success_def_flops_covered, "# Flops", "# Hit-Savvies", labels)
    else:
        draw_multi_histogram(success_def_hits_covered, "# Hits", "# Flop-adopter", labels)"""

def retrieve_hits(success_def, hit_or_flop):
    """
        Function which retrives the hits classified as such by each success definition
    """

    hits = []
    flops = []
    if success_def == 1:
        file_name = "info_data/listeners_made_by_seed_users_ground_truth_file_v_38"
    elif success_def == 2:
        file_name = "info_data/playcounts_made_by_seed_users_ground_truth_file_v_38"
    elif success_def == 3:
        file_name = "info_data/google_searches_ground_truth_file_v_38"
    elif success_def == 4:
        file_name = "info_data/listeners_ground_truth_file_v_38"
    elif success_def == 5:
        file_name = "info_data/playcounts_ground_truth_file_v_38"
    elif success_def == 6:
        file_name = "info_data/my_new_continous_AT_with_bias_ground_truth_file5"
    elif success_def == 7:
        file_name = "info_data/adoptions_ground_truth_file_v_38"

    f = open(file_name, "r", encoding="utf-8")
    for line in f:
        line_array = line.split(",")
        g = line_array[0]
        ht = int(line_array[1].replace("\n", ""))
        if ht == -1:
            if g not in flops:
                flops.append(g)
        else:
            if g not in hits:
                hits.append(g)
    f.close()
    if hit_or_flop == "hit":
        return hits
    else:
        return flops

def compare_hits_with_other_hits():
    """
        Function which compares the Hits defined by a success definition with the ones defined by the others
    """

    comparison_dict = {}
    for i in tqdm.tqdm(range(1, 8)):
        current_hits = retrieve_hits(i, "hit")
        current_flops = retrieve_hits(i, "flop")
        for j in range(i + 1, 8):
            next_hits = retrieve_hits(j, "hit")
            next_flops = retrieve_hits(j, "flop")

            jaccard_hh = compute_jaccard(set(current_hits), set(next_hits))
            jaccard_ff = compute_jaccard(set(current_flops), set(next_flops))
            jaccard_hf = compute_jaccard(set(current_hits), set(next_flops))
            jaccard_fh = compute_jaccard(set(current_flops), set(next_hits))

            key = "success_" + str((i)) + "&" + "success_" + str((j))
            comparison_dict[key] = {}
            comparison_dict[key]["jaccard_hh"] = jaccard_hh
            comparison_dict[key]["jaccard_ff"] = jaccard_ff
            comparison_dict[key]["jaccard_hf"] = jaccard_hf
            comparison_dict[key]["jaccard_fh"] = jaccard_fh

    comparison_file = "hits_and_flops_comparison.json"
    with open(comparison_file, 'w', encoding='utf-8') as out:
        json.dump(comparison_dict, out, indent=4)



compare_success_definitions("/part2/", "hit")
compare_success_definitions("/part2/", "flop")
compare_active_period("/part2/", "hit")
compare_active_period("/part2/", "flop")

compare_success_definitions_hitsavvies("/part2/", "hit")
compare_success_definitions_hitsavvies("/part2/", "flop")

compare_percentage_same_hitters("/part2/", "hit")
compare_percentage_same_hitters("/part2/", "flop")

compare_hits_with_other_hits()

draw_m_threshold("/part2/", "hit")
draw_m_threshold("/part2/", "flop")

# retrive and plot all 6 success definition, splits and predictive period prediction results
get_best_split_and_predictiveperiod_success_def("/part2/")
