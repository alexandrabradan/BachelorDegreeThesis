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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
import collections

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


def distribuition_tags_per_artists():
    """
        Function which computes and draws the distribuition of the main tags among
        the artists  of the action set (all the artists, even the one without leaders)
    """

    main_tags = {"alternative": 0, "blues": 0, "classical": 0, "country": 0, "dance": 0, "electronic": 0,
                 "hip-hop/rap": 0, "jazz": 0, "latin": 0, "pop": 0, "r&b/soul": 0, "reggae": 0, "rock": 0}

    artists_with_main_tag_file = open("filtered_artists_with_main_tag", "r", encoding="utf-8")

    for line in artists_with_main_tag_file:
        line_array = line.split("::")
        artist_encoding = line_array[0]
        artist_tag = line_array[1].replace("\n", "")

        try:
            main_tag_counter = int(main_tags[str(artist_tag)])
            main_tag_counter += 1
            main_tags[str(artist_tag)] = main_tag_counter
        except KeyError:
            sys.exit(-1)

    artists_with_main_tag_file.close()

    x = list(main_tags.keys())
    y = list(main_tags.values())

    print(x)
    print(y)

    tmp_sum = 0
    for elem in y:
        tmp_sum += int(elem)
    print(tmp_sum)

    # draw hystogram
    plt.bar(x, y, align='center', color="red")  # A bar chart
    plt.xlabel('Tags')
    plt.ylabel('# Artists')
    plt.xticks(rotation=45)  # rotate x' labels in order to prevent crowdiness

    # add extra y tick
    y_axis_values = list(plt.yticks()[0])
    first_value = y_axis_values[0]
    second_value = y_axis_values[1]
    difference = second_value - first_value
    last_value = y_axis_values[(len(y_axis_values) - 1)]
    extraticks_y = [(last_value + difference)]
    plt.yticks(y_axis_values + extraticks_y)

    plt.show()


def get_main_music_from_encoding(music_tag_encoding):
    """
        Function which return the main music genres which belong to the encoding passed by argument
        :param music_tag_encoding: encoding for which to return the main music genre
    """

    # the main tags are encoded with the index in this array
    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic",
                 "hip-hop/rap", "jazz", "latin", "pop", "r&b/soul", "reggae",
                 "rock"]

    if 0 <= int(music_tag_encoding) < len(main_tags):
        return main_tags[int(music_tag_encoding)]
    else:
        return ""


def distribuition_tags_per_artists_with_leaders():
    """
        Function which computes and draws the distribuition of the main tags among
        the artists which have a leader
    """

    main_tags = {"alternative": 0, "blues": 0, "classical": 0, "country": 0, "dance": 0, "electronic": 0,
                 "hip-hop/rap": 0, "jazz": 0, "latin": 0, "pop": 0, "r&b/soul": 0, "reggae": 0, "rock": 0}

    leaders_file = open("filtered_artists_with_leaders", "r", encoding="utf-8")

    for line in leaders_file:
        line_array = line.split("::")
        tag_encoding = line_array[1].replace("\n", "")

        # get music tag from encoding
        artist_tag = get_main_music_from_encoding(tag_encoding)
        if artist_tag == "":
            sys.exit(-1)

        try:
            main_tag_counter = int(main_tags[str(artist_tag)])
            main_tag_counter += 1
            main_tags[str(artist_tag)] = main_tag_counter
        except KeyError:
            sys.exit(-1)

    leaders_file.close()

    x = list(main_tags.keys())
    y = list(main_tags.values())

    print(x)
    print(y)

    tmp_sum = 0
    for elem in y:
        tmp_sum += int(elem)
    print(tmp_sum)

    # draw hystogram
    plt.bar(x, y, align='center', color="red")  # A bar chart
    plt.xlabel('Tags')
    plt.ylabel('# Artists')
    plt.xticks(rotation=45)  # rotate x' labels in order to prevent crowdiness

    # add extra y tick
    y_axis_values = list(plt.yticks()[0])
    first_value = y_axis_values[0]
    second_value = y_axis_values[1]
    difference = second_value - first_value
    last_value = y_axis_values[(len(y_axis_values) - 1)]
    extraticks_y = [(last_value + difference)]
    plt.yticks(y_axis_values + extraticks_y)

    plt.show()


def get_main_tag_for_artist(artist_encoding):
    """
        Function which searches the main tag belonging to the artist passed by argument
        and present in the "filtered_artists_with_main_tag" file
    :param artist_encoding: artist for which to search main tag
    :return: the main tag belonging to the artist
            an empty string if the tag is not present
    """

    main_tag = ""
    artists_with_main_tag_file = open("filtered_artists_with_main_tag", "r")
    for line in artists_with_main_tag_file:
        line_array = line.split("::")
        art_encoding = line_array[0]
        main_tag = line_array[1].replace("\n", "")
        if int(artist_encoding) == int(art_encoding):
            return main_tag
    artists_with_main_tag_file.close()
    return main_tag


def distribuition_tags_per_leaders():
    """
        Function which computes and draws the distribuition of the main tags among
        the leaders
    """

    main_tags = {"alternative": 0, "blues": 0, "classical": 0, "country": 0, "dance": 0, "electronic": 0,
                 "hip-hop/rap": 0, "jazz": 0, "latin": 0, "pop": 0, "r&b/soul": 0, "reggae": 0, "rock": 0}

    final_leaders_file = open("final_leaders_DIRECTED", "r", encoding="utf-8")
    for line in final_leaders_file:
        line_array = line.split("::")
        artist_encoding = int(line_array[0])
        tag_encoding = line_array[7].replace("\n", "")

        # get music tag from encoding
        artist_tag = get_main_music_from_encoding(tag_encoding)
        if artist_tag == "":
            sys.exit(-1)

        try:
            main_tag_counter = int(main_tags[str(artist_tag)])
            main_tag_counter += 1
            main_tags[str(artist_tag)] = main_tag_counter
        except KeyError:
            sys.exit(-1)
    final_leaders_file.close()

    x = list(main_tags.keys())
    y = list(main_tags.values())

    print(x)
    print(y)

    tmp_sum = 0
    for elem in y:
        tmp_sum += int(elem)
    print(tmp_sum)

    # draw hystogram
    plt.bar(x, y, align='center', color="red")  # A bar chart
    plt.xlabel('Tags')
    plt.ylabel('# Leaders')
    plt.xticks(rotation=45)  # rotate x' labels in order to prevent crowdiness

    # add extra y tick
    y_axis_values = list(plt.yticks()[0])
    first_value = y_axis_values[0]
    second_value = y_axis_values[1]
    difference = second_value - first_value
    last_value = y_axis_values[(len(y_axis_values) -1 )]
    extraticks_y = [(last_value + difference)]
    plt.yticks(y_axis_values + extraticks_y)

    plt.show()


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


def distribuition_artists_per_leaders():
    """
        Function which computes and draws the distribuition of the artists adopted and the
        the leaders that adopted them (obviuosly I consider only the 9,899 new artists)
    """

    tmp_users_adoption_counter = {}

    final_leaders_file = open("final_leaders_DIRECTED", "r", encoding="utf-8")
    for line in final_leaders_file:
        line_array = line.split("::")
        user_encoding = line_array[1]

        try:
            tmp_user_counter = int(tmp_users_adoption_counter[str(user_encoding)])
            tmp_user_counter += 1  # increment users's number of adoption with current artist
            tmp_users_adoption_counter[str(user_encoding)] = tmp_user_counter
        except KeyError:
            # I add the current user to the dict, beause he isn't present
            tmp_users_adoption_counter[str(user_encoding)] = 1
    final_leaders_file.close()

    x = list(tmp_users_adoption_counter.values())

    fig, ax1 = plt.subplots()

    # print bin's numbers and bins's size
    num_bins, binwidth = get_doane_num_bin_and_binwidth(x)
    print("num_bins = " + str(num_bins))
    print("binwidth = " + str(binwidth))

    # binning done with Doane formula
    # y axis is logarithmic
    n, bins_edges, patches = ax1.hist(x, log=True, bins='doane', color='red', width=binwidth/2)
    print(n)
    print(len(n))

    plt.xlabel('# Artists')
    plt.ylabel('# Leaders')
    plt.show()


def correlation_between_three_dimension_of_social_prominance():
    """
        Function which computes and draws the Pearson correlation coefficent among the
        three dimensions of social prominance
    """
    # Import Dataset
    df = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["tribe","width", "depth", "strength"])

    # Plot
    plt.figure()
    ax = sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='rainbow', center=0,
                     annot=True)

    # Decorations
    plt.xticks(rotation=45)  # rotate x' labels in order to prevent crowdiness
    plt.yticks(rotation=45)  # rotate x' labels in order to prevent crowdiness
    # avoid heatmap's cutting off
    b, t = plt.ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    plt.ylim(b, t)  # update the ylim(bottom, top) values

    plt.show()


def elbow_criterion_for_optimal_k_for_kmeans():

    # data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["depth", "width", "strength"])
    data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["tribe", "depth", "width", "strength"])
    print(data)

    sse = {}
    for k in range(1, 21):
        print("sse for k = " + str(k))
        kmeans = KMeans(n_clusters=k, max_iter=100000000).fit(data)
        data["clusters"] = kmeans.labels_
        # print(data["clusters"])
        sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    plt.figure()
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.ylabel("SSE")
    plt.show()


def silhoutte_coefficent_for__optimal_k_for_kmeans():

    # data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["depth", "width", "strength"])
    data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t",
                       usecols=["tribe", "depth", "width", "strength"])
    print(data)

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
    plt.xlabel("Number of cluster")
    plt.ylabel("Silhouette Coefficient")
    plt.show()


def get_tag_from_encoding(tag_encoding):
    # the main tags are encoded with the index in this array
    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic",
                 "hip-hop/rap", "jazz", "latin", "pop", "r&b/soul", "reggae",
                 "rock"]
    try:
        tag = main_tags[int(tag_encoding)]
        return tag
    except Exception:
        return ""


def compute_Revealed_Comparative_Advantage(original_labels, new_labels):
    """
        Function which computes RCA(i, j): (freq_{i, j} / freq_{i, *}) / (freq_{*, i} / freq_{*, *})
        where:
        a) i is a tag
        b) j is a cluster
        c) freq_{i, j} is the # of leaders who spread an artist tagget with i and present in cluster j
    """

    max_tags_in_each_cluster = {}
    max_tags_in_each_cluster[str(0)] = (-1, -1)
    max_tags_in_each_cluster[str(1)] = (-1, -1)
    max_tags_in_each_cluster[str(2)] = (-1, -1)
    max_tags_in_each_cluster[str(3)] = (-1, -1)

    freq_i_j = {}
    freq_i_all = {}
    freq_all_j = {}
    freq_all_all = len(original_labels)

    for index in range(0, len(original_labels)):
        j = new_labels[index]  # leader's cluster
        i = str(original_labels.iloc[index].values).replace("[", "").replace("]", "")  # leader's action music tag

        try:
            j_dict = freq_i_j[str(j)]
        except KeyError:
            freq_i_j[str(j)] = {}

        try:
            j_dict = freq_i_j[str(j)]
            tmp_sum = int(j_dict[str(i)])
            tmp_sum += 1
            j_dict[str(i)] = tmp_sum
            freq_i_j[str(j)] = j_dict
        except KeyError:
            j_dict = freq_i_j[str(j)]
            j_dict[str(i)] = 1
            freq_i_j[str(j)] = j_dict

        try:
            tmp_sum = int(freq_i_all[str(i)])
            tmp_sum += 1
            freq_i_all[str(i)] = tmp_sum
        except KeyError:
            freq_i_all[str(i)] = 1

        try:
            tmp_sum = int(freq_all_j[str(j)])
            tmp_sum += 1
            freq_all_j[str(j)] = tmp_sum
        except KeyError:
            freq_all_j[str(j)] = 1

    f = open("diffusion_trees_revealed_comparative_advantage", "a", encoding="utf-8")
    for cluster, cluster_dict in freq_i_j.items():
        for tag_encoding, tag_counter_in_cluster in cluster_dict.items():
            rca_i_j = (tag_counter_in_cluster / freq_i_all[str(tag_encoding)]) / (
                        freq_all_j[str(cluster)] / freq_all_all)

            tag = get_tag_from_encoding(tag_encoding)
            if tag == "":
                print("Wrong music tag encoding = " + str(tag_encoding))
                sys.exit(-1)

            if float(rca_i_j) > float(max_tags_in_each_cluster[str(cluster)][1]):
                max_tags_in_each_cluster[str(cluster)] = (tag_encoding, rca_i_j)

            str_to_write = str(cluster) + "::" + str(tag_encoding) + "::" + str(rca_i_j) + "\n"
            f.write(str_to_write)
    f.close()

    return max_tags_in_each_cluster


def count_music_genre_in_clusters(tag, new_labels, original_labels):
    print()
    print("tag = " + str(tag))
    print(new_labels)
    print(original_labels)

    count_tag0 = 0
    count_tag1 = 0
    count_tag2 = 0
    count_tag3 = 0
    count_tag_total = 0
    for index in range(0, len(original_labels)):
        j = new_labels[index]  # leader's cluster
        i = str(original_labels.iloc[index].values).replace("[", "").replace("]", "")  # leader's action music tag

        if i == int(tag):
            if j == 0:
                count_tag0 += 1
            elif j == 1:
                count_tag1 += 1
            elif j == 2:
                count_tag2 += 1
            else:
                count_tag3 += 1

            count_tag_total += 1

    print(count_tag0)
    print(count_tag1)
    print(count_tag2)
    print(count_tag3)
    print(count_tag_total)


def cluster_with_kmeans(num_clusters):
    data = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["tribe", "depth", "width", "strength"])
    print(data)

    original_labels = pd.read_csv("final_leaders_DIRECTED.csv", delimiter="\t", usecols=["music_tag"])
    # print("original_labels = " + str(original_labels))

    km = KMeans(n_clusters=num_clusters, n_init=42, max_iter=100000000)
    km.fit(data)

    new_labels = km.labels_
    # print("new_labels = " + str(new_labels))

    counter_dict = collections.Counter(list(new_labels))
    # sort dict by key
    sorted_counter_dict = {k: v for k, v in sorted(counter_dict.items(), key=lambda x: x[0])}
    for key, value in sorted_counter_dict.items():
        print("Cluster " + str(key) + "'s members = " + str(value))

    max_tags_in_each_cluster = compute_Revealed_Comparative_Advantage(original_labels, new_labels)

    # for tag, value in max_tags_in_each_cluster.values():
    #      count_music_genre_in_clusters(int(tag), new_labels, original_labels)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(data.iloc[:, 3], data.iloc[:, 1], data.iloc[:, 4], c=data.iloc[:, 5], cmap='gist_rainbow', edgecolor='k',
               s=150)
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Strength')
    plt.title("Original data", fontsize=18)
    # plt.show()

    print(len(data.iloc[:, 2]))
    print(len(new_labels))

    # fig = plt.figure()
    # ax2 = fig.add_subplot(projection='3d')
    ax.scatter(data.iloc[:, 2], data.iloc[:, 1], data.iloc[:, 3], c=new_labels, cmap='jet', edgecolor='k', s=150)
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')
    ax.set_zlabel('Strength')
    plt.title("K-means", fontsize=18)
    plt.show()

    centers = km.cluster_centers_
    print(centers)

    fig = gcf()
    axes = fig.gca()
    s = axes.scatter(centers[:, 2], centers[:, 1], c=centers[:, 3], cmap='jet', edgecolor='k', s=150)
    axes.set_xlabel('Width')
    axes.set_ylabel('Depth')

    n = ["  centroid_1", "  centroid_2", "  centroid_3", "centroid_4"]
    for i, txt in enumerate(n):
        axes.annotate(txt, (centers[i, 2], centers[i, 1]))

    cb = plt.colorbar(s)
    cb.set_label('Strenght')
    plt.show()



# distribuition_tags_per_artists()
# distribuition_tags_per_artists_with_leaders()
# distribuition_tags_per_leaders()
# distribuition_artists_per_leaders()

# elbow_criterion_for_optimal_k_for_kmeans()
# silhoutte_coefficent_for__optimal_k_for_kmeans()
# cluster_with_kmeans(4)  # 4 => best k for k-means

# correlation_between_three_dimension_of_social_prominance()
