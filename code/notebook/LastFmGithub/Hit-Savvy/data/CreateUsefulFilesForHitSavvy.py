import os
import gzip
import sys
import csv
import json
import sqlite3
from pathlib import Path
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


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


def collect_seed_users():
    """
        Function which retrieves the seed users (they are present in the "seed_users_node_map1" file)
        ::return an array which contains the seed users
    """
    seed_users = {}
    f = open("seed_users_node_map1", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        username = line_array[0]
        user_encoding = int(line_array[1].replace("\n", ""))
        seed_users[str(username)] = user_encoding
    f.close()
    return seed_users


def collect_seed_users_info():
    """
        Function which collects the seed users' info present in the 55 Datasets and storing theam all
        together in the "seed_user_info" file
    """
    seed_user_info_file = open("seed_users_info.csv", "a")
    adopter = "adopter"
    gender = "gender"
    age = "age"
    country = "country"
    playcount = "playcount"
    registered_on = "registered_on"
    friends = "friends"
    seed_user_info_file.write(
        f"{adopter}\t{gender}\t{age}\t{country}\t{playcount}\t{registered_on}\t{friends}\n")

    seed_users = collect_seed_users()
    seed_users_usernames = seed_users.keys()
    seed_users_encodings = seed_users.values()

    count = 0

    # iterate over the 55 DataSets
    for fl in range(1, 56):
        # glob = module which finds all the pathnames matching a specified pattern
        pathlist = "/home/bradan/mydata/Lastfm_Bradan/DataSets/DataSet" + str(fl) + "/data/users_info/"
        list_of_files = os.listdir(pathlist)

        for filename in tqdm(list_of_files):
            username_array = str(filename).split("_info.json.gz")
            username = username_array[0]

            file_path = pathlist + filename
            if username in seed_users_usernames:
                count += 1
                print("COUNT = " + str(count) + ", retrieved seed user = " + str(username))
                with gzip.GzipFile(file_path, 'r') as fin:
                    data = json.loads(fin.read().decode('utf-8'))
                try:
                    # retrieve seed user's info
                    gender = data['gender']
                    age = data['age']
                    country = data['country']
                    playcount = data['playcount']
                    registered_on = data['registered_on']
                    friends = len(data['friends'])

                    print(
                        f"{seed_users[str(username)]}\t{gender}\t{age}\t{country}\t{playcount}\t{registered_on}\t{friends}\n")

                    seed_user_info_file.write(
                        f"{seed_users[str(username)]}\t{gender}\t{age}\t{country}\t{playcount}\t{registered_on}\t{friends}\n")
                    seed_user_info_file.flush()

                except KeyError:
                    sys.exit(-1)

    seed_user_info_file.close()


def collect_filtered_target_artists_with_main_tag():
    """
        Function which collect the artists present in the "filtered_target_artists_with_main_tag" file
        (the files contains the artists classified with a main music tag, which have at least 100 listeners
        on Last.fm and which were listened by at least 5 seed users of my dataset)
    """

    f = open("filtered_artists_with_main_tag", "r")
    filtered_target_artists = []

    for line in f:
        line_array = line.split("::")
        artist_encoding = int(line_array[0])
        filtered_target_artists.append(artist_encoding)

        print("Collected filtered target artist = " + str(artist_encoding))

    print("len filtered target artist = " + str(len(filtered_target_artists)))
    return filtered_target_artists


def check_if_encoding_is_an_integer(encoding):
    """
        Function which checks the correctness of the encoding passed by argument
    """
    try:
        int(encoding)
        return 0
    except Exception:
        print("NOT AN INTEGER: " + str(encoding))
        return -1


def create_adoption_log_file():
    """
    Function which create the "adopion_log.csv" file, which contains the artists/items for whose I assignes a main
    tag and the relative user/adopters and adoptions. The file has the followinf format:
        artist_encoding, user_encoding, week_encosing, playcount
    """

    filtered_target_artists = collect_filtered_target_artists_with_main_tag()

    current_user = ""

    adoption_log_file = open("adoption_log.csv", "a")
    with gzip.open("week_user_artist_count.gz", 'rt', encoding='utf-8') as week_count:
        for line in tqdm(week_count, ncols=100, ascii=True,
                         desc="username = " + str(current_user)):

            # get data from the line
            data = line.split("::")

            # remove all white spaces and newlines
            week_encoding = check_if_encoding_is_an_integer(data[0])
            user_encoding = check_if_encoding_is_an_integer(data[1])
            artist_encoding = check_if_encoding_is_an_integer(data[2])
            playcount = check_if_encoding_is_an_integer(data[3].replace("\n", ""))

            if week_encoding == -1 or user_encoding == -1 or artist_encoding == -1 or playcount == -1:
                sys.exit(999)

            week_encoding = data[0]
            user_encoding = data[1]
            artist_encoding = data[2]
            playcount = data[3]

            if int(artist_encoding) in filtered_target_artists:
                str_to_write = str(artist_encoding) + "," + str(user_encoding) + "," + str(week_encoding) + "," + str(playcount)
                adoption_log_file.write(str_to_write)
                adoption_log_file.flush()

    adoption_log_file.close()


def count_all_adoptions_made():
    """
        Function which iterates over the "rank_artists_with_total_adoptions" file (which contains
        for every good the numbber of total adoptions, with the goods ordered in descendent number of adoptions)
        and counts all the adoptions made by the seed users
    """

    adoption_log_file = open("adoption_log.csv", "r", encoding="utf-8")
    total_adoptions = len(list(adoption_log_file))
    print("Total number of adoptions made by all seed users = " + str(total_adoptions))
    adoption_log_file.close()
    return adoption_log_file


def compute_number_of_uivocal_adopters():
    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct adopter from adoptions""")
    adopters = cur.fetchall()

    print(adopters)
    print()
    print("Univocal adopters = " + str(len(adopters)))

    conn.close()


def get_artists_with_top_listeners_and_top_playcounts():
    """
        Function which iterates over the "filtered_target_artists_info" directory, in order to rank
        in descendent number of listeners and descendent number of playcounts, the artists present here.
        The result of these rankings are stored in the "rank_artists_with_top_listeners"
        and "rank_artists_with_top_playcounts" files respectively. Files' format:
            artist_encoding::num_listeners / playcounts
    """

    rank_artists_with_top_listeners_file = open("rank_artists_with_top_listeners", "a")
    rank_artists_with_top_playcounts_file = open("rank_artists_with_top_playcounts", "a")

    tmp_dict_listeners = {}
    tmp_dict_playcounts = {}

    filtered_target_artists_info_directory = "filtered_target_artists_info/"
    list_of_files = os.listdir(filtered_target_artists_info_directory)
    for file in list_of_files:
        file_path = filtered_target_artists_info_directory + file

        file_array = file.split("_artist_info.json")
        artist_encoding = int(file_array[0])

        with open(file_path, 'rt', encoding='utf-8') as infile:
            artist_info = json.load(infile)
            infile.close()

        artist_listeners = int(artist_info["listeners"])
        artist_playcounts = int(artist_info["playcount"])

        tmp_dict_listeners[str(artist_encoding)] = artist_listeners
        tmp_dict_playcounts[str(artist_encoding)] = artist_playcounts

    # order dict by descendet values
    sorted_tmp_dict_listeners = {k: v for k, v in sorted(tmp_dict_listeners.items(), key=lambda x: x[1], reverse=True)}
    print(sorted_tmp_dict_listeners)

    # order dict by descendet values
    sorted_tmp_dict_playcounts = {k: v for k, v in sorted(tmp_dict_playcounts.items(), key=lambda x: x[1], reverse=True)}
    sorted_tmp_dict_playcounts = {k: v for k, v in sorted(tmp_dict_playcounts.items(), key=lambda x: x[1], reverse=True)}
    print(sorted_tmp_dict_playcounts)

    for key, value in sorted_tmp_dict_listeners.items():
        str_to_write = str(key) + "::" + str(value) + "\n"
        rank_artists_with_top_listeners_file.write(str_to_write)

    for key, value in sorted_tmp_dict_playcounts.items():
        str_to_write = str(key) + "::" + str(value) + "\n"
        rank_artists_with_top_playcounts_file.write(str_to_write)

    rank_artists_with_top_listeners_file.close()
    rank_artists_with_top_playcounts_file.close()


def get_artists_with_top_playcounts_made_by_seed_users(num_training_set,
                                                 training_set_start_edge, training_set_end_edge, training_set_flag):
    """
        Function which counts the total number of playcounts made by the restricted
        users of my dataset, in order to rank the artists to the greatest to lowest playcount number.
        File's' format:
            artist_encoding::num_listeners / playcounts
        :param num_training_set : number of training set that I'm considering
        :param training_set_start_edge: training set start week
        :param training_set_end_edge: training set end week
    """

    training_set_directory = "training_set" + str(num_training_set) + "/"

    if training_set_flag is True:
        f1 = training_set_directory + "rank_artists_with_top_playcounts_TRS" + str(num_training_set)
    else:
        f1 = training_set_directory + "rank_artists_with_top_playcounts_ts" + str(num_training_set)
    rank_artists_with_top_playcounts_file = open(f1, "a")

    tmp_dict_playcounts = {}
    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct good from adoptions""")
    goods = cur.fetchall()

    for good in goods:
        good = good[0]

        cur = conn.cursor()
        cur.execute("""SELECT sum(quantity) from adoptions 
                            where (good='%s' AND slot >= '%d' AND slot <='%d')
                            """ % (good, training_set_start_edge, training_set_end_edge))
        res = cur.fetchall()  # [(total_playcounts_seed_users )]

        try:
            seed_playcounts = int(res[0][0])
        except TypeError:
            seed_playcounts = 0
        tmp_dict_playcounts[str(good)] = seed_playcounts
        print("good " + str(good) + ", seed_playcounts = " + str(seed_playcounts))

    # order dict by descendet values
    sorted_tmp_dict_playcounts = {k: v for k, v in
                                       sorted(tmp_dict_playcounts.items(), key=lambda x: x[1], reverse=True)}
    print(sorted_tmp_dict_playcounts)

    for key, value in sorted_tmp_dict_playcounts.items():
        str_to_write = str(key) + "::" + str(value) + "\n"
        rank_artists_with_top_playcounts_file.write(str_to_write)
    rank_artists_with_top_playcounts_file.close()


def get_artists_with_top_listeners_made_by_seed_users(num_training_set,
                                                    training_set_start_edge, training_set_end_edge, training_set_flag):
    """
        Function which counts the total number of listeners  of my restricted
        users of my dataset, in order to rank the artists to the greatest to lowest listeners number.
        File's' format:
            artist_encoding::num_listeners
        :param num_training_set : number of training set that I'm considering
        :param training_set_start_edge: training set start week
        :param training_set_end_edge: training set end week
    """

    training_set_directory = "training_set" + str(num_training_set) + "/"

    if training_set_flag is True:
        f1 = training_set_directory + "rank_artists_with_top_listeners_TRS" + str(num_training_set)
    else:
        f1 = training_set_directory + "rank_artists_with_top_listeners_ts" + str(num_training_set)
    rank_artists_with_top_listeners_file = open(f1, "a")

    tmp_dict_listeners = {}
    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct good from adoptions""")
    goods = cur.fetchall()

    for good in goods:
        good = good[0]

        cur = conn.cursor()
        cur.execute("""SELECT count(distinct adopter) from adoptions 
                            where (good='%s' AND slot >= '%d' AND slot <='%d')
                            """ % (good, training_set_start_edge, training_set_end_edge))
        res = cur.fetchall()  # [(total_listenenings_seed_users )]

        try:
            seed_listeners = int(res[0][0])
        except TypeError:
            seed_listeners = 0
        tmp_dict_listeners[str(good)] = seed_listeners
        print("good " + str(good) + ", seed_listeners = " + str(seed_listeners))

    # order dict by descendet values
    sorted_tmp_dict_listeners = {k: v for k, v in
                                       sorted(tmp_dict_listeners.items(), key=lambda x: x[1], reverse=True)}
    print(sorted_tmp_dict_listeners)

    for key, value in sorted_tmp_dict_listeners.items():
        str_to_write = str(key) + "::" + str(value) + "\n"
        rank_artists_with_top_listeners_file.write(str_to_write)
    rank_artists_with_top_listeners_file.close()


def get_artists_with_top_google_searches(num_training_set, training_set_start_edge, training_set_end_edge, training_set_flag):

    google_trens_directory = "pytrends-master/google_trends/"
    trends_directory = google_trens_directory + "trends/"

    start = ""
    if training_set_start_edge == 26:  # training set
        start = "2018-01-14"
    elif training_set_start_edge == 38:  # test set
        start = "2018-04-08"
    elif training_set_start_edge == 50:
        start = "2018-07-01"
    elif training_set_start_edge == 62:
        start = "2018-09-23"
    elif training_set_start_edge == 74:
        start = "2018-12-16"
    elif training_set_start_edge == 86:
        start = "2019-03-10"

    if training_set_end_edge == 37:
        end = "2018-04-01"
    elif training_set_end_edge == 49:
        end = "2018-06-24"
    elif training_set_end_edge == 61:
        end = "2018-09-16"
    elif training_set_end_edge == 73:
        end = "2018-12-09"
    elif training_set_end_edge == 85:
        end = "2019-03-03"
    elif training_set_end_edge == 103:
        end = "2019-07-14"

    print(start)
    print(end)

    tmp_goods_search_counter = {}

    list_of_files = os.listdir(trends_directory)
    for file in list_of_files:

        file_array = file.split("_trends.csv")
        artist_encoding = int(file_array[0])

        start_found = False
        end_found = False

        filename = trends_directory + file
        f = open(filename, "r", encoding="utf-8")
        # f.next()  # skip header
        for line in f:
            if line == "date\tgoogle_trend_search\n" or line == "date\n":
                tmp_goods_search_counter[str(artist_encoding)] = 0
                continue
            line_array = line.split("\t")
            date = line_array[0].strip()
            search_count = line_array[1].replace("\n", "").strip()

            if date == start:
                start_found = True

            if end_found is False and start_found is True:
                try:
                    tmp_c = tmp_goods_search_counter[str(artist_encoding)]
                    tmp_c += int(search_count)
                    tmp_goods_search_counter[str(artist_encoding)] = tmp_c
                except KeyError:
                    tmp_goods_search_counter[str(artist_encoding)] = int(search_count)

                if date == end:
                    end_found = True
        f.close()

        # order dict by descendet values
    sorted_tmp_goods_search_counter = {k: v for k, v in
                                  sorted(tmp_goods_search_counter.items(), key=lambda x: x[1], reverse=True)}
    print(sorted_tmp_goods_search_counter)

    training_set_directory = "training_set" + str(num_training_set) + "/"
    if training_set_flag is True:
        f1 = training_set_directory + "rank_artists_with_top_google_searches_TRS" + str(num_training_set)
    else:
        f1 = training_set_directory + "rank_artists_with_top_google_searches_ts" + str(num_training_set)
    rank_artists_with_top_google_searches_file = open(f1, "a")
    for key, value in sorted_tmp_goods_search_counter.items():
        str_to_write = str(key) + "::" + str(value) + "\n"
        rank_artists_with_top_google_searches_file.write(str_to_write)
    rank_artists_with_top_google_searches_file.close()


def get_artists_with_top_adoptions():
    file_name = "info_data/rank_artists_with_top_adoptions"

    conn = sqlite3.connect("lastfm.db")
    curr = conn.cursor()
    curr.execute("""SELECT distinct good from adoptions""")
    res = curr.fetchall()

    goods = []
    for g in res:
        g = g[0]
        if g not in goods:
            goods.append(g)
    goods_to_adoptions = {}
    for g in goods:
        goods_to_adoptions[str(g)] = 0

    curr = conn.cursor()
    curr.execute("""SELECT * from adoptions""")
    res = curr.fetchall()

    for elem in res:
        g = elem[0]
        goods_to_adoptions[str(g)] += 1
    curr.close()
    conn.close()

    # sort dict by descending values
    sorted_goods_to_adoptions = {k: v for k, v in sorted(goods_to_adoptions.items(), key=lambda x: x[1], reverse=True)}

    f = open(file_name, "a", encoding="utf-8")
    for g, adoption_counter in sorted_goods_to_adoptions.items():
        str_to_write = str(g) + "::" + str(adoption_counter) + "\n"
        f.write(str_to_write)
        f.flush()
    f.close()


collect_seed_users_info()
create_adoption_log_file()
compute_number_of_uivocal_adopters()
count_all_adoptions_made()

get_artists_with_top_listeners_and_top_playcounts()
get_artists_with_top_google_searches(0, 26, 103, True)
get_artists_with_top_adoptions()
get_artists_with_top_listeners_made_by_seed_users(0, 26, 103, True)
get_artists_with_top_playcounts_made_by_seed_users(0, 26, 103, True)
