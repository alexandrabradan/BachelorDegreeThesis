import os
import sys
import json
import gzip
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import csv

import networkx as nx
import pandas as pd
import json
import sys

from cdlib import algorithms
from cdlib import evaluation

import warnings
warnings.filterwarnings('ignore')


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


def create_filtered_edge_list():
    filtered_edge_list_file = open("filtered_edge_list.csv", "a", encoding="utf-8")
    source = "Source"
    target = "Target"
    filtered_edge_list_file.write(
        f"{source}\t{target}\n")

    seed_users = collect_seed_users()
    seed_users_usernames = list(seed_users.keys())
    seed_users_encodings = list(seed_users.values())

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
                    # retrieve seed user's friends and check if he is friend with the other's filtered users
                    try:
                        index = seed_users_usernames.index(str(username))
                    except ValueError:
                        print("username = " + str(username))
                        sys.exit(-1)
                    user_encoding = seed_users_encodings[index]

                    friends = data['friends']
                    for friend, friend_info in friends.items():
                        try:
                            index = seed_users_usernames.index(str(friend))
                        except ValueError:
                            continue  # the friend is not a filtered user
                        friend_encoding = seed_users_encodings[index]

                        filtered_edge_list_file.write(f"{user_encoding}\t{friend_encoding}\n")
                        filtered_edge_list_file.flush()

                except KeyError:
                    sys.exit(-1)

    filtered_edge_list_file.close()


def create_country_map_file():
    data = pd.read_csv("seed_users_info.csv", delimiter="\t", usecols=["country"])

    countries = []
    i = 0

    f = open("country_node_map1", "a", encoding="utf-8")
    for row in data.itertuples():
        c = row.country
        if c not in countries:
            countries.append(c)
            str_to_write = str(c) + "\t" + str(i) + "\n"
            i += 1
            f.write(str_to_write)
            f.flush()
    f.close()

def get_country_encoding(country):
    f = open("country_node_map1", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        c = line_array[0]
        c_encoding = int(line_array[1].replace("\n", ""))
        if country == c:
            return c_encoding
    f.close()
    return -1


def get_user_country(user_encoding):
    f = open("seed_users_info.csv", "r", encoding="utf-8")
    count = 0
    for line in f:
        if count == 0:
            count = 1
            continue  # skip header
        line_array = line.split("\t")
        u_encoding = int(line_array[0])
        country = line_array[3]
        if int(user_encoding) == u_encoding:
            f.close()
            return get_country_encoding(country)
    f.close()
    return -1


def get_country_color(country):
    color_traduction = {"0": "green", "1": "red", "2": "black", "3": "yellow", "4": "blue", "5": "gray"}
    f = open("country_color_map1", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        c = line_array[0]
        c_color = int(line_array[1].replace("\n", ""))
        if country == c:
            f.close()
            return color_traduction[str(c_color)]
    f.close()
    return -1


def get_artists_main_tag(artist_encoding):

    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic",
                 "hip-hop/rap", "jazz", "latin", "pop", "r&b/soul", "reggae",
                 "rock"]

    f = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        art_encoding = int(line_array[0])
        music_genre = line_array[1].replace("\n", "")

        music_genre_encoding = -1
        for i in range(0, len(main_tags)):
            if music_genre == main_tags[i]:
                music_genre_encoding = i

        if int(artist_encoding) == art_encoding:
            f.close()
            return music_genre_encoding
    f.close()
    return -1


def get_user_main_genre(user_encoding):

    tmp_adopter_main_music_genre = {}

    conn = sqlite3.connect("lastfm.db")
    cur = conn.cursor()
    cur.execute("""SELECT distinct good from adoptions where adopter='%s'""" % (user_encoding))
    goods = cur.fetchall()

    for good in goods:
        good = int(good[0])
        # get artist's main music genre
        genre = get_artists_main_tag(good)
        if genre == -1:
            print("good = " + str(good))
            sys.exit(-1)

        try:
            counter = tmp_adopter_main_music_genre[str(genre)]
            counter += 1
            tmp_adopter_main_music_genre[str(genre)] = counter
        except KeyError:
            tmp_adopter_main_music_genre[str(genre)] = 1

    # order dict by descendet values
    sorted_tmp_adopter_main_music_genre = {k: v for k, v in
                                        sorted(tmp_adopter_main_music_genre.items(), key=lambda x: x[1], reverse=True)}

    genres = list(sorted_tmp_adopter_main_music_genre.keys())
    return int(genres[0])


def get_user_continent(country):
    color_traduction = {"0": "green", "1": "red", "2": "black", "3": "yellow", "4": "blue", "5": "gray"}
    f = open("country_color_map1", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        c = line_array[0]
        c_color = int(line_array[1].replace("\n", ""))
        if country == c:
            return c_color
    f.close()
    return -1


def get_continents_main_genre():
    data = pd.read_csv("filtered_node_list.csv", delimiter="\t", usecols=["Continent", "Main_genre"])

    tmp_continents_main_genre_dict = {}

    for row in data.itertuples():

        if row.Continent == "Continent" and row.Main_genre == "Main_genre":
            continue

        c = int(row.Continent)
        mg = int(row.Main_genre)

        try:
            tmp_counter = tmp_continents_main_genre_dict[str(c)][str(mg)]
            tmp_counter += 1
            tmp_continents_main_genre_dict[str(c)][str(mg)] = tmp_counter
        except KeyError:
            empty_counter_dict = {}
            for i in range(0, 13):
                empty_counter_dict[str(i)] = 0
            tmp_continents_main_genre_dict[str(c)] = empty_counter_dict
            tmp_counter = tmp_continents_main_genre_dict[str(c)][str(mg)]
            tmp_counter += 1
            tmp_continents_main_genre_dict[str(c)][str(mg)] = tmp_counter

    for key, value in tmp_continents_main_genre_dict.items():
        # order dict by descendet values
        sorted_value = {k: v for k, v in sorted(value.items(), key=lambda x: x[1],reverse=True)}
        print("continet " + str(key) + " = " + str(sorted_value))

        count = 0
        for elem in sorted_value.values():
            count += elem
        print(count)


collect_seed_users_info()
create_filtered_edge_list()
create_country_map_file()
get_continents_main_genre()


