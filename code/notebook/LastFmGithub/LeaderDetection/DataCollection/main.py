#!/usr/bin/env python3
import ast
import os
import json
import subprocess
import time
import urllib
from datetime import date
import gzip
import networkx as nx
import requests

from LastfmDataCrawler import LastfmDataCrawler
from LastfmDataEncoder import LatfmDataEncoding
from LastfmCollector import LastfmCollector
from Costants import Costants
from BuildLeaderDetectionFiles import BuildLeaderDetectionFiles
from Diffusion import Diffusion
from GlobalFileLock import GlobalFileLock

state_variable = ""  # state variable to memorize deleted user after restart of log


def check_if_file_is_corrupted(last_user_crawled):
    last_username_file_path = "data/users_info/" + last_user_crawled + "_info.json.gz"
    last_user_listenings_file_path = "data/users_listenings/" + last_user_crawled + "_listenings.json.gz"
    try:
        with gzip.open(last_username_file_path, 'rt', encoding='utf-8') as infile:
            user_info = json.load(infile)
            check = user_info['friends']  # check if the user has info about his/her friends
            infile.close()

        with gzip.open(last_user_listenings_file_path, 'rt', encoding='utf-8') as infile2:
            user_listenings = json.load(infile2)
            check = user_listenings['crawled']  # check if the user has crawling weeks
            infile2.close()

        return 0  # the file is not corrupted

    except FileNotFoundError:
        # if raised, it means that one of the files don't exist
        return 1  # file corrupted
    except KeyError:
        # if raised, it means that some fields in the files are missing
        return 1  # file corrupted


def get_index_to_start_crawling(user_names, users_log_file_path):
    index_last_user_crawled = dg.get_last_line_from_file(users_log_file_path)

    if index_last_user_crawled == "":  # log file is empty (not yet crawled users)
        print("log file was empty")

        # delete first user's files, if created
        first_user_info_file_path = "data/users_info/" + user_names[0] + "_info.json.gz"
        first_user_listenings_file_path = "data/users_listenings/" + user_names[0] + "_listenings.json.gz"
        dg.delete_file(first_user_info_file_path)
        dg.delete_file(first_user_listenings_file_path)

        state_variable = user_names[0]
        print("DELETE " + first_user_info_file_path)
        return 0  # start crawling from beginning
    else:  # there is at least one user crawled
        # check if last username in the log file is corrupted:
        # 1. is it is corrupted, delete the user's info_file and listenings_file and redo crawling
        # 2. if not, it means that the next username in the "data/users.csv file" => delete this file and redo crawling
        index_last_user_crawled = int(index_last_user_crawled)
        last_user_crawled = user_names[index_last_user_crawled]
        last_user_info_file_path = "data/users_info/" + last_user_crawled + "_info.json.gz"
        last_user_listenings_file_path = "data/users_listenings/" + last_user_crawled + "_listenings.json.gz"

        print("last_user_crawled = " + last_user_crawled + " index_last_user_crawled = " + str(index_last_user_crawled))

        check = check_if_file_is_corrupted(last_user_crawled)
        if check == 1:  # last username in the log file is corrupted
            # delete "index_last_user_crawled" from log file x2 beacuse first I delete newline, than last_user_crawled
            # dg.delete_line_from_file(index_last_user_crawled, users_log_file_path)
            # dg.delete_line_from_file(index_last_user_crawled, users_log_file_path)

            # delete user's files
            dg.delete_file(last_user_info_file_path)
            dg.delete_file(last_user_listenings_file_path)
            return index_last_user_crawled  # redo last username in the log file crawling
        else:  # last username is not corrupted => the next username in the "data/users.csv" file is corrupted
            next_index = index_last_user_crawled + 1

            # check if there is another user after the "last_user_crawled" in the array
            if next_index <= (len(user_names)-1):
                next_user = user_names[next_index]
                # delete next username's info files (they're corrupted, otherwise the username would be in the log file)
                next_user_info_file_path = "data/users_info/" + next_user + "_info.json.gz"
                next_user_listenings_file_path = "data/users_listenings/" + next_user + "_listenings.json.gz"
                dg.delete_file(next_user_info_file_path)
                dg.delete_file(next_user_listenings_file_path)
                state_variable = user_names[next_index]
                print("DELETE " + next_user)
            return next_index  # redo next username crawling


def process_user(start_index, users, lfc, charts_type=("artist", "album", "track"), start_date=None, end_date=None):

    for i in range(len(users)):
        if i >= start_index:
            user = users[i]
            # get user's charts
            result = lfc.get_user_available_charts(user, charts_type, start_date, end_date)

            if result == -1:  # user doesn't exit and/or files couldn't be created
                continue
            else:  # user's info and listenings files exist
                user_info_file_path = "data/users_info/" + user + "_info.json.gz"
                user_listenings_file_path = "data/users_listenings/" + user + "_listenings.json.gz"

                # get user's friends
                res = lfc.get_network(user)
                if res == -1:   # impossible to add friends
                    lfc.delete_file(user_info_file_path)  # delete user's infofile
                    lfc.delete_file(user_listenings_file_path)  # delete users's listenings file
                    continue
                else:  # friends added correctly
                    # write user's index in the main's users[] on the "users_log.csv" file
                    # (in case of crash to restart from here)
                    users_log_file_path = "data/users_log.csv"
                    users_log_file = open(users_log_file_path, 'a', encoding='utf-8')
                    users_log_file.write(str(i)+"\n")  # in the log file I write the index of the user just crawled
                    users_log_file.close()

                    if user == state_variable:
                        user_info_file_path = next_username_file_path = "data/users_info/" + user + "_info.json.gz"
                        print("RECREATE " + user_info_file_path)


if __name__ == "__main__":

    lastfm_api = '3878e8d3f604944d12eaef5f34ada2d1'
    crawling_date_start = date(2017, 7, 15)
    crawling_date_end = date(2019, 7, 16)
    start_date_in_unix = time.mktime(crawling_date_start.timetuple())
    end_date_in_unix = time.mktime(crawling_date_end.timetuple())

    print("STARTING DATE = " + str(start_date_in_unix))
    print("ENDING DATE = " + str(end_date_in_unix))

    user_names = ["MarcusKRZ"]
    users_file_path = "data/users.csv"
    users_log_file_path = "data/users_log.csv"

    dg = LastfmCollector(lastfm_api)

    # ------------------------------------------CLEAN UP------------------------------------------------------------ #

    # if needed, delete all the files present in the  "data/users_info/" directory
    # dg.delete_directories_files("data/users_info/")

    # if needed, delete all the files present in the  "data/users_listenings/" directory
    # dg.delete_directories_files("data/users_listenings/")

    # if needed, emty "data/users_log.csv"
    # dg. delete_content_of_a_file("data/users_log.csv")

    # ---------------------------------------CREATE FILES/DIRECTORIES NEEDED---------------------------------------- #

    # create "data/" directory, if not exists
    dg.check_if_directory_exist_and_create_it("data/")

    # create "data/users_info/" directory, if not exists
    dg.check_if_directory_exist_and_create_it("data/users_info/")

    # create "data/users_listenings/" directory, if not exists
    dg.check_if_directory_exist("data/users_listenings/")

    # create "data/users_log.csv" file, if not exists
    dg.check_if_file_exist_and_create_it("data/users_log.csv")

    # ----------------------------------------COLLECT USERNAMES----------------------------------------------------- #

    # class which constructs the user's database, starting from the artist's charts present in the
    # "artists_charts_links.csv" file and retrieving their top listeners
    dataRetrieve = LastfmDataCrawler()

    # ----------------------------------------COLLECT USERS' DATA--------------------------------------------------- #
    f = open(users_file_path, 'r')
    for username in f:
        username = username.rstrip()  # remove all the ending white spaces of the username
        user_names.append(username)

    start_index = get_index_to_start_crawling(user_names, users_log_file_path)
    print("start_crawling_index = " + str(start_index))

    # collect users' info, the tracks, artists, albums listened by each of them and get their friend list
    # process_user(start_index, user_names, dg, ("artist", "album", "track"), start_date_in_unix, end_date_in_unix)
    process_user(start_index, user_names, dg, "artist", start_date_in_unix, end_date_in_unix)












