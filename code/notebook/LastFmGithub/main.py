#!/usr/bin/env python3

import os
import json
import gzip
import time
from datetime import date
from LastfmCollector import LastfmCollector
from LastfmDataCrawler import LastfmDataCrawler

state_variable = ""  # state variable to memorize deleted user after restart of log


def check_if_file_is_corrupted(last_user_crawled):
    last_username_file_path = "data/users_info/" + last_user_crawled + "_info.json"
    try:
         with gzip.open(last_username_file_path, 'rt', encoding='utf-8') as infile:
            user_info = json.load(infile)

        check1 = user_info["crawled"]
        check2 = user_info["friends"]

        return 0
    except FileNotFoundError:
        # if raised, it means that the file doesn't exist
        return 1  # file corrupted
    except KeyError:
        # if raised, it means that some fields are missing
        return 1  # file corrupted


def get_index_to_start_crawling(user_names, users_log_file_path):
    index_last_user_crawled = dg.get_last_line_from_file(users_log_file_path)

    if index_last_user_crawled == "":  # log file is empty (not yet crawled users)
        print("log file was empty")

        # delete first user's file, if created
        first_user_file_path = "data/users_info/" + user_names[0] + "_info.json"
        dg.delete_file(first_user_file_path)
        state_variable = user_names[0]
        print("DELETE " + first_user_file_path)
        return 0  # start crawling from beginning
    else:  # there is at least one user crawled
        # check if last username in the log file is corrupted:
        # 1. is it is corrupted, delete the user's info_file and redo crawling
        # 2. if not, it means that the next username in the "data/users.csv file" => delete this file and redo crawling
        index_last_user_crawled = int(index_last_user_crawled)
        last_user_crawled = user_names[index_last_user_crawled]

        print("last_user_crawled = " + last_user_crawled + " index_last_user_crawled = " + str(index_last_user_crawled))

        check = check_if_file_is_corrupted(last_user_crawled)
        if check == 1:  # last username in the log file is corrupted
            # delete "index_last_user_crawled" from log file
            dg.delete_line_from_file(index_last_user_crawled, users_log_file_path)
            return index_last_user_crawled  # redo last username in the log file crawling
        else:  # last username is not corrupted => the next username in the "data/users.csv" file is corrupted
            next_index = index_last_user_crawled + 1

            # check if there is another user after the "last_user_crawled" in the array
            if next_index <= (len(user_names)-1):
                # delete next username's info file (it's corrupted, otherwise the username would be in the log file)
                next_username_file_path = "data/users_info/" + user_names[next_index] + "_info.json"
                dg.delete_file(next_username_file_path)
                state_variable = user_names[next_index]
                print("DELETE " + next_username_file_path)
            return next_index  # redo next username crawling


def process_user(start_index, users, lfc, charts_type=("artist", "album", "track"), start_date=None, end_date=None):

    for i in range(len(users)):
        if i >= start_index:
            user = users[i]
            # get user's charts
            result = lfc.get_user_available_charts(user, charts_type, start_date, end_date)

            if result == -1:  # user doesn't exit and/or file couldn't be created
                continue
            else:  # user's file created and his/her listenings added to it
                # get user's friends
                res = lfc.get_network(user)
                if res == -1:   # impossible to add friends
                    continue
                else:  # friends added correctly
                    # write user's index in the main's users[] on the "users_log.csv" file
                    # (in case of crash to restart from here)
                    users_log_file_path = "data/users_log.csv"
                    users_log_file = open(users_log_file_path, 'a', encoding='utf-8')
                    # check if file is not empty => not adding "\n" to first line(which doesn't need it)
                    if os.stat(users_log_file_path).st_size != 0:
                        users_log_file.write("\n")
                    users_log_file.write(str(i))  # in the log file I write the index of the user just crawled
                    users_log_file.close()

                    if user == state_variable:
                        user_info_file_path = next_username_file_path = "data/users_info/" + user + "_info.json"
                        print("RECREATE " + user_info_file_path)


def get_additional_infos(users, lfc):
    for u in users:
        user_file = "data/users_info/" + u + "_info.json"

        # check if user's file exist (if not it means it wasn't create in the crawling phase)
        if lfc.check_if_file_exist(user_file) == 0:
            # get artists info
            lfc.collect_artist_info_from_weekly_charts(u)
            # get albums info
            # lfc.collect_album_info_from_weekly_charts(u)


if __name__ == "__main__":

    lastfm_api = '3878e8d3f604944d12eaef5f34ada2d1'
    crawling_date_start = date(2017, 7, 15)
    crawling_date_end = date(2019, 7, 16)
    start_date_in_unix = time.mktime(crawling_date_start.timetuple())
    end_date_in_unix = time.mktime(crawling_date_end.timetuple())

    print("STARTING DATE = " + str(start_date_in_unix))
    print("ENDING DATE = " + str(end_date_in_unix))

    user_names = []
    users_file_path = "data/users.csv"
    users_log_file_path = "data/users_log.csv"

    dg = LastfmCollector(lastfm_api)

    # ------------------------------------------CLEAN UP------------------------------------------------------------ #

    # if needed,delete files' content present in the "data/artists_info.csv" and in the "data/users_info.csv"directories
    # dg.clean_directories()

    # if needed, delete all the files in the "data/artists_info.csv" and in the "data/users_info.csv" directories
    # dg.delete_files_directories()

    # if needed, delete all content of the "data/users_log.csv"
    # dg.delete_content_of_a_file(users_log_file_path)

    # ----------------------------------------COLLECT USERNAMES----------------------------------------------------- #

    # class which constructs the user's database, starting from the artist's charts present in the
    # "artists_charts_links.csv" file and retrieving their top listeners
    # dataRetrieve = LastfmDataCrawler()

    # ----------------------------------------COLLECT USERS' DATA--------------------------------------------------- #
    f = open(users_file_path, 'r')
    for username in f:
        username = username.rstrip()  # remove all the ending white spaces of the username
        user_names.append(username)

    start_index = get_index_to_start_crawling(user_names, users_log_file_path)
    print("start_crawling_index = " + str(start_index))

    # collect users' info, the tracks, artists, albums listened by each of them and get their friend list
    process_user(start_index, user_names, dg, ("artist", "album", "track"), start_date_in_unix, end_date_in_unix)

    # get info about the artists and albums listened by the users
    get_additional_infos(user_names, dg)




