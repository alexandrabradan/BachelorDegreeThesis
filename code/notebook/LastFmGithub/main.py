from LastfmCollector import LastfmCollector
from LastfmDataCrawler import LastfmDataCrawler
import time
from datetime import date

__author__ = 'Giulio Rossetti', 'Alexandra Bradan'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com", "alexandrabradan@gmail.com"


def process_user(users, lfc, charts_type=("artist", "album", "track"), start_date=None, end_date=None):

    user_names_to_delete = []

    for username in users:
        if "track" in charts_type:
            # get user's charts
            result = lfc.get_user_available_charts(username, charts_type, start_date, end_date)

            if result == -1:  # user doesn't exit and/or file couldn't be created
                user_names_to_delete.append(username)  # append username to the list

    # delete all username's lacking of info/file in the users array
    for u in user_names_to_delete:
        index = users.index(u)  # get user's index to delete
        del users[index]  # delete element at given index

    length = len(user_names_to_delete)
    print("USELESS USERS = " + str(length))

    # iterate on the updated users list
    # for username in users:
        # get user's friends
        # lfc.get_network(username)


def get_additional_infos(users, lfc):
    for username in users:
        # get artists info
        lfc.collect_artist_info_from_weekly_charts(username)
        # get albums info
        lfc.collect_album_info_from_weekly_charts(username)



if __name__ == "__main__":

    lastfm_api = '3878e8d3f604944d12eaef5f34ada2d1'
    crawling_date_start = date(2017, 7, 15)
    crawling_date_end = date(2019, 7, 16)
    start_date_in_unix = time.mktime(crawling_date_start.timetuple())
    end_date_in_unix = time.mktime(crawling_date_end.timetuple())

    print("STARTING DATE = " + str(start_date_in_unix))
    print("ENDING DATE = " + str(end_date_in_unix))

    # class which collect user's info (personal information, tracks, artists and albums listened during the case study,
    # friends) as well as the info of the listened artist
    dg = LastfmCollector(lastfm_api)

    # --------------------------------------CLEAN DIRECTORIES------------------------------------------------------- #

    # delete all content of the files present in the "data/artists_info.csv" and in the "data/users_info.csv"directories
    # dg.clean_directories()

    # if needed, delete all the files in the "data/artists_info.csv" and in the "data/users_info.csv" directories
    # dg.delete_files_directories()

    # ----------------------------------------COLLECT USERNAMES----------------------------------------------------- #

    # class which constructs the user's database, starting from the artist's charts present in the
    # "artists_charts_links.csv" file and retrieving their top listeners
    # dataRetrieve = LastfmDataCrawler()

    # ----------------------------------------COLLECT USERS' DATA--------------------------------------------------- #

    f = open("data/users.csv")
    user_names = []
    for username in f:
        username = username.rstrip()  # remove all the ending white spaces of the username
        user_names.append(username)

    # collect users' info, the tracks, artists, albums listened by each of them and get their friend list
    process_user(user_names, dg, ("artist", "album", "track"), start_date_in_unix, end_date_in_unix)

    # get info about the artists and albums listened by the users
    get_additional_infos(user_names, dg)
