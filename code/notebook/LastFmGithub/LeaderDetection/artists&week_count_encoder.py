import os
import re
import zipfile
import gzip
import json
from pathlib import Path

from tqdm import tqdm

__author__ = 'Giulio Rossetti', 'Alexandra Bradan'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com", "alexandrabradan@gmail.com"


def touch(path):
    """
    Create a new file in the specified path
    :param path: path where to create the new file
    """
    with open(path, 'a', encoding='utf-8') as f:
        os.utime(path, None)
        f.close()


def check_if_file_exist(path):
    """
       Check if a file exist
       :param path: file's path to check existance
       :return: 0 if the file already exist
                -1 if the file didn't exist
    """
    try:
        # check if the user's info file exist
        open(path).close()
        return 0
    except FileNotFoundError:
        # if the FileNotFoundError is raised it means that the file doesn't exist
        return -1

def check_if_user_is_encoded(username):
    """
        Function which checks if a username is econded in our  "node_map1" file (in case we don't find him, it means
        that is a friends with less then 100 users analyzed in our DataSets)
        :param username to search in the  "node_map1" file
        :return >= 0 the encoding, if the username is encoded
                -1 otherwise
    """

    # Compile a regular expression pattern into a regular expression object, which can be used
    # for matching using its match(), search() and other methods
    b = bytes(username, 'utf-8')  # convert username to bytes
    my_regex = re.compile(b)

    with zipfile.ZipFile('networks.zip') as z:
        for filename in z.namelist():
            # check is user is encoded (if not jump to next iterationn beacuse we don't need his listenings)
            if filename == "node_map1":
                # open  "node_map1" file
                with z.open(filename) as f:
                    for line in f:
                        match = my_regex.match(line)

                        if match is not None:
                            # retrive user's encoding
                            username_encoding = line.lstrip(b).decode("utf-8")  # convert bytes to string
                            username_encoding = username_encoding.strip('\t')
                            username_encoding = username_encoding.strip('\n')
                            f.close()
                            z.close()
                            return username_encoding

    # signal to the caller that the username is not encoded in our file
    return -1


def get_week_encoding(week):
    """
    Function which returns the corresponding  encoding of the week passed by argument, from the "weeks_map1" file
    :param week: week express in unixtime
    :return >= 0 the encoding, if the week is encoded
            -1 otherwise
    """

    # Compile a regular expression pattern into a regular expression object, which can be used
    # for matching using its match(), search() and other methods
    b = bytes(week, 'utf-8')  # convert week to bytes
    my_regex = re.compile(b)

    with zipfile.ZipFile('leader_detection.zip') as z:
        for filename in z.namelist():
            # check is week is encoded
            if filename == "weeks_map1":

                with z.open(filename) as f:
                    for line in f:
                        match = my_regex.match(line)

                        if match is not None:
                            # retrieve week's encoding
                            week_encoding = line.lstrip(b).decode("utf-8")  # convert bytes to string
                            week_encoding = week_encoding.strip('\t')
                            week_encoding = week_encoding.strip('\n')
                            f.close()
                            z.close()
                            return week_encoding

    # signal to the caller that the week is not encoded in our file
    return -1


def week_user_artist_count_builder():
    """
        Function which encodes the listenings present in the 55 "users_listenings" directories of the 55 DataSets
        of our study. The result of the the encoding is stored in the "week_user_artist_count.gz" file.
        The function has also the task to encode the listenings' artists and to store them in the "artists_map1" file.
    """

    artists = {}  # artists encoded
    i = 0  # starting encoding

    artists_map1 = "artists_map1"  # file which contains artists encodings

    week_user_artist_count = "week_user_artist_count.gz"  # file which contains user's weekly listenings encoded

    # check if artists's encoding file exist, otherwise create it
    check = check_if_file_exist(artists_map1)

    if check == 0:
        return  # file already exists

    # create artists's encoding file
    touch(artists_map1)

    with open("artists_map1", 'a', encoding='utf-8') as o:

        # check if week_user_artist_count file, otherwise create it
        check = check_if_file_exist(week_user_artist_count)

        if check == 0:
            o.close()  # close artists' encoding file
            return    # file already exists

        # create week_user_artist_count file
        touch(week_user_artist_count)

        with gzip.open(week_user_artist_count, 'at', encoding='utf-8') as outfile:

            # iterate over the "users_listenings" directories of the 55 DataSets, in order to
            # retrieve the listenings and while doing so, encoding the artists encoutered
            for fl in range(0, 56):
                # get all the users' files present in the DataSet{fl}/data/users_listenings/'s directory
                # glob = module which finds all the pathnames matching a specified pattern
                pathlist = Path(f"/data/users/bradan/Lastfm_Bradan/DataSets/DataSet{fl}/data/users_listenings/").glob("*gz")

                #dataset_data_users_listenings = "data/users_listenings/"
                # get the files present in the "users_listenings"
                # list_of_files = os.listdir("data/users_listenings/")
                # for filename in list_of_files:

                for filename in pathlist:

                    # construct user's file path in order to open it
                    # file_path = dataset_data_users_listenings + filename

                    # retrieve user's listenings
                    # with gzip.open(file_path, 'rt', encoding='utf-8') as fin:
                    with gzip.open(filename, 'rt', encoding='utf-8') as fin:
                        user_info = json.load(fin)
                        fin.close()

                    try:
                        # retrieve user's name
                        username = user_info['user_id']
                        # check is user is encoded
                        username_encoding = check_if_user_is_encoded(username)

                        if username_encoding == -1:  # username not encoded
                            continue  # iterate over the next user

                        # username is encoded (is part of our study and we have his/her friends data)
                        # retrieve user's listenings, encode them and while doing so encode the artist listened
                        # (if not already encoded)
                        info_to_print = "  in DataSets" + str(fl)
                        for key, crawled_week in tqdm(user_info["crawled"].items(), ncols=100, ascii=True,
                                                           desc=username + info_to_print):

                            # iterate over the tracks of a given crawled week to extract artists
                            for kkey, artists_chart in crawled_week["listened artists"].items():

                                artist = artists_chart['artist']

                                # check if artist is present or is "<Unknown>" (don't collect data in this case)
                                if artist == "<Unknown>":
                                    continue  # next artist

                                # get number of playcounts
                                playcount = artists_chart['playcount']

                                # get artists's encoding
                                artist_encoding = -1

                                if artist in artists:
                                    artist_encoding = artists[artist]

                                if artist_encoding == -1:  # artist is not encoded
                                    # encode artist
                                    artist_encoding = i
                                    artists[artist] = artist_encoding

                                    i += 1  # increment encoding

                                    # write encoded artist to "artists_map1" file
                                    o.write(f"{artist}\t{ artist_encoding}\n")

                                # get week's encoding
                                week_encoding = get_week_encoding(key)  # key = week

                                if week_encoding == -1:
                                    print("Week |" + key + "| is NOT encoded")
                                    continue  # strage, it shouldn't happen

                                str_to_write = str(week_encoding) + "::" + str(username_encoding) + "::" + str(artist_encoding) + "::" + str(playcount)

                                outfile.write(str_to_write)
                                outfile.write("\n")
                    except:
                        continue

        outfile.close()  # close week_user_artist_count file

    o.close()  # close artist encoding file


def add_file_to_zip_directory(filepath, zip_archive):
    """
        Function which add a file to a zip archive
        :param filepath filepath of the file to add to the zip archive
        :param zip_archive zip archive in which to add the file
    """

    # get reference to the zip archieve
    z = zipfile.ZipFile(zip_archive, 'a')

    # write file to the zip archive
    z.write(os.path.join(filepath))

    # remove original file
    os.remove(filepath)


week_user_artist_count_builder()

# add "week_user_artist_count.gz" file to the "leader_detection.zip" archive
add_file_to_zip_directory("week_user_artist_count.gz", "leader_detection.zip")
