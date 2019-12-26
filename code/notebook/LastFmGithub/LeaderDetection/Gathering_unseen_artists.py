import gzip
import os
import sys


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


def gather_unseen_artists():
    """
        Function which gether the artists whose firts listening among all the users in our DataSets is dated before
        14/06/18 12:00 PM (unixtime = 1515931200, week_encoding = 26). Our datasets collect the listenings
        starting from 16/07/17 12:00 PM (unixtime = 1500206400, week_encoding = 0), but we focus only on artists
        whose first listening is recorded SIX MONTHS (5 months and 30 days, alias 183 days in our case)
        after the beginning of our observation period. If an artist was in activity before our observation
        time window, there is no way to know if a user has listened to it before, therefore
        nullifying our leader detection strategy.
    """
    check = check_if_file_exist("target_artists")

    if check == 0:
        return
    else:
        touch("target_artists")

    first_week_directory = "first_week/"

    with open("target_artists", 'a') as outfile:

        list_of_files = os.listdir(first_week_directory)
        for file in list_of_files:
            # read artist's first listenings file
            file_path = first_week_directory + file

            tmp_min_first_week = [sys.maxsize, -1, -1, -1]  # first_week, user, artist, playcount
            discard_artist = False

            with gzip.open(file_path, 'r') as infile:
                for line in infile:
                    line_array = line.decode("utf-8").split("::")

                    week_encoding = line_array[0]
                    user_encoding = line_array[1]
                    artist_encoding = line_array[2]
                    playcount = line_array[3].strip()  # remove "\n"

                    # I have to check if the current listening week is >= 14/06/18 12:00 PM
                    # (and eventually update the absolute first artist listening made by some user).
                    # If I found out that the current listening week is <  14/06/18 12:00 PM
                    # I have to delete the artist from the dict (some user has listened to him before our observation
                    # time window)
                    if int(week_encoding) < 26:
                        discard_artist = True
                        print("Artist <<" + str(artist_encoding) + ">> is a excluded")
                        break  # discard artist and iterate over the next artist's file

                    if int(week_encoding) < int(tmp_min_first_week[0]):   # week_encoding >= 26, update minimum
                            tmp_min_first_week = [week_encoding, user_encoding, artist_encoding, playcount]

                infile.close()

                if discard_artist is False:

                    print("Artist |" + str(tmp_min_first_week[2]) + "| is a TARGET")

                    # write target artist on string (I found his absolute first week of listenings and it is
                    # >= 14/01/18)
                    str_to_write = str(tmp_min_first_week[0]) + "::" + str(tmp_min_first_week[1]) + "::" + \
                                   str(tmp_min_first_week[2]) + "::" + str(tmp_min_first_week[3])
                    outfile.write(str_to_write)
                    outfile.write("\n")

        outfile.close()


gather_unseen_artists()
