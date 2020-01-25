import math
import os
import re
import zipfile
import gzip
import string

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


def create_directory(path):
    """
   Create a new directory in the specified path
   :param path: path where to create the new directory
   """
    os.makedirs(path)


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


def get_week_from_encoding(week_encoding):
    """
        Function which returns the corresponding week of the week_encoding passed by argument
        :return: >=0 unixtime of the week corresponding to the week_encoding
                 -1 otherwise
    """

    # Compile a regular expression pattern into a regular expression object, which can be used
    # for matching using its match(), search() and other methods
    # convert the week_encoding to bytes and search the week_encoding in every line that starts with something first
    # and ends with a '\n'
    # (in our case first we have the unixtime week, followed by a '\t', then the week encoding and '\n')
    b = bytes(".*" + week_encoding + "\n", 'utf-8')
    my_regex = re.compile(b)

    with zipfile.ZipFile('leader_detection.zip') as z:
        for filename in z.namelist():
            # check is week_encoding has a corresponding unixtime week
            if filename == "weeks_map1":

                with z.open(filename) as f:
                    for line in f:
                        match = my_regex.match(line)

                        if match is not None:
                            # retrieve unixtime week from week_encoding
                            # line's format : unixtime_week \t week_encoding
                            line_array = line.decode("utf-8").split('\t')
                            week = line_array[0].strip('\t')
                            week = week.strip('\n')
                            f.close()
                            z.close()
                            return week

    # the week_encoding is not present in the  "weeks_map1" file
    return -1


def user_artist_totalistening_periodlength_builder():
    """
        Function which iterates over the "week_user_artist_count.gz" file in order to count:
        1. how many times in our period of study a given user has listened to a given artist (total listenings)
        2. first and last period of listenings made by a given user to a given artist (period length of listening)

        The function also keeps track on the "first_week_user_artist_count.gz" file of the first week of
        listening of an artist made by a particular user.
    """

    # file which contains users' first weekly listenings encoded
    first_week_user_artist_count = "first_week_user_artist_count.gz"

    # check if first_week_user_artist_count file exists, otherwise create it
    check = check_if_file_exist(first_week_user_artist_count)

    if check == 0:
        return  # file already exists

    # create first_week_user_artist_count file
    touch(first_week_user_artist_count)

    check = check_if_directory_exist("total_listenings")

    if check == 0:
        return  # directory already exists

    # create "total_listenigs" directory
    create_directory("total_listenings")

    current_user = ""  # current user examinated
    tmp_user_esteem = {}  # tmp dict where to store user's statistics of period of listenings and total playcounts

    # file which contains users' weekly listenings encoded
    week_user_artist_count = "week_user_artist_count.gz"
    with gzip.open(first_week_user_artist_count, 'at', encoding='utf-8') as first_week_outfile:

        # read "week_user_artist_count.gz" file and from it, for each user, compute:
        # 1. username + "_first_week_user_artist_count.gz"
        # 2. "user_artist_totalistening_periodlength.gz"
        with gzip.open(week_user_artist_count, 'rt', encoding='utf-8') as infile:
            # iterate over the  "week_user_artist_count.gz" file's lines, in order to get the info needed
            for line in infile:

                print("LINE = " + str(line))

                # get data from the line
                data = line.split("::")

                # remove all white spaces and newlines
                week_encoding = data[0]
                user_encoding = data[1]
                artist_encoding = data[2]
                playcount = data[3].translate({ord(c): None for c in string.whitespace})

                if current_user != user_encoding and current_user != "":
                    # we have a new user to analyse => first we have to write the updated result
                    # of the current user to the "user_artist_totalistening_periodlength.gz" file

                    # iterate over the esteems of the artists listened by the user
                    for key, value in tmp_user_esteem.items():

                        try:
                            # get corresponding weeks from encodings (NEED STRING AS PARAMETER)
                            first_week = get_week_from_encoding(str(value["first_week"]))
                            last_week = get_week_from_encoding(str(value["last_week"]))

                            if first_week == -1 or last_week == -1:
                                continue  # it shouldn't happen, but we skip iteration

                            total_listening_period_unixtime = int(last_week) - int(first_week)
                            # 1 week = 604800 seconds
                            total_listening_period = math.floor(total_listening_period_unixtime / 604800)
                            total_playcounts = value["total_playcount"]

                            # write results on the user_artist_totalistening_periodlength file
                            str_to_write = str(current_user) + "::" + str(key) + "::" + str(total_playcounts) + "::" + str(
                                total_listening_period)

                            # file in which to write the the estimations that we have to make from the "week_user_artist_count.gz" file
                            user_artist_totalistening_periodlength = "total_listenings/" + str(current_user) +  \
                                                                            "user_artist_totalistening_periodlength.gz"
                            # check if user_artist_totalistening_periodlength exists file, otherwise create it
                            check = check_if_file_exist(user_artist_totalistening_periodlength)
                            if check == -1:
                                # create user_artist_totalistening_periodlength file
                                touch(user_artist_totalistening_periodlength)

                            with gzip.open(user_artist_totalistening_periodlength, 'at', encoding='utf-8') as outfile:
                                outfile.write(str_to_write)
                                outfile.write("\n")
                                outfile.close()

                        except Exception as e:  # it should not happen
                            print(e)
                            print()

                    # update current user to analyze
                    current_user = user_encoding

                    # re-initialize the dict with a new empty dict
                    tmp_user_esteem = {}

                    # keep track of new user's listenings
                    tmp_user_esteem[str(artist_encoding)] = {"first_week": week_encoding,
                                                             "last_week": week_encoding,
                                                             "total_playcount": playcount}

                    # keep track of the artist's first week of listening made by the user
                    str_to_write = str(week_encoding) + "::" + str(current_user) + "::" + str(
                        artist_encoding) + "::" + str(playcount)
                    first_week_outfile.write(str_to_write)
                    first_week_outfile.write("\n")

                else:

                    # used for initializing at the bigining the current_user variable
                    if current_user == "":
                        current_user = user_encoding

                    try:
                        value = tmp_user_esteem[str(artist_encoding)]

                        # update the value of the key-artist
                        # first_week of listening of the artist = first_week (already encountered)
                        # last_week of listening of the artist = week_encoding
                        # partial total playcount of the artist = sum of playcounts
                        first_week = tmp_user_esteem[str(artist_encoding)]["first_week"]
                        new_last_week = week_encoding
                        new_total_playcount = int(value["total_playcount"]) + int(playcount)

                        new_value = {"first_week": first_week, "last_week": new_last_week,
                                     "total_playcount": new_total_playcount}

                        tmp_user_esteem[str(artist_encoding)] = new_value

                    except KeyError:
                        # if the error is rased it meand that the key is not present in the dict => add it
                        # first_week of listening of the artist = week_encoding
                        # last_week of listening of the artist = week_encoding
                        # partial total playcount of the artist = playcount
                        tmp_user_esteem[str(artist_encoding)] = {"first_week": week_encoding,
                                                                 "last_week": week_encoding, "total_playcount": playcount}

                        # keep track of the artist's first week of listening made by the user
                        str_to_write = str(week_encoding) + "::" + str(current_user) + "::" + str(
                            artist_encoding) + "::" + str(playcount)
                        first_week_outfile.write(str_to_write)
                        first_week_outfile.write("\n")


             # compute and write on file last user's esteem
            for key, value in tmp_user_esteem.items():

                try:
                    # get corresponding weeks from encodings (NEED STRING AS PARAMETER)
                    first_week = get_week_from_encoding(str(value["first_week"]))
                    last_week = get_week_from_encoding(str(value["last_week"]))

                    if first_week == -1 or last_week == -1:
                        continue  # it shouldn't happen, but we skip iteration

                    total_listening_period_unixtime = int(last_week) - int(first_week)
                    # 1 week = 604800 seconds
                    total_listening_period = math.floor(total_listening_period_unixtime / 604800)
                    total_playcounts = value["total_playcount"]

                    # write results on the user_artist_totalistening_periodlength file

                    str_to_write = str(current_user) + "::" + str(key) + "::" + str(total_playcounts) + "::" + str(
                        total_listening_period)

                    # file in which to write the the estimations that we have to make from the "week_user_artist_count.gz" file
                    user_artist_totalistening_periodlength = "total_listenings" + str(current_user) + \
                                                             "user_artist_totalistening_periodlength.gz"
                    # check if user_artist_totalistening_periodlength exists file, otherwise create it
                    check = check_if_file_exist(user_artist_totalistening_periodlength)
                    if check == -1:
                        # create user_artist_totalistening_periodlength file
                        touch(user_artist_totalistening_periodlength)

                    with gzip.open(user_artist_totalistening_periodlength, 'at', encoding='utf-8') as outfile:
                        outfile.write(str_to_write)
                        outfile.write("\n")
                        outfile.close()

                except Exception as e:  # it should not happen
                    print(e)
                    print()

            infile.close()


def convert_total_listenings_file_into_single_file():
    """
        Function which converts the total listenings directory's content into a single file
    """

    check = check_if_file_exist("user_artist_totalistening_periodlength.gz")

    if check == 0:
        return
    else:
        touch("user_artist_totalistening_periodlength.gz")

    total_listenings_directory = "total_listenings/"

    with gzip.open("user_artist_totalistening_periodlength.gz", 'a') as outfile:
        # get list of files from total_listenings directory
        list_of_files = os.listdir(total_listenings_directory)

        for file in list_of_files:
            file_path = total_listenings_directory + str(file)

            print("FILE ANALYZED = " + str(file_path))
            print()

            # read user's total listenings in order to construct a single total listening file
            with gzip.open(file_path, 'r') as infile:
                for line in infile:
                    outfile.write(line)
                infile.close()

        outfile.close()



def convert_first_week_file_into_directory():
    """
        Function which converts the first week file into a directory
    """

    first_week_directory = "first_week/"

    check = check_if_directory_exist(first_week_directory)

    if check == 0:
        return
    else:
        create_directory(first_week_directory)

    count = 1

    with gzip.open("first_week_user_artist_count.gz", 'r') as infile:
        for line in infile:
            line_array = line.decode("utf-8").split("::")

            print("LINE " + str(count) + " = " + str(line))
            count += 1

            # artist_encoding is the discriminator where to write => every file in the "first_week_directory"
            # is relative to all the users which have listen for the first time to a particular artist
            artist_file_name = first_week_directory + str(line_array[2]) + "first_week_artist_count.gz"
            check = check_if_file_exist(artist_file_name)
            if check == -1:
                touch(artist_file_name)
            with gzip.open(artist_file_name, 'a') as outfile:
                outfile.write(line)
            outfile.close()


        infile.close()


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

user_artist_totalistening_periodlength_builder()

# convert_total_listenings_file_into_single_file()

# add "user_artist_totalistening_periodlength.gz" file to the "leader_detection.zip" archive
# add_file_to_zip_directory("user_artist_totalistening_periodlength.gz", "leader_detection.zip")

# add "first_week_user_artist_count.gz" file to the "leader_detection.zip" archive
# add_file_to_zip_directory("first_week_user_artist_count.gz", "leader_detection.zip")

# convert_first_week_file_into_directory()

