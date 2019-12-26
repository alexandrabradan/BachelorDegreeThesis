import gzip
import os


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


def convert_single_file_into_directory():
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


convert_single_file_into_directory()
