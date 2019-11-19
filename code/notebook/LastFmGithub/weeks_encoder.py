import os
import zipfile

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

def encode_weeks():
    """
     Function which encodes the weeks of our study as integers (we start from the first Lastfm availabe week
     and we increment every week of 7 days, util we reach our end date). The result of our encoding is stored in
     the "weeks_map1" file
     N.B. First Lastfm available week is 16/07/2017  12:00 PM
    """

    check = check_if_file_exist("weeks_map1")

    if check == -1:
        touch("weeks_map1")

        with open("weeks_map1", 'a', encoding='utf-8') as o:

            start_date_pm = 1500206400  # 16/07/2017  12:00 PM
            end_date_pm = 1562500800  # 16/07/2019  12:00 PM
            end_date_increased = end_date_pm + 1  # + 1 for range(start_index, end_index - 1)

            i = 0  # current encoding

            # iterate every week (starting from the begin week of out study) and encode it in the "weeks_map1" file
            for week in range(start_date_pm, end_date_increased, 604800):     # +604800 => 1 week
                o.write(f"{week}\t{i}\n")
                i += 1

            o.close()


def add_file_to_zip_directory(filepath, zip_archive):
    """
        Function which add a file to a zip archive
        :param filepath filepath of the file to add to the zip archive
        :param zip_archive zip archive in which to add the file
    """

    # create the zip archieve, overwrite it if already exists
    z = zipfile.ZipFile(zip_archive, 'w')

    # write file to the zip archive
    z.write(os.path.join(filepath))

    # remove original file
    os.remove(filepath)


# encode the weeks of our study as integers
encode_weeks()

# add "weeks_map1" file to the "leader_detection.zip" archive
add_file_to_zip_directory("weeks_map1", "leader_detection.zip")
