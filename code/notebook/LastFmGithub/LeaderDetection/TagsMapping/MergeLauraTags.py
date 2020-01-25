import json
import os


def touch(path):
    """
        Create a new file in the specified path
        :param path: path where to create the new file
    """
    with open(path, 'a') as f:
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


def get_laura_encoding(tag):
    """
        Function which checks if the tag passed by argument is present in the "laura_music_genre_list.json"
        file
    """

    # read Laura's tags
    with open("laura_music_genre_list.json", 'rt', encoding='utf-8') as infile:
        laura_map = json.load(infile)
        infile.close()

    laura_encoding = ""

    for key, value in laura_map.items():
        for elem in value:  # value is a list
            if elem == tag:
                laura_encoding = key
                break

    return laura_encoding


def merge_tags_files():

    # check = check_if_file_exist("collided_music_tags_map_with_laura.json")
    check = check_if_file_exist("original_collided_music_tags_map_with_laura.json")

    if check == 0:
        return
    else:
        # touch("collided_music_tags_map_with_laura.json")
        touch("original_collided_music_tags_map_with_laura.json")

    # read the tags mapped crawled from musicgenrelist and wikipedia
    # with open("merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
    with open("original_merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    collided_tags_map = {}

    for key, value in tags_map.items():
        laura_encoding = get_laura_encoding(key)

        if laura_encoding != "":
            if value != laura_encoding:
                collided_tags_map[str(key)] = [value, laura_encoding]

    # write collided_tags_map on file
    # with open("collided_music_tags_map_with_laura.json", 'wt', encoding='utf-8') as outfile:
    with open("original_collided_music_tags_map_with_laura.json", 'wt', encoding='utf-8') as outfile:
        json.dump(collided_tags_map, outfile, indent=4)
        outfile.close()

merge_tags_files()
