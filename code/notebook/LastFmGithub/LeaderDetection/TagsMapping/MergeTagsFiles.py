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


def merge_tags_files():
    """
        Function which checks the tags in common between the "music_tag_map.json" file extracted from
        "https://www.musicgenreslist.com/" and the "music_wikipedia_tags_map.json" exctracted from
        "https://en.wikipedia.org/wiki/List_of_music_styles". The result of the merging is written on the
        "original_merged_music_tags_map.json" file and the tags that collide are written on the "collided_music_tags_map.json"
        file, in the following format:
            "key_which_collide": [musicgenre_mapping, wikipedia_mapping]
        From multiple runs I detected the the best assign to the colliding tags is given by the  musicgenre_mapping,
        so for the colliding keys I ignore the wikipedia mapping.
    """

    check = check_if_file_exist("merged_music_tags_map.json")

    if check == -1:
        touch("original_merged_music_tags_map.json")

    check = check_if_file_exist("collided_music_tags_map.json")

    if check == -1:
        touch("collided_music_tags_map.json")

    # read the tags mapped crawled from Wikipedia
    with open("music_wikipedia_tags_map.json", 'rt', encoding='utf-8') as infile:
        wikipedia_tags_map = json.load(infile)
        infile.close()

    # read the tags mapped crawled from musicgenrelist
    with open("music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    merged_tags_map = {}
    collided_tags_map = {}

    # iterate over the wikipedia tags and check if they are present in the musicgenrelist tags
    for wiki_tag in wikipedia_tags_map:
        try:
            tag_encoding = tags_map[str(wiki_tag)]

            # wiki_tag is present in the  musicgenrelist tags => I check if the mapping is the same
            wiki_tag_encoding = wikipedia_tags_map[str(wiki_tag)]

            if tag_encoding != wiki_tag_encoding:  # mapping is not the same
                collided_tags_map[str(wiki_tag)] = [tag_encoding, wiki_tag_encoding]

                # I assigned musicgenrelist in any away to the tag
                merged_tags_map[str(wiki_tag)] = tag_encoding

            else:
                merged_tags_map[str(wiki_tag)] = tag_encoding

        except KeyError:
            # if the error is raised it means that the wikipedia tag is not presente in the musicgenrelist tags
            merged_tags_map[str(wiki_tag)] = wikipedia_tags_map[str(wiki_tag)]
            continue

    # iterate over the musicgenrelist tags in order to check if they are present in the merged_tags_map dict
    for tag in tags_map:
        try:
            tag_encoding = merged_tags_map[str(tag)]
        except Exception:
            # if the error is raised it means that the tag is not present in the  merged_tags_map dict
            merged_tags_map[str(tag)] = tags_map[str(tag)]

    # write merged_tags_map dict on file
    with open("original_merged_music_tags_map.json", 'wt', encoding='utf-8') as outfile:
        json.dump(merged_tags_map, outfile, indent=4)
        outfile.close()

    # write collided_tags_map on file
    with open("collided_music_tags_map.json", 'wt', encoding='utf-8') as outfile:
        json.dump(collided_tags_map, outfile, indent=4)
        outfile.close()

# merge the two tags files
merge_tags_files()
