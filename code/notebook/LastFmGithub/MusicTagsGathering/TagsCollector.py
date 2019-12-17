import gzip
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


def check_if_main_tag_is_in_tag_word(tag_word):
    """
        Function which check if the tag_word has one one of the main tags as substring in its name
        :param tag_word: tag in which to check if one of the main tags is present as a substring
        :return main tag: if the tag_word contains a main_tag as substring
                -1: otherwise
    """
    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic", "hip-hop", "hiphop", "hip hop",
                 "rap", "jazz", "latino", "pop", "r&b", "rnb", "soul",  "reggae", "rock"]

    check = ""

    for main_tag in main_tags:

        if main_tag in tag_word:

            if main_tag == "hip-hop" or main_tag == "hiphop" or main_tag == "hip hop" or main_tag == "rap":
                check = "hip-hop" + "/" + "rap"
            elif main_tag == "r&b" or main_tag == "soul":
                check = "r&b" + "/" + "soul"
            else:
                check = main_tag
            break

    return check


def check_tags_mapping():
    """
        Function which checks if all the possible tags encountered are mapped in a main music genre.
        If a tag isn't encoded the tag is written on the "not_mapped_tags"
    """

    check = check_if_file_exist("not_mapped_tags")

    if check == -1:
        touch("not_mapped_tags")
    else:
        f = open("not_mapped_tags", "w")
        f.write("")
        f.close()

    with open("merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_encodings = json.load(infile)
        infile.close()

    with open("not_mapped_tags", 'a') as outfile:

        artists_info_directory = "artists_info/"

        # get directory's list of files
        list_of_files = os.listdir(artists_info_directory)
        for file in list_of_files:
            file_path = artists_info_directory + file

            # for every artist's file check if its tags are encoded
            try:
                with gzip.open(file_path, 'rt', encoding='utf-8') as infile:
                    artist_info = json.load(infile)
                    infile.close()

                tags_list = artist_info["tags"]

                # iterate over the tags list and for every artist's tag chech
                # if there is a corresponding mapping in the "music_tags_map.json" file,
                # otherwise write the tag on the "not_mapped_tags" file
                for tag in tags_list:

                    tag = tag.lower()

                    # check if one of the main tag is present as substring in the current tag
                    # (if its present the mapping is the main tag returned)
                    check = check_if_main_tag_is_in_tag_word(tag)

                    if check == "":
                        try:
                            tag_encoding = tags_encodings[str(tag)]
                        except KeyError:
                            # if the error is raised it means that the tag hasn't an encoding
                            outfile.write(tag)
                            outfile.write("\n")
            except:
                continue


def get_single_word_tags():
    """
        Function which retrieves the keys composed by a single word and present
        in the "merged_music_tags_map.json" file
        return a dict formed by single word keys and the corresponding values
    """

    single_word_tags_dict = {}

    # read the tags mapped crawled from musicgenrelist and wikipedia
    with open("merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    for key, value in tags_map.items():

        # check if the key is composed by a single word
        if len(key.split()) == 1:
            single_word_tags_dict[str(key)] = value

    return single_word_tags_dict


def collect_nations_and_nationalities():
    """
        Function which retrieves the nations and nationalities present in the "NationAndNationalities" file in order
        to add them to the "not_considered_tags" file and so exclude this words from the tags' classification
    """
    f = open("NationAndNationalities", "r")
    ff = open("not_considered_tags", "a")
    for line in f:
        line_array = line.split()
        ff.write(line_array[0].lower())
        ff.write("\n")
        ff.write(line_array[1].lower())
        ff.write("\n")
    f.close()
    ff.close()


def collect_musical_instruments():
    """
        Function which retrieves the musical instruments present in the "MusicalInstruments" file in order
        to add them to the "merged_music_tags_map.json" file and map them to the "classical genre"
    """
    f = open("MusicalInstruments", "r")

    # read the tags mapped crawled from musicgenrelist and wikipedia
    with open("merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    for line in f:
        l = line.strip()
        tag_to_add = l.lower()
        tags_map[str(tag_to_add)] = "classical"

    f.close()

    # write new dict to the tags mapped crawled from musicgenrelist and wikipedia
    with open("merged_music_tags_map.json", 'wt', encoding='utf-8') as outfile:
        json.dump(tags_map, outfile, indent=4)
        outfile.close()


def clean_not_mapped_tags():
    """
        Function which checks the tags present in the "not_considered_tags" file (getting rid of eventual duplicates
        present in it, because the file is manually updated and contains the tags that I don't  consider)
        and gets rid of all the tags contained here and eventually present in the "not_mapped_tags" file
        (this file is used to track the tags which don't have already a mapping in the "merged_music_tags_map.json").
        In addition I check if some tag present in the "not_mapped_tags" has as substring a tag present in
        the "merged_music_tags_map.json" file and I keep track of them in the "to_add.json" file
        to add them latter to the "merged_music_tags_map.json" file
    """

    check = check_if_file_exist("to_add.json")

    if check == -1:
        touch("to_add.json")

    single_word_tags_dict = get_single_word_tags()

    print(single_word_tags_dict)
    print(len(single_word_tags_dict))

    # read the tags mapped crawled from musicgenrelist and wikipedia
    with open("merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    # get rid of duplicated tags in "not_considered_tags"
    not_considered_tags_list = []

    f = open("not_considered_tags", "r")
    for line in f:
        l = line.split("\n")

        if l[0] not in not_considered_tags_list:

            # check if the tag is not already mapped in the "merged_music_tags_map.json" (in this
            # case it was my fault to add it to the "not_considered tags" file)
            try:
                tag_encoding = tags_map[str(l[0])]
            except KeyError:
                # if the error is raised it means that the not_considered_tag is not in the mapping file, so I
                # added it correctly to the "not_considered_tags" file beacuse I wanted to exclude it
                not_considered_tags_list.append(l[0])
    f.close()

    # get rid of the file's content (I will append it ex-novo)
    f = open("not_considered_tags", "w")
    f.write("")
    f.close()
    # write not considered tags without duplicates on file
    f = open("not_considered_tags", "a")
    for tag in not_considered_tags_list:
        f.write(tag)
        f.write("\n")
    f.close()

    # get all the tags not mapped
    not_mapped_tags_list = []
    f = open("not_mapped_tags", "r")
    for line in f:
        l = line.split("\n")
        not_mapped_tags_list.append(l[0])
    f.close()

    # get rid of the file's content (I will append it ex-novo)
    f = open("not_mapped_tags", "w")
    f.write("")
    f.close()

    tmp_to_add_dict = {}

    # iterete over the not_mapped_tags_list and get rid of any tag also present in the
    # not_considered_tags_list. This allow me to get rid of the unwanted tags present in the
    # "not_mapped_tags" and facilitate my manually interpetetion of the corresponding file (I have to manually
    # read all the not_mapped_tags and understand why they aren't mapped)
    f = open("not_mapped_tags", "a")
    for tag in not_mapped_tags_list:
        if tag not in not_considered_tags_list:

            for key, value in single_word_tags_dict.items():
                if key in tag:
                    tmp_to_add_dict[str(tag)] = str(value)

            f.write(tag)
            f.write("\n")

    f.close()

    print(tmp_to_add_dict)
    print(len(tmp_to_add_dict))

    with open("to_add.json", 'wt', encoding='utf-8') as outfile:
        json.dump(tmp_to_add_dict, outfile, indent=4)
        outfile.close()


    """
    for i in range(len(not_mapped_tags_list) - 1, -1, -1):  # iterate over reversed indices's
        if str(not_mapped_tags_list[i]) in str(not_considered_tags_list):
            del not_mapped_tags_list[i]

    # write the not_mapped_tags' clean-up made on  file
    for tag in not_mapped_tags_list:
        f.write(tag)
        f.write("\n")
    f.close()
    """


def merge_two_dicts():

    # read the tags mapped crawled from musicgenrelist and wikipedia
    with open("merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    # read the tags to add to the tags map
    with open("to_add.json", 'rt', encoding='utf-8') as infile:
        to_add_dict = json.load(infile)
        infile.close()

    for key, value in to_add_dict.items():
        try:
            tags_map[str(key)]
        except KeyError:
            tags_map[str(key)] = value

    with open("merged_music_tags_map.json", 'wt', encoding='utf-8') as outfile:
        json.dump(tags_map, outfile, indent=4)
        outfile.close()

# collect_nations_and_nationalities()
# collect_musical_instruments()
check_tags_mapping()
clean_not_mapped_tags()
merge_two_dicts()
