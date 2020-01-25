import json
import re
import os
import sys
import time
import urllib

import requests
import shutil
import pandas as pd


def get_artist_from_encoding(artist_encoding):
    """
        Function which returns the corresponding artist of the artist_encoding passed by argument
        :return:  artist name of the week corresponding to the artist_encoding
                 "" otherwise
    """

    b = bytes(".*" + artist_encoding + "\n", 'utf-8')
    my_regex = re.compile(b)

    with open("artists_map1", 'rb') as f:
        for line in f:
            match = my_regex.match(line)

            if match is not None:
                line_array = line.decode("utf-8").split('\t')
                week = line_array[0].strip('\t')
                week = week.strip('\n')
                f.close()
                return week

    # the artist_encoding is not present in the  "artists_map1" file
    return ""


def fetch_HTTP_response(url, max_num_attempts_read_url):
    """
       Try to read HTTP response from an URL, until reaching success
       or the maximum number of attempts
       :param url: url from which to read HTTP response
       :param max_num_attempts_read_url: max number of retires to read the HTTP response
    """
    try:
        response = urllib.request.urlopen(url)

        # decode: decodes the string using the codec registered for encoding(UTF-8, encoding errors
        # raise a UnicodeError)
        data = json.loads(response.read().decode('UTF-8', 'strict'))
        response.close()

        return data
    except requests.exceptions.ConnectionError as e:
        # suspend requests for 5 seconds
        time.sleep(5)

        # try again to read HTTP response, if there are still attempts
        max_num_attempts_read_url -= 1
        if max_num_attempts_read_url > 0:
            fetch_HTTP_response(url, max_num_attempts_read_url)
        else:
            print()
            print(e)
            print("Connection refused: " + url)
            return {}  # failure, return empty json response
    except urllib.error.HTTPError as e:
        # suspend requests for 5 seconds
        time.sleep(5)

        # try again to read HTTP response, if there are still attempts
        max_num_attempts_read_url -= 1
        if max_num_attempts_read_url > 0:
            fetch_HTTP_response(url, max_num_attempts_read_url)
        else:
            print()
            print(e)
            print("Problem caused by url: " + url)
            return {}  # failure, return empty json response
    except Exception as e:  # like OpenSSL.SSL.SysCallError: (10053, 'WSAECONNABORTED')
        # suspend requests for 5 seconds
        time.sleep(5)

        # try again to read HTTP response, if there are still attempts
        max_num_attempts_read_url -= 1
        if max_num_attempts_read_url > 0:
            fetch_HTTP_response(url, max_num_attempts_read_url)
        else:
            print()
            print(e)
            print("Problem caused by url: " + url)
            return {}  # failure, return empty json response


def return_artist_info(artist):
    """
        Return artist's info in dict format
        :param artist: artist's name for which we have to retrieve info
    """

    lfm_apikey = '3878e8d3f604944d12eaef5f34ada2d1'

    # ask artist's info to the API
    query = "&artist=%s" % (
        urllib.parse.quote_plus(artist))  # convert whitespace in + and utf-8 '\x..' char in '%'
    url = "http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&api_key=%s&format=json" % lfm_apikey
    url = url + query

    max_num_attempts_read_url = 3
    data = fetch_HTTP_response(url, max_num_attempts_read_url)

    if data != {} and data is not None:

        if len(data) > 0 and 'artist' in data:

            try:
                artist_profile = {
                    "name": data['artist']['name'],
                    # "mbid": data['mbid'],
                    # "stats": data['artist']['stats'],  # translated
                    "listeners": data['artist']['stats']['listeners'],
                    "playcount": data['artist']['stats']['playcount'],
                    "similar": [data['artist']['similar']['artist'][x]['name'] for x in
                                range(0, len(data['artist']['similar']['artist']))],
                    "tags": [data['artist']['tags']['tag'][x]['name'] for x in
                             range(0, len(data['artist']['tags']['tag']))],
                }

                return artist_profile  # return artist's info

            except KeyError as e:
                # print("KeyError " + str(e))
                # print(url)
                # print("In get_artist_artist_tag: " + "artist= " + artist)
                return  {}  # return empty dict
            except Exception as e:
                # print(e)
                # print(url)
                # print("In get_artist_artist_tag: " + "artist= " + artist)
                return  {}  # return empty dict

    return {}  # return empty dict


def get_more_tags(artist):
    """
        Function which retrieves the tags associated to the artist, as well as the number of users attributed
        them, using the Lastf.fm  API <<artist.getTopTags>>
        :param artist: artist for which to retrieve the tags associated
        :return a list containing the tags associated to the artist
    """

    lastfm_tags_list = []

    lfm_apikey = '3878e8d3f604944d12eaef5f34ada2d1'

    query = "&artist=%s" % (
        urllib.parse.quote_plus(artist))  # convert whitespace in + and utf-8 '\x..' char in '%'
    url = "http://ws.audioscrobbler.com/2.0/?method=artist.getTopTags&api_key=%s&format=json" % lfm_apikey
    url = url + query

    max_num_attempts_read_url = 3
    data = fetch_HTTP_response(url, max_num_attempts_read_url)

    if data != {} and data is not None:

        try:

            for tag in data['toptags']['tag']:
                t = str(tag['name']).lower()
                pair = (t, tag['count'])
                lastfm_tags_list.append(pair)

            return lastfm_tags_list

        except Exception:
            return lastfm_tags_list

    return lastfm_tags_list


def artists_for_which_I_have_already_info():
    """
        Function which returns a list of the artists already analyzed in the "filtered_artists_with_main_tag"
        (artists who have already a main tag associated)
    """

    artists_already_examinated = []

    f = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        artist_encoding = int(line_array[0])
        artists_already_examinated.append(artist_encoding)
    f.close()

    return artists_already_examinated


def collect_tags_list_for_target_artists():
    """
        Function which collect the tags and info of the target artists present in the "filtered_target_artists" file.
        The file is a restriction among all the artists in the dataset and contains the artists:
        a) with at least 100 listeners on Last.fm
        b) listened by at least 5 seed users
        c) whit music tags on Last.fm and relative taggings used by at least 100 people
        The format of the file is:
            week_encoding::user_encoding::artist_encoding::playcount
        and every line corresponds to the absolute first listening of an artists.
    """

    # artists already classified because I found leaders with the "Three dimensions of social prominance"
    artists_already_examinated = artists_for_which_I_have_already_info()

    f = open("filtered_target_artists", "r")
    for line in f:
        line_array = line.split("::")
        artist_encoding = line_array[2]

        if int(artist_encoding) not in artists_already_examinated:

            # get artist from encoding
            artist = get_artist_from_encoding(artist_encoding)

            if artist == "":
                continue  # next artist to analyze

            # get artist's info
            artist_info = return_artist_info(artist)

            if artist_info == {}:
                continue  # next artist to analyze

            # get artist's top tags
            artist_top_tags = get_more_tags(artist)

            if artist_top_tags == {}:
                continue  # next artist to analyze

            # overwrite artist's info tags with the one retrived from "artist.getTopTags" beacuse it
            # returns also how many user assigned the tags
            artist_info['tags'] = artist_top_tags

            # create a dedicate artists info file and populate it with his info
            artist_info_file = "filtered_target_artists_info/" + str(artist_encoding) + "_artist_info.json"
            with open(artist_info_file, 'wt', encoding='utf-8') as outfile:
                json.dump(artist_info, outfile, indent=4)
                outfile.close()

            print("Artist = " + str(artist) + "'s tags and info retrieved")
    f.close()


def check_presence_in_tags_list(artist_tags, tag):
    """
        Function which checks the presence of the tag passed by argument in the artist_tags list
        (format: [[tag_1, counter_1], ... , [tag_n, counter_n]])
    :param artist_tags: list in which to check the presence of the tag
    :param tag: tag for which to check the presence in the list
    :return: True: if the tag is presenr in the list
             False otherwise
    """
    for key, value in artist_tags:
        if key == tag:
            return True

    return False


def retrieve_similar_artists_tags():
    """
    Function called if after the first assignation still exist artists without a main genre assigned,
    due to  multiple choises to make (file "filtered_artists_with_multiple_main_tags").
    I try to collect similar artists tags and make a huge mash up among them in order to to try to have an assignation.
    """

    artists_already_examinated = artists_for_which_I_have_already_info()

    artists_with_multiple_or_no_main_tags = open("filtered_artists_without_main_tag", "r")
    for line in artists_with_multiple_or_no_main_tags:
        line_array = line.split("::")
        artist_encoding = line_array[0]

        if int(artist_encoding) not in artists_already_examinated:

            # get artist's name
            target_artists_info_directory = "filtered_target_artists_info/"
            file_path = target_artists_info_directory + str(artist_encoding) + "_artist_info.json"

            with open(file_path, 'rt', encoding='utf-8') as infile:
                artist_info = json.load(infile)
                infile.close()

            artist = artist_info["name"]
            print("Artist = " + str(artist) + " retirved similar artists tags")

            similar_artists = artist_info['similar']
            artist_tags = artist_info['tags']

            for similar_art in similar_artists:
                # get all tags associated with the artist, without restrictions, in order to
                # try to enstablish a main tag from them
                lastfm_tags_list = get_more_tags(similar_art)

                for tag, counter in lastfm_tags_list:
                    if check_presence_in_tags_list(artist_tags, tag) is False:
                        t = str(tag.lower())
                        pair = [t, counter]
                        artist_tags.append(pair)

            # overwrite previous tags
            artist_info['tags'] = artist_tags

            # create a dedicate artists info file and populate it with his info
            with open(file_path, 'wt', encoding='utf-8') as outfile:
                json.dump(artist_info, outfile, indent=4)
                outfile.close()

    artists_with_multiple_or_no_main_tags.close()


def get_single_word_tags():
    """
        Function which retrieves the keys composed by a single word and present
        in the "original_merged_music_tags_map.json" file
        return a dict formed by single word keys and the corresponding values
    """

    single_word_tags_dict = {}

    # read the tags mapped crawled from musicgenrelist and wikipedia
    with open("original_merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    for key, value in tags_map.items():

        # check if the key is composed by a single word
        if len(key.split()) == 1:
            single_word_tags_dict[str(key)] = value

    return single_word_tags_dict


def simple_classification():
    """
        Function which tries to classify all the artists present in the "filtered_target_artists_info"
        directory with a main music genre. The main music tag is assigned as the tag
        which have the grater value, calculated as:
            tag_counter_i / sum(tag_counter_1, ..., tag_counter_n)
    """

    # first empty file
    artists_without_main_tag = open("filtered_artists_without_main_tag", "w")
    artists_without_main_tag.write("")
    artists_without_main_tag.close()

    artists_already_examinated = artists_for_which_I_have_already_info()

    artists_without_main_tag = open("filtered_artists_without_main_tag", "a")
    artists_with_main_tag = open("filtered_artists_with_main_tag", "a")

    # read the tags mapped crawled from musicgenrelist and wikipedia
    with open("original_merged_music_tags_map.json", 'rt', encoding='utf-8') as infile:
        tags_map = json.load(infile)
        infile.close()

    single_word_tags_dict = get_single_word_tags()

    target_artists_info_directory = "filtered_target_artists_info/"
    list_of_files = os.listdir(target_artists_info_directory)
    for file in list_of_files:
        # read artist's info in order to find his main music tag
        file_path = target_artists_info_directory + file

        tmp_mappings = {}  # artist's tag mappings
        tmp_num_of_total_tagging = 0  # total number of tags assignation
        tmp_mappings_helper = {}  # artist's tag mappings mean sum

        with open(file_path, 'rt', encoding='utf-8') as infile:
            artist_info = json.load(infile)
            infile.close()

            artist = artist_info['name']
            file_array = file.split("_artist_info.json")
            artist_encoding = int(file_array[0])

            if artist_encoding in artists_already_examinated:
                 continue

            for tag, counter in artist_info['tags']:
                try:
                    mapping = tags_map[str(tag)]
                except KeyError:
                    # if the error is raised it means that the tag is not mapped
                    mapping = ""

                    # I try to force the finding of an existing mapping,
                    # checking if the current tag has as substring a single world tag
                    # already mapped. If so I assign to the current tag the mapping
                    # otherwise I skip the current tag in the computing
                    for key, value in single_word_tags_dict.items():
                        if key in tag:
                            mapping = value

                    if mapping == "":
                        continue   # I skip the current tag in the computing

                # I update the main music genre counter
                try:
                    tmp_counter = tmp_mappings[str(mapping)]
                    new_tmp_counter = int(tmp_counter) + int(counter)
                    tmp_mappings[str(mapping)] = new_tmp_counter

                except KeyError:
                    tmp_mappings[str(mapping)] = int(counter)
                except Exception:
                    print("Exceptio cause by tag = " + str(tag))
                    sys.exit(-999)

                # I update the number of tags assignations
                tmp_num_of_total_tagging += int(counter)

            if len(tmp_mappings) == 0:
                str_to_write = str(artist_encoding) + "::" + str(artist_info['tags']) + "\n"
                artists_without_main_tag.write(str_to_write)
                continue  # next artist

            # I calculate the mean sum of the mappings retrieved
            for mapping, counter in tmp_mappings.items():
                mean_sum = float(int(counter) / tmp_num_of_total_tagging)
                tmp_mappings_helper[str(mapping)] = mean_sum

            # among the mean sums I get the greater one, in order to enstablish
            # which is artist's main music genre. If there are multiple tags with the same highest mean sums
            # [AND THEY ARE SINCE LAST.FM DOENS'T SHOW TAGGINGS ABOVE 100] since Last.fm returns a list of tags in
            # decrescent order of taggings and since the tmp_mappings_helper dictionary was build iterating over this
            # decrescent list, the first tag with the highest mean sums that I encountered is the most representative
            # (on Last.fm it was used more times, even if its taggings isn't greater than 100 and so conflicts with
            # other tags in the same situation, but less frequently used)
            max_key = -1
            max_value = -1
            for mapping, mean_sum in tmp_mappings_helper.items():
                if float(mean_sum) > float(max_value):
                    max_key = mapping
                    max_value = mean_sum

            str_to_write = str(artist_encoding) + "::" + str(max_key) + "\n"
            artists_with_main_tag.write(str_to_write)

    artists_without_main_tag.close()
    artists_with_main_tag.close()


def collect_filtered_target_artists_with_main_tag():
    """
        Function which collect the artists present in the "filtered_target_artists_with_main_tag" file
        (the files contains the artists classified with a main music tag, which have at least 100 listeners
        on Last.fm and which were listened by at least 5 seed users of my dataset)
    """

    f = open("filtered_artists_with_main_tag", "r")
    filtered_target_artists = []

    for line in f:
        line_array = line.split("::")
        artist_encoding = int(line_array[0])
        filtered_target_artists.append(artist_encoding)

        # print("Collected filtered target artist = " + str(artist_encoding))

    # print("len filtered target artist = " + str(len(filtered_target_artists)))
    return filtered_target_artists


def delete_files_of_filtered_artists_without_main_tag():
    filtered_target_artists = collect_filtered_target_artists_with_main_tag()
    filtered_target_artists_directory = "filtered_target_artists_info/"


    list_of_files = os.listdir(filtered_target_artists_directory)
    for file in list_of_files:

        file_array = file.split("_artist_info.json")
        artist_encoding = int(file_array[0])

        if artist_encoding not in filtered_target_artists:
            # get complete file's path
            file_path = filtered_target_artists_directory + file
            # remove file
            os.remove(file_path)
            print(file_path + " removed")


def check_existance_of_all_artists_with_main_tag_files():
    """
        Function which checks if all the artists with a main tag have also an info file dedicated in
        the "filtered_artists_info" directory
    """
    filtered_target_artists = collect_filtered_target_artists_with_main_tag()
    filtered_target_artists_directory = "filtered_target_artists_info/"

    missing_files = []

    for artist in filtered_target_artists:
        file_path = filtered_target_artists_directory + str(artist) + "_artist_info.json"

        try:
            with open(file_path, 'rt', encoding='utf-8') as infile:
                artist_info = json.load(infile)
                infile.close()
        except FileNotFoundError:
            missing_files.append(artist)

    if len(missing_files) == 0:
        print("All filtered target artists files are in the \"filtered_artists_info\" directory")
    else:
        print("Missing files for the following artists = " + str(missing_files))


def get_main_music_encoding(main_tag):
    """
        Function which return the main music genres encoding
        :param main_tag: main music genre for which to return the encoding
    """

    # the main tags are encoded with the index in this array
    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic",
                 "hip-hop/rap", "jazz", "latin", "pop", "r&b/soul", "reggae",
                 "rock"]

    for i in range(0, len(main_tags)):
        if str(main_tag) == str(main_tags[i]):
            return i

    # if the main_tag is not present in the array
    return -1


def get_artist_main_tag(artist_encoding):
    """
        Function which returns the main tags associated with the artist passed by argument
        :param artist_encoding: artist for which to retrieve the main tag present in the "artists_with_main_tag" file
    """

    artist_with_main_tag_file = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in artist_with_main_tag_file:
        line_array = line.split("::")
        art_encoding = line_array[0]
        main_tag = line_array[1].replace("\n", "")

        if int(art_encoding) == int(artist_encoding):
            artist_with_main_tag_file.close()
            return main_tag

    artist_with_main_tag_file.close()

    # if the artist is not present in the file
    return ""

def create_final_leadears_DIRECTED_file():
    """
        Function which create the definitive file for the Leader Detection strategy
    :return:
    """
    final_leaders_file = open("final_leaders_DIRECTED", "a")
    artists_with_main_tag = artists_for_which_I_have_already_info()

    leaders_file = open("leaders_DIRECTED", "r", encoding="utf-8")
    for line in leaders_file:
        line_array = line.split("::")
        artist_encoding = int(line_array[0])

        if artist_encoding in artists_with_main_tag:
            # get artist's main tag
            action_main_tag = get_artist_main_tag(artist_encoding)
            if action_main_tag == "":
                print("action = " + str(artist_encoding))
                print("action main tag empty")
                sys.exit(-1)

            # get artist's main tag encoding
            action_main_tag_encoding = get_main_music_encoding(action_main_tag)
            if action_main_tag_encoding == -1:
                print("action = " + str(action))
                print("action's main tag wrong spelled")
                sys.exit(-1)

            tmp_new_line = line.replace("\n", "")
            new_line = tmp_new_line + "::" + str(action_main_tag_encoding) + "\n"
            final_leaders_file.write(new_line)

    leaders_file.close()
    final_leaders_file.close()

def create_univocal_final_leaders_file():
    """
        Function which take account of the univocal leaders present in the "final_leaders_DIRECTED" file
        and write them on the "univocal_final_leaders" file
    """

    final_leaders = []

    univocal_final_leaders_file = open("univocal_final_leaders", "a")
    leaders_file = open("final_leaders_DIRECTED", "r", encoding="utf-8")
    for line in leaders_file:
        line_array = line.split("::")
        user_encoding = line_array[1]

        if int(user_encoding) not in final_leaders:
            final_leaders.append(int(user_encoding))

            univocal_final_leaders_file.write(str(user_encoding))
            univocal_final_leaders_file.write("\n")

    leaders_file.close()
    univocal_final_leaders_file.close()

def create_filtered_artists_with_leaders():
    unique_artists = []

    filtered_artists_with_leaders_file = open("filtered_artists_with_leaders", "a")
    final_leaders_file = open("final_leaders_DIRECTED", "r", encoding="utf-8")
    for line in final_leaders_file:
        line_array = line.split("::")
        artist_encoding = int(line_array[0])
        tag_encoding = int(line_array[7].replace("\n", ""))

        if artist_encoding not in unique_artists:
            unique_artists.append(artist_encoding)

            str_to_write = str(artist_encoding) + "::" + str(tag_encoding) + "\n"
            filtered_artists_with_leaders_file.write(str_to_write)

    filtered_artists_with_leaders_file.close()
    final_leaders_file.close()


def convert_final_leaders_DIRECTED_file_to_csv():
    """
        Function which converts the "final_leaders_DIRECTED" file into a .cvs one
    """

    final_diffusion_trees_info_file = open("final_leaders_DIRECTED.csv", "a")

    final_leaders_file = open("final_leaders_DIRECTED", "r", encoding="utf-8")
    action = "action"
    leader = "leader"
    tribe = "tribe"
    width = "width"
    depth = "depth"
    mean_depth = "mean_depth"
    strength = "strength"
    music_tag = "music_tag"
    final_diffusion_trees_info_file.write(f"{action}\t{leader}\t{tribe}\t{depth}\t{mean_depth}\t{width}\t{strength}\t{music_tag}\n")

    for line in final_leaders_file:
        line_array = line.split("::")
        action = int(line_array[0])
        leader = int(line_array[1])
        l_tribe = int(line_array[2])  # diffusion tree's size
        l_depth = float(line_array[3])
        l_mean_depth = float(line_array[4])
        l_width = float(line_array[5])
        l_strength = float(line_array[6])
        l_music_tag = int(line_array[7].replace("\n", ""))

        print(f"{action}\t{leader}\t{l_tribe}\t{l_depth}\t{l_mean_depth}\t{l_width}\t{l_strength}\t{l_music_tag}\n")

        final_diffusion_trees_info_file.write(f"{action}\t{leader}\t{l_tribe}\t{l_depth}\t{l_mean_depth}\t{l_width}\t{l_strength}\t{l_music_tag}\n")

    final_leaders_file.close()
    final_diffusion_trees_info_file.close()


def check_if_artist_is_in_list(artists_with_main_tag, artist):
    """
        Function which checks if the artist passed by argument is present
        in the list and returns the relative value associated (memorized also
        in the list)
    :return: return the value associated with the artist if present
            empty string otherwise
    """
    for art, main_tag in artists_with_main_tag:
            if str(artist) == str(art):
                return main_tag
    return ""

def group_directories(directory):
    """
        Support function for the "group_directories_according_to_main_music_tag" function
        :param directory: "leaders_diffusion_trees/" or "less_then_4_nodes" directory
    """

    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic",
                 "hip-hop_rap", "jazz", "latin", "pop", "r&b_soul", "reggae",
                 "rock"]

    # create the 13 directories
    for main_t in main_tags:
        dirName = directory + main_t

        if not os.path.exists(dirName):
            os.mkdir(dirName)

    # create one extra directory for the excluded files
    excluded_directory = directory + "excluded/"
    if not os.path.exists(excluded_directory):
        os.mkdir(excluded_directory)

    artists_with_main_tag = []

    # collect artists who have a main tag assigned
    artists_with_main_tag_file = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in artists_with_main_tag_file:
        line_array = line.split("::")
        artist_encoding = line_array[0]
        artist_main_tag = line_array[1].replace("\n", "")
        pair = (artist_encoding, artist_main_tag)
        artists_with_main_tag.append(pair)
    artists_with_main_tag_file.close()

    # iterate over the files present in the directory, in order to
    # group them in the 14 directories just created, according with the tag assignation
    # made in the "artists_with_main_tag" file
    list_of_files = os.listdir(directory)
    for file in list_of_files:
        file_path = directory + file

        if file not in main_tags and file != "excluded":
            # get artist's encoding
            file_array = file.split("_")

            print(file_array)

            # artist is encoded in the third position of the file name
            artist_encoding = file_array[3]

            # get artist's main tag
            artist_main_tag = check_if_artist_is_in_list(artists_with_main_tag, artist_encoding)

            if artist_main_tag != "":
                # move file into corresponding directory
                if artist_main_tag == "alternative":
                    destination = directory + "alternative/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "blues":
                    destination = directory + "blues/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "classical":
                    destination = directory + "classical/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "country":
                    destination = directory + "country/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "dance":
                    destination = directory + "dance/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "electronic":
                    destination = directory + "electronic/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "hip-hop/rap":
                    destination = directory + "hip-hop_rap/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "jazz":
                    destination = directory + "jazz/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "latin":
                    destination = directory + "latin/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "pop":
                    destination = directory + "pop/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "r&b/soul":
                    destination = directory + "r&b_soul/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "reggae":
                    destination = directory + "reggae/" + file
                    shutil.move(file_path, destination)
                elif artist_main_tag == "rock":
                    destination = directory + "rock/" + file
                    shutil.move(file_path, destination)

            else:  # artist hasn't a main tag
                destination = excluded_directory + file
                shutil.move(file_path, destination)


def group_directories_according_to_music_tags():
    """
        Function which creates 13 directories, one for every main genre:
            ['Alternative', 'Blues', 'Classical', 'Country', 'Dance', 'Electronic', 'Hip-Hop/Rap', 'Jazz',
            'Latin', 'Pop', 'R&B/Soul', 'Reggae', 'Rock']
        and inside them groups the files present in the "leaders_diffusion_trees" and "less_then_4_nodes"
        directories
    """
    leaders_diffusion_trees_directory = "leaders_diffusion_trees/"
    less_then_4_nodes = "less_then_4_nodes/"

    group_directories(leaders_diffusion_trees_directory)
    group_directories(less_then_4_nodes)

def get_top_tracks_and_top_albums_for_artists():
    """
        Function which updates the files present in the ""
    :return:
    """

# create for all filtered_artists an info file and collect their music tags
# collect_tags_list_for_target_artists()
# try to classify all the artists with a main music genre
# simple_classification()

# CHECK MANUALLU IF THE "filtered_artists_with_multiple_main_tags" AND "filtered_artists_without_main_tag"
# ARE EMPTY, OTHERWISE TRY TO COLLECT ARTIST'S SIMILAR MUSICIANS TAGS IN ORDER TO TRY TO SOLVE THE INDECISION
# OR THE MISSING OF VALID MUSIC TAGS
# retrieve_similar_artists_tags()
# retry classification
# simple_classification()

# delete info file for the artists without main tags
# delete_files_of_filtered_artists_without_main_tag()
# check_existance_of_all_artists_with_main_tag_files()

# craete the final file for the Leader detection startegy
# create_final_leadears_DIRECTED_file()

# create file which contains the univocal leaders present in the "final_leadears_DIRECTED" file
# create_univocal_final_leaders_file()

# create file which holds only the artists for which I found a leader
# create_filtered_artists_with_leaders()

# organize " "leaders_diffusion_trees/"" and "less_then_4_nodes" directories according to the tags of the actions they belong
# group_directories_according_to_music_tags()


# create final diffusion trees files ("final_leaders_DIRECTED file" + music genre))
# convert_final_leaders_DIRECTED_file_to_csv()
