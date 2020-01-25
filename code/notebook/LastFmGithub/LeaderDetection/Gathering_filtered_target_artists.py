import gzip
import os
import sys
import re
import json
import urllib
import time

import requests

__author__ = 'Giulio Rossetti', 'Alexandra Bradan'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com", "alexandrabradan@gmail.com"



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


def get_artist_tags(artist):
    """
        Function which retrieves the tags associated to the artist passed by argument
        using the Lastf.fm  API <<artist.getTopTags>>
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
                # I check if the tag is assigned at least 100 times to the artist
                if tag['count'] >= 100:
                    t = str(tag['name']).lower()
                    pair = (t, tag['count'])
                    lastfm_tags_list.append(pair)

            return lastfm_tags_list

        except Exception:
            return lastfm_tags_list

    return lastfm_tags_list


def get_artist_listeners(artist):
    """
        Function which retrieves the listeners associated to the artist passed by argument
        using the Lastf.fm  API <<artist.getInfo>>
        :param artist: artist for which to retrieve the listeners associated
        :return an integer representing the number of listeners of the artist
    """

    lfm_apikey = '3878e8d3f604944d12eaef5f34ada2d1'

    query = "&artist=%s" % (
        urllib.parse.quote_plus(artist))  # convert whitespace in + and utf-8 '\x..' char in '%'
    url = "http://ws.audioscrobbler.com/2.0/?method=artist.getInfo&api_key=%s&format=json" % lfm_apikey
    url = url + query

    max_num_attempts_read_url = 3
    data = fetch_HTTP_response(url, max_num_attempts_read_url)

    if data != {} and data is not None:

        try:
            listeners = int(data['artist']['stats']['listeners'])
            return listeners
        except Exception:
            return -1
    return -1


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
    check = check_if_file_exist("target_artists_NEW")

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
            lines_counter = 0

            with open(file_path, 'r') as infile:
                for line in infile:

                    lines_counter += 1

                    line_array = line.split("::")

                    week_encoding = line_array[0]
                    user_encoding = line_array[1]
                    artist_encoding = line_array[2]
                    playcount = line_array[3].strip()  # remove "\n"

                    # I have to check if the current listening week is >= 14/01/18 12:00 PM
                    # (and eventually update the absolute first artist listening made by some user).
                    # If I found out that the current listening week is <  14/01/18 12:00 PM
                    # I have to delete the artist from the dict (some user has listened to him before our observation
                    # time window)
                    if int(week_encoding) < 26:
                        discard_artist = True
                        print("Artist <<" + str(artist_encoding) + ">> is a excluded")
                        break  # discard artist and iterate over the next artist's file

                    if int(week_encoding) < int(tmp_min_first_week[0]):   # week_encoding >= 26, update minimum
                            tmp_min_first_week = [week_encoding, user_encoding, artist_encoding, playcount]

                if discard_artist is False and lines_counter > 1:
                    print("Artist |" + str(tmp_min_first_week[2]) + "| is a TARGET")

                    # write target artist on string (I found his absolute first week of listenings and it is
                    # >= 14/01/18)
                    str_to_write = str(tmp_min_first_week[0]) + "::" + str(tmp_min_first_week[1]) + "::" + \
                                   str(tmp_min_first_week[2]) + "::" + str(tmp_min_first_week[3])
                    outfile.write(str_to_write)
                    outfile.write("\n")
                else:
                    print("Artist is a excluded")


def filter_target_artists():
    # list of artist encountered
    encountered_artists = []
    # retrieved_tags_list[i] = True if LastFm return a tag list for encountered_artists[i], False otherwise
    retrieved_tags_list = []

    # store artists' tags list
    tmp_artist_tag_id = {}
    # stor artist with less then 100 listeners
    less_the_100_listeners = []

    tmp_lastfm_list = []

    count = 0

    f = open("target_artists", "r")
    new_f = open("filtered_target_artists", "a")
    for line in f:
        line_array = line.split("::")
        artist_encoding = line_array[2]

        count += 1

        if artist_encoding not in less_the_100_listeners:

            # get artist from encoding
            artist = get_artist_from_encoding(artist_encoding)

            if artist == "":
                continue  # next artist to analyze

            # check if the artist has at least 100 listeners
            artist_listeners = get_artist_listeners(artist)

            print(str(artist) + " listeners = " + str(artist_listeners) + " COUNT = " + str(count))

            if int(artist_listeners) < 100:
                tmp_lastfm_list = []  # re-initialization
                less_the_100_listeners.append(artist_encoding)
                continue  # next iteration

            if artist_encoding not in encountered_artists:
                encountered_artists.append(artist_encoding)

                # get artist's associated tags list
                lastfm_tags_list = get_artist_tags(artist)

                tmp_lastfm_list = lastfm_tags_list  # keep tmp track

                if len(lastfm_tags_list) == 0:
                    retrieved_tags_list.append(False)
                else:
                    retrieved_tags_list.append(True)

                    tmp_artist_tag_id[str(artist_encoding)] = lastfm_tags_list

                    print("artist " + str(artist) + " tags = " + str(lastfm_tags_list))

            if len(tmp_lastfm_list) > 0:
                # write line on filtered leader_DIRECTED file
                new_f.write(line)
    f.close()
    new_f.close()

# gather_unseen_artists()
# filter_target_artists()

