#!/usr/bin/env python3

import time
import urllib.request
import urllib.request
import urllib.error
import urllib.parse
import json
import tqdm
import os
import requests

__author__ = 'Giulio Rossetti', 'Alexandra Bradan'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com", "alexandrabradan@gmail.com"


# noinspection PyBroadException
class LastfmCollector(object):

    def __init__(self, lastfm_api):
        """
        Constructor
        :param lastfm_api: Last.fm api key
        """
        self.lfm_apikey = lastfm_api

    def delete_content_of_a_file(self, file_path):
        """
            Delete the content of the file specified in the path passed by argument
            :param file_path: file's path
        """
        try:
            # delete file's content
            open(file_path, 'w').close()
        except FileNotFoundError:
            # don't nothing if file doesn't exist
            pass

    def delete_file(self, file_path):
        """
            Delete the file specified in the path passed by argument
            :param file_path: file's path
        """
        try:
            open(file_path, encoding='utf-8').close()
            # remove file if it exists
            os.remove(file_path)
        except FileNotFoundError:
            # don't nothing if file doesn't exist
            pass

    def clean_directories(self):
        """
            Delete all content of the files present in the "data/artists_info.csv" and in the "data/users_info.csv"
            directories
        """

        # get all the files in the "data/artists_info" directory and delete their content
        list_of_files = os.listdir("data/artists_info")
        for file in list_of_files:
            # delete content of the file
            file_path = "data/artists_info/" + file
            self.delete_content_of_a_file(file_path)

        # get all the files in the "data/users_info" directory and delete their content
        list_of_files = os.listdir("data/users_info")
        for file in list_of_files:
            # delete content of the file
            file_path = "data/users_info/" + file
            self.delete_content_of_a_file(file_path)

    def delete_files_directories(self):
        """
            Delete all the files present in the "data/artists_info.csv" and in the "data/users_info.csv" directories
        """

        # get all the files in the "data/artists_info" directory and delete their content
        list_of_files = os.listdir("data/artists_info")
        for file in list_of_files:
            # delete content of the file
            file_path = "data/artists_info/" + file
            self.delete_file(file_path)

        # get all the files in the "data/users_info" directory and delete their content
        list_of_files = os.listdir("data/users_info")
        for file in list_of_files:
            # delete content of the file
            file_path = "data/users_info/" + file
            self.delete_file(file_path)

    def touch(self, path):
        """
        Create a new file in the specified path
        :param path: path where to create the new file
        """
        with open(path, 'a', encoding='utf-8') as f:
            os.utime(path, None)
            f.close()

    def check_if_file_exist(self, path):
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

    def check_if_file_exist_and_create_it(self, path):
        """
           Check if a file exist and if not craete it
           :param path: file's path to check existance
           :return: 0 if the file already exist
                    1 if the file didn't exist and I just create it

        """
        try:
            # check if the user's info file exist
            open(path).close()
            return 0
        except FileNotFoundError:
            # if the FileNotFoundError is raised it means that the file doesn't exist => create it
            self.touch(path)
            return 1

    def delete_line_from_file(self, element_to_delete, file_path):
        """
           Function which opens the file passed as argument and get all the lines from it.
           Then reopen the file in write mode and writes your lines back, except for the line you want to delete
           :param element_to_delete: element you want to delete from the file
           :param file_path: file from which you want to delete a line
        """

        with open(file_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
        with open(file_path, "a", encoding='utf-8') as f:
            for line in lines:
                # need to strip("\n") the newline character in the comparison because element doesn't end with a newline
                if line.strip("\n") != element_to_delete:
                    f.write(line)

    def get_last_line_from_file(self, file_path):
        """
            Function which opens the file passed as argument and get its last line.
            :param file_path: file from which you want to extract last line
         """
        with open(file_path, 'r') as f:
            lines = f.read().splitlines()  # return an array with all the lines of the file
            if len(lines) == 0:
                return ""
            else:
                last_line = lines[-1]
                return last_line

    def write_output_to_file(self, file_path, output):
        """
           Try write output data to the file passed by argument
           :param file_path: file in which to write data
           :param output: data to write to file
        """

    def fetch_HTTP_response(self, url, max_num_attempts_read_url):
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
                self.fetch_HTTP_response(url, max_num_attempts_read_url)
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
                self.fetch_HTTP_response(url, max_num_attempts_read_url)
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
                self.fetch_HTTP_response(url, max_num_attempts_read_url)
            else:
                print()
                print(e)
                print("Problem caused by url: " + url)
                return {}  # failure, return empty json response

    def get_user_info(self, username):
        """
            Get user's details
            :param username: lastfm username
        """

        user_info_file_path = "data/users_info/" + username + "_info.json"

        # ask user's info to the API
        url = "http://ws.audioscrobbler.com/2.0/?method=user.getinfo&user=%s&api_key=%s&format=json" % \
              (username, self.lfm_apikey)

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        if data != {} and data is not None:

            if len(data) > 0 and 'user' in data:

                try:
                    user_info = {
                        'user_id': data['user']['name'],
                        'gender': data['user']['gender'],
                        'age': int(data['user']['age']),
                        'country': data['user']['country'],
                        'playcount': data['user']['playcount'],
                        'registered_on': data['user']['registered']['unixtime'],
                    }

                    # check if file exist, otherwise create it
                    result = self.check_if_file_exist_and_create_it(user_info_file_path)

                    if result == 1:  # file didn't exist
                        # write user's info to file
                        with open(user_info_file_path, 'w', encoding='utf-8') as outfile:
                            json.dump(user_info, outfile, indent=4)
                            outfile.close()
                            return 0  # user's info file correctly created
                    elif result == 0:  # result == 0: file already exist and user's info already there
                        return 0
                except KeyError as e:
                    print()
                    print("KeyError " + str(e))
                    print(url)
                    print("In get_user_info: " + "username= " + username)
                    # delete user's file, because it is empty
                    self.delete_file(user_info_file_path)
                    return -1
                except Exception as e:
                    print()
                    print(e)
                    print(url)
                    print("In get_user_info: " + "username= " + username)
                    # delete user's file, because it is empty
                    self.delete_file(user_info_file_path)
                    return -1
            else:
                #  user's file hasn't been created
                # TO DO : remove user from user file
                # return -1 to signal to the main.py that the user must be removed also from it's user_list array
                return -1

        else:
            #  user's file hasn't been created
            # TO DO : remove user from user file
            # return -1 to signal to the main.py that the user must be removed also from it's user_list array
            return -1

    def get_artist_info(self, artist, artist_mbid):
        """
            Collect artist's details
            :param artist: name (some artist don't have an musicbrainz id)
            :param artist_mbid: artist musicbrainz id
        """
        # check if artist is present or is "<Unknown>" (don't collect data in this case)
        if artist == "<Unknown>":
            return

        # remove the "feat" artists from the artist's name, if present
        delim = "feat"
        if delim in artist:
            art = artist.partition(delim)[0]
            art = art.replace("(", "")  # remove remaining [ of feat., if present
            art = art.replace("[", "")  # remove remaining ( of feat., if present
            art = art.rstrip()  # remove all the ending white spaces
        else:
            art = artist

        char_not_supported_by_file_name = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']

        tmp_artist_name_file = art
        for c in char_not_supported_by_file_name:
            # if the special character "c" is present in the file's name replace it with an underscore
            tmp_artist_name_file = tmp_artist_name_file.replace(c, '_')

        artist_info_file_path = "data/artists_info/" + tmp_artist_name_file + "_info.json"

        # check if file exist, otherwise create it
        result = self.check_if_file_exist_and_create_it(artist_info_file_path)

        if result == 1:  # file didn't exist and I've just create it
            # ask artist's info to the API
            query = "&artist=%s" % (
                urllib.parse.quote_plus(art))  # convert whitespace in + and utf-8 '\x..' char in '%'
            url = "http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&api_key=%s&format=json" % self.lfm_apikey
            url = url + query

            max_num_attempts_read_url = 3
            data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

            if data != {} and data is not None:

                if len(data) > 0 and 'artist' in data:
                    try:
                        artist_profile = {
                            "name": data['artist']['name'],
                            "mbid": artist_mbid,
                            "stats": data['artist']['stats'],
                            "similar": [data['artist']['similar']['artist'][x]['name'] for x in
                                        range(0, len(data['artist']['similar']['artist']))],
                            "tags": [data['artist']['tags']['tag'][x]['name'] for x in
                                     range(0, len(data['artist']['tags']['tag']))],
                            "crawled albums": {}
                        }

                        # write user's info to file
                        # if the file doesn't exist, create it and if it already exist, overwrite it
                        with open(artist_info_file_path, 'w', encoding='utf-8') as outfile:
                            json.dump(artist_profile, outfile, indent=4)
                            outfile.close()

                    except KeyError as e:
                        # print("KeyError " + str(e))
                        # print(url)
                        # print("In get_artist_info: " + "artist= " + art)
                        # delete file, because it is empty
                        self.delete_file(artist_info_file_path)
                        return
                    except Exception as e:
                        print(e)
                        print(url)
                        print("In get_artist_info: " + "artist= " + art)
                        # delete file, because it is empty
                        self.delete_file(artist_info_file_path)
                        return
                else:
                    # delete file, because it is empty
                    self.delete_file(artist_info_file_path)
                    return
            else:
                # delete file, because it is empty
                self.delete_file(artist_info_file_path)
                return
        elif result == 0:  # result == 0: file already exist and artist's info are already there
            return

    def get_album_info(self, album, album_mbid, artist, artist_mbid):
        """
             Collect album's details
            :param album: album's name
            :param album_mbid: album's mbid
            :param artist: artist who recorded the album
            :param artist_mbid: artist's mbid
        """
        # check if artist is present or is "<Unknown>" (don't collect data in this case)
        if artist == "<Unknown>":
            return

        # check if file exist, otherwise create it and add to it artist's info
        self.get_artist_info(artist, artist_mbid)

        # remove the "feat" artists from the artist's name, if present
        delim = "feat"
        if delim in artist:
            art = artist.partition(delim)[0]
            art = art.replace("(", "")  # remove remaining [ of feat., if present
            art = art.replace("[", "")  # remove remaining ( of feat., if present
            art = art.rstrip()  # remove all the ending white spaces
        else:
            art = artist

        char_not_supported_by_file_name = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']

        tmp_artist_name_file = art
        for c in char_not_supported_by_file_name:
            # if the special character "c" is present in the file's name replace it with an underscore
            tmp_artist_name_file = tmp_artist_name_file.replace(c, '_')

        artist_info_file_path = "data/artists_info/" + tmp_artist_name_file + "_info.json"

        # check artist's file existance
        if self.check_if_file_exist(artist_info_file_path) == -1:  # file doesn't exist
            return

        artist_info = json.load(open(artist_info_file_path, encoding='utf-8'))

        try:
            value = artist_info["crawled albums"][album]
            return  # album's info are already in the artist's file
        except KeyError:
            # if the KeyError is raised it means that the album isn't present in the artist's file => add it

            # ask album's info to the API
            # convert whitespace in + and utf-8 '\x..' char in '%'
            query1 = "&album=%s" % (urllib.parse.quote_plus(album))
            query2 = "&artist=%s" % (urllib.parse.quote_plus(art))
            url = "http://ws.audioscrobbler.com/2.0/?method=album.getinfo&api_key=%s&format=json" % self.lfm_apikey
            url = url + query1 + query2

            max_num_attempts_read_url = 3
            data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

            if data != {} and data is not None:

                if data == 0:
                    return

                if len(data) > 0:

                    date_of_publication = ""

                    try:
                        date_of_publication = data['album']['wiki']['published']
                    except Exception:  # Wikipedia's info aren't present
                        pass

                    try:
                        album_profile = {
                            "title": data['album']['name'],
                            # "artist": data['album']['artist'],
                            "mbid": album_mbid,
                            "published on": date_of_publication,
                            "playcount": data['album']['playcount'],
                            "tags": [data['album']['tags']['tag'][x]['name'] for x in
                                     range(0, len(data['album']['tags']['tag']))],
                            "listeners": data['album']['listeners'],
                            "tracks": [{
                                'title': data['album']['tracks']['track'][x]['name'],
                                'rank': data['album']['tracks']['track'][x]['@attr']['rank'],
                                'duration': data['album']['tracks']['track'][x]['duration']
                            } for x in range(0, len(data['album']['tracks']['track']))]
                        }

                        # add a (key:value) pair to dictionary: Dictionary_Name[New_Key_Name] = New_Key_Value
                        artist_info["crawled albums"][album] = album_profile

                        # write album's info in the artist's info file
                        with open(artist_info_file_path, 'w', encoding='utf-8') as outfile:
                            json.dump(artist_info, outfile, indent=4)
                            outfile.close()
                    except KeyError as e:
                        # print("KeyError " + str(e))
                        # print(url)
                        # print("In get_album_info: " + "artist= " + art + "album= " + album)
                        return  # not add album to the user's file
                    except Exception as e:
                        # print(e)
                        # print(url)
                        # print("In get_album_info: " + "artist= " + art + "album= " + album)
                        return  # not add album to the user's file
            else:
                return  # not add album to the user's file
        except Exception as e:
            # print(e)
            return

    def get_user_available_charts(self, username, charts_type, start_date, end_date):
        """
        Collect data of weekly charts for a given username
        :param username: lastfm username
        :param charts_type: a tuple containing at least one among ("artist", "album", "track")
        :param start_date: initial date
        :param end_date: final date
        """

        # ask the user's chart to the API
        url = "http://ws.audioscrobbler.com/2.0/?method=user.getweeklychartlist&user=%s&api_key=%s&format=json" % \
              (username, self.lfm_apikey)

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        user_info_file_path = "data/users_info/" + username + "_info.json"

        if data != {} and data is not None:

            if data == 0:
                # TO DO : delete user's file
                # return -1 to signal to the main.py that the user must be removed also from it's user_list array
                self.delete_file(user_info_file_path)
                return -1  # exit

            # try to retrieve the user's info(if user's info file exist it means that the user has already info in it)
            try:
                # read jason file and decode it in order to translate the json object in a dict
                user_info = json.load(open(user_info_file_path, encoding='utf-8'))
            except FileNotFoundError:
                # if the FileNotFoundError is raised it means that the file doesn't exist =>
                # create the user's info file and add info to it
                result = self.get_user_info(username)
                if result == -1:  # can't be able to create and add info to the user's file
                    # return -1 to signal to the main.py that the user must be removed also from it's user_list array
                    return -1  # exit
                else:
                    # read jason file and decode it in order to translate the json object in a dict
                    user_info = json.load(open(user_info_file_path, encoding='utf-8'))

            # keep track of already crawled weeks
            try:
                value = user_info["crawled"]

                # insert in the "crawled_weeks" list, the weeks already examined (present in the user'info file)
                crawled_weeks = {(value["from"], value["to"]): None for key, value in user_info["crawled"].items()}
            except KeyError:
                # if the KeyError is raised it means that the user file don't have the "crawled" key
                # => no crawled_weeks
                crawled_weeks = []

            selected_weeks = []

            # keep track of weeks not yet crawled
            for ch in data['weeklychartlist']['chart']:
                try:
                    fr = ch['from']
                    to = ch['to']

                    if int(fr) >= start_date and int(to) <= end_date and (fr, to) not in crawled_weeks and \
                            int(fr) >= int(user_info['registered_on']):
                        # insert in the "selected_weeks" list, the weeks that have to be examined
                        selected_weeks.append((fr, to))
                except KeyError as e:
                    print()
                    print("KeyError " + str(e))
                    print(url)
                    print("In get_user_available_charts: " + "fr= " + str(start_date) + "to= " + str(end_date))
                    # skip iteration
                    continue
                except Exception as e:
                    print()
                    print(e)
                    print(url)
                    print("In get_user_available_charts: " + "fr= " + str(start_date) + "to= " + str(end_date))
                    # skip iteration
                    continue

            tmp_charts = {}

            """
                tqdm :
                     Decorate an iterable object, returning an iterator which acts exactly
                     like the original iterable, but prints a dynamically updating
                     progressbar every time a value is requested.

                Parameters:
                    ncols => The width of the entire output message. If specified, dynamically resizes 
                             the progressbar to stay within this bound
                    ascii => use unicode (smooth blocks) to fill the meter
                    desc => Prefix for the progressbar
              """
            # get charts for the weeks not already crawled
            for week in tqdm.tqdm(selected_weeks, ncols=100, ascii=True, desc=username + " crawling weeks"):
                tmp_charts[week[0]] = {"from": week[0], "to": week[1]}

            user_info = json.load(open(user_info_file_path, encoding='utf-8'))
            user_info["crawled"] = tmp_charts

            try:
                # write uptaded crawled listened tracks
                with open(user_info_file_path, 'w', encoding='utf-8') as outfile:
                    json.dump(user_info, outfile, indent=4)
                    outfile.close()
            except IOError:
                self.delete_file(user_info_file_path)
                return -1

            if len(selected_weeks) > 0:
                # get tracks, artists or/and albums listened during the selected week
                for week in tqdm.tqdm(selected_weeks, ncols=100, ascii=True,
                                      desc=username + " crawling tracks/artists/albums"):
                    if "track" in charts_type:
                        self.get_user_weekly_track_chart(username, week[0], week[1])
                    if "artist" in charts_type:
                        self.get_user_weekly_artist_chart(username, week[0], week[1])
                    if "album" in charts_type:
                        self.get_user_weekly_album_chart(username, week[0], week[1])

            # ACCESS USER'S CRAWLED WEEKS(charts expressed ad 'from'=star_date and 'to'=end_date)
            # user_info = json.load(open(user_info_file_path))
            # for key, value in user_info["crawled"].items():
            # print(value['from'])
            # print(value['to'])

        else:
            # TO DO : delete user's file
            # return -1 to signal to the main.py that the user must be removed also from it's user_list array
            self.delete_file(user_info_file_path)
            return -1  # exit

    def get_user_weekly_track_chart(self, username, fr, to):
        """
        Collect weekly chart data of listened tracks for a given user
        :param username: lastfm username
        :param fr: crawled week start date
        :param to: crawled week end date
        """

        user_info_file_path = "data/users_info/" + username + "_info.json"

        # ask which tracks the user has listened during the crawled week passed by argument to the API
        url = "http://ws.audioscrobbler.com/2.0/?method=user.getweeklytrackchart&user=%s&api_key=%s&from=%s&to=%s" \
              "&format=json" % \
              (username, self.lfm_apikey, int(fr), int(to))

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        tmp_listened_tracks = {}

        if data != {} and data is not None:

            # add tracks to the crawled week only if the user has listened to
            # something in this period
            if len(data['weeklytrackchart']['track']) > 0:
                for tr in data['weeklytrackchart']['track']:
                    try:
                        track_name = tr['name']
                        record = {
                            'track_rank': tr['@attr']['rank'],
                            'track': tr['name'],
                            'artist': tr['artist']['#text'],
                            'track_mbid': tr['mbid'],
                            'artist_mbid': tr['artist']['mbid'],
                            'playcount': tr['playcount']
                        }

                        """
                            N.B. In the user file can be found:
                             1) user's info, 
                             2) his/her crawled weeks of listenings (the time range of the study is found in the 
                                "main.py" file of the program)
                             3) which tracks, artists and/or albums he/she has listened during this period.

                             Every crawled week (7 days stored as unixtime range of [start week, end week])stores 
                             the following data:
                             a) listened tracks => dict composed by:
                                - key = track's name
                                - value = json object with info about the track and the artist who performed it
                             b) listened artists => dict composed by:
                                  - key = artists's name
                                  - value = json object with info about artist's mbdi and his/her playcounts
                             c) listened albums => dict composed by:
                                  - key = albums's name
                                  - value = json object with info about album's mbdi, the artist who performed it
                                            and its playcounts

                             Since a song's name, an artist's name and an album's name are NOT univocal, when one 
                             or more tracks/artists/songs happens to have the same dict's key, we craete a list of
                             colliding elements on the same key.

                             REMEBER TO CHECK THIS COLLISION WHEN YOU RETRIEVE DATA FROM THE USER'S FILE !
                        """
                        try:
                            tmp_listened_tracks[track_name] = [tmp_listened_tracks[track_name]]
                            tmp_listened_tracks[track_name].append(record)

                        except KeyError:
                            # if the KeyError is raise it means that there isn't the key => I can my new key
                            # without problems. Add a (key:value) pair to dictionary:
                            # Dictionary_Name[New_Key_Name] = New_Key_Value
                            tmp_listened_tracks[track_name] = record
                    except KeyError as e:
                        print()
                        print("KeyError " + str(e))
                        print(url)
                        print("In get_user_weekly_track_chart: " + "fr= " + str(fr) + "to= " + str(to))
                        # skip iteration
                        continue
                    except Exception as e:
                        print()
                        print(e)
                        print(url)
                        print("In get_user_weekly_track_chart: " + "fr= " + str(fr) + "to= " + str(to))
                        # skip iteration
                        continue

        user_info = json.load(open(user_info_file_path, encoding='utf-8'))
        user_info["crawled"][str(fr)]["listened tracks"] = tmp_listened_tracks

        # write uptaded crawled listened tracks
        with open(user_info_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(user_info, outfile, indent=4)
            outfile.close()

    def get_user_weekly_artist_chart(self, username, fr, to):
        """
           Collect weekly chart data of listened artists for a given user
           :param username: lastfm username
           :param fr: start date
           :param to: end date
        """

        user_info_file_path = "data/users_info/" + username + "_info.json"

        # ask which artists the user has listened during the crawled week passed by argument to the API
        url = "http://ws.audioscrobbler.com/2.0/?method=user.getweeklyartistchart&user=%s&api_key=%s&from=%s&to=%s" \
              "&format=json" % \
              (username, self.lfm_apikey, int(fr), int(to))

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        tmp_listened_artists = {}

        if data != {} and data is not None:

            # add tracks to the crawled week only if the user has listened to
            # some artist in this period
            if len(data['weeklyartistchart']['artist']) > 0:
                for tr in data['weeklyartistchart']['artist']:
                    try:
                        artist_name = tr['name']
                        record = {
                            'artist_rank': tr['@attr']['rank'],
                            'artist': tr['name'],
                            'artist_mbid': tr['mbid'],
                            'playcount': tr['playcount']
                        }

                        """
                            N.B. In the user file can be found:
                             1) user's info, 
                             2) his/her crawled weeks of listenings (the time range of the study is found in the 
                                "main.py" file of the program)
                             3) which tracks, artists and/or albums he/she has listened during this period.

                             Every crawled week (7 days stored as unixtime range of [start week, end week])stores 
                             the following data:
                             a) listened tracks => dict composed by:
                                - key = track's name
                                - value = json object with info about the track and the artist who performed it
                             b) listened artists => dict composed by:
                                  - key = artists's name
                                  - value = json object with info about artist's mbdi and his/her playcounts
                             c) listened albums => dict composed by:
                                  - key = albums's name
                                  - value = json object with info about album's mbdi, the artist who performed it
                                            and its playcounts

                             Since a song's name, an artist's name and an album's name are NOT univocal, when one 
                             or more tracks/artists/songs happens to have the same dict's key, we craete a list of
                             colliding elements on the same key.

                             REMEBER TO CHECK THIS COLLISION WHEN YOU RETRIEVE DATA FROM THE USER'S FILE !
                        """
                        try:
                            tmp_listened_artists[artist_name] = [tmp_listened_artists[artist_name]]
                            tmp_listened_artists[artist_name].append(record)

                        except KeyError:
                            # if the KeyError is raise it means that there isn't the key => I can my new key
                            # without problems. Add a (key:value) pair to dictionary:
                            # Dictionary_Name[New_Key_Name] = New_Key_Value
                            tmp_listened_artists[artist_name] = record
                    except KeyError as e:
                        print()
                        print("KeyError " + str(e))
                        print(url)
                        print("In get_user_weekly_artist_chart: " + "fr= " + str(fr) + "to= " + str(to))
                        # skip iteration
                        continue
                    except Exception as e:
                        print()
                        print(e)
                        print(url)
                        print("In get_user_weekly_artist_chart: " + "fr= " + str(fr) + "to= " + str(to))
                        # skip iteration
                        continue

        user_info = json.load(open(user_info_file_path, encoding='utf-8'))
        user_info["crawled"][str(fr)]["listened artists"] = tmp_listened_artists

        # write uptaded crawled listened artists
        with open(user_info_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(user_info, outfile, indent=4)
            outfile.close()

    def get_user_weekly_album_chart(self, username, fr, to):
        """
           Collect weekly chart data of listened albums for a given user
           :param username: lastfm username
           :param fr: start date
           :param to: end date
        """

        user_info_file_path = "data/users_info/" + username + "_info.json"

        # ask which albums the user has listened during the crawled week passed by argument to the API
        url = "http://ws.audioscrobbler.com/2.0/?method=user.getweeklyalbumchart&user=%s&api_key=%s&from=%s&to=%s" \
              "&format=json" % \
              (username, self.lfm_apikey, fr, to)

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        tmp_listened_albums = {}  # empty dict

        if data != {} and data is not None:

            # add tracks to the crawled week only if the user has listened to
            # some album in this period
            if len(data['weeklyalbumchart']['album']) > 0:
                for tr in data['weeklyalbumchart']['album']:
                    try:
                        album_name = tr['name']
                        record = {
                            'album_rank': tr['@attr']['rank'],
                            'album': tr['name'],
                            'album_mbid': tr['mbid'],
                            'artist': tr['artist']['#text'],
                            'artist_mbid': tr['mbid'],
                            'playcount': tr['playcount']
                        }

                        """
                            N.B. In the user file can be found:
                             1) user's info, 
                             2) his/her crawled weeks of listenings (the time range of the study is found in the 
                                "main.py" file of the program)
                             3) which tracks, artists and/or albums he/she has listened during this period.

                             Every crawled week (7 days stored as unixtime range of [start week, end week])stores 
                             the following data:
                             a) listened tracks => dict composed by:
                                - key = track's name
                                - value = json object with info about the track and the artist who performed it
                             b) listened artists => dict composed by:
                                  - key = artists's name
                                  - value = json object with info about artist's mbdi and his/her playcounts
                             c) listened albums => dict composed by:
                                  - key = albums's name
                                  - value = json object with info about album's mbdi, the artist who performed it
                                            and its playcounts

                             Since a song's name, an artist's name and an album's name are NOT univocal, when one 
                             or more tracks/artists/songs happens to have the same dict's key, we craete a list of
                             colliding elements on the same key.

                             REMEBER TO CHECK THIS COLLISION WHEN YOU RETRIEVE DATA FROM THE USER'S FILE !
                        """
                        try:
                            if isinstance(tmp_listened_albums[album_name], list):
                                tmp_listened_albums[album_name].append(record)
                            else:
                                tmp_listened_albums[album_name] = [tmp_listened_albums[album_name]]
                        except KeyError:
                            # if the KeyError is raise it means that there isn't the key => I can my new key
                            # without problems. Add a (key:value) pair to dictionary:
                            # Dictionary_Name[New_Key_Name] = New_Key_Value
                            tmp_listened_albums[album_name] = record
                    except KeyError as e:
                        print()
                        print("KeyError " + str(e))
                        print(url)
                        print("In get_user_weekly_album_chart: " + "fr= " + str(fr) + "to= " + str(to))
                        # skip iteration
                        continue
                    except Exception as e:
                        print()
                        print(e)
                        print(url)
                        print("In get_user_weekly_album_chart: " + "fr= " + str(fr) + "to= " + str(to))
                        # skip iteration
                        continue

        user_info = json.load(open(user_info_file_path, encoding='utf-8'))
        user_info["crawled"][str(fr)]["listened albums"] = tmp_listened_albums

        # write uptaded crawled listened albums
        with open(user_info_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(user_info, outfile, indent=4)
            outfile.close()

    def collect_artist_info_from_weekly_charts(self, username):
        """
        Collect details on artists listeded by a given user
        :param username: lastfm username
        """
        user_info_file_path = "data/users_info/" + username + "_info.json"

        # verify existance of user's file info
        res = self.check_if_file_exist(user_info_file_path)
        if res == -1:
            return
        else:
            user_info = json.load(open(user_info_file_path, encoding='utf-8'))

        # iterate over the crawled weeks
        for key, crawled_week in tqdm.tqdm(user_info["crawled"].items(), ncols=100, ascii=True,
                                           desc=username + " collect artists' info"):

            try:
                # iterate over the tracks of a given crawled week to extract artists
                for kkey, tracks_chart in crawled_week["listened tracks"].items():
                    if isinstance(tracks_chart, list):
                        # iterate on the colliding artists
                        for index in range(len(tracks_chart)):
                            artist_name = tracks_chart[int(index)]["artist"]
                            artist_mbid = tracks_chart[int(index)]["artist_mbid"]
                            # get artist's info
                            self.get_artist_info(artist_name, artist_mbid)
                    else:
                        artist_name = tracks_chart['artist']
                        artist_mbid = tracks_chart['artist_mbid']
                        # get artist's info
                        self.get_artist_info(artist_name, artist_mbid)
            except KeyError as e:
                print()
                print("KeyError " + str(e))
                print("In collect_artist_info_from_weekly_charts")
                continue
            except Exception as e:
                print()
                print(e)
                print("In collect_artist_info_from_weekly_charts")
                continue

    def collect_album_info_from_weekly_charts(self, username):
        """
        Collect details on albums listeded by a given user
        :param username: lastfm username
        """
        user_info_file_path = "data/users_info/" + username + "_info.json"

        # verify existance of user's file info
        res = self.check_if_file_exist(user_info_file_path)
        if res == -1:
            return
        else:
            user_info = json.load(open(user_info_file_path, encoding='utf-8'))

        # iterate over the crawled weeks
        for key, crawled_week in tqdm.tqdm(user_info["crawled"].items(), ncols=100, ascii=True,
                                           desc=username + " collect albums' info"):
            try:
                # iterate over the albums of a given crawled week to extract albums
                for kkey, albums_chart in crawled_week["listened albums"].items():
                    if isinstance(albums_chart, list):
                        # iterate on the colliding albums
                        for index in range(len(albums_chart)):
                            album_name = albums_chart[int(index)]["album"]
                            album_mbid = albums_chart[int(index)]["album_mbid"]
                            artist_name = albums_chart[int(index)]["artist"]
                            artist_mbid = albums_chart[int(index)]["artist_mbid"]
                            # get album's info
                            self.get_album_info(album_name, album_mbid, artist_name, artist_mbid)
                    else:
                        album_name = albums_chart['album']
                        album_mbid = albums_chart['album_mbid']
                        artist_name = albums_chart['artist']
                        artist_mbid = albums_chart['artist_mbid']
                        # get album's info
                        self.get_album_info(album_name, album_mbid, artist_name, artist_mbid)
            except KeyError as e:
                print()
                print("KeyError " + str(e))
                print("In collect_album_info_from_weekly_charts")
                continue
            except Exception as e:
                print()
                print(e)
                print("In collect_album_info_from_weekly_charts")
                continue

    def get_network(self, username):
        """
        Collect data on user's friends
        :param username: lastfm usename
        """

        user_info_file_path = "data/users_info/" + username + "_info.json"

        tmp_friends = {}

        url = "http://ws.audioscrobbler.com/2.0/?method=user.getfriends&user=%s&api_key=%s&limit=500" \
              "&format=json" % \
              (username, self.lfm_apikey)

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        if data != {} and data is not None:

            if len(data) > 0:
                for k in tqdm.tqdm(data['friends']['user'], ncols=100, ascii=True,
                                   desc=username + " crawling friends "):

                    details = {'country': None, 'registered_on': None}
                    try:
                        details['country'] = k['country']
                        details['registered_on'] = k['registered']['unixtime']

                        tmp_friends[k['name']] = details
                    except KeyError as e:
                        print()
                        print("KeyError " + str(e))
                        print(url)
                        print("In get_network")
                        continue
                    except Exception as e:
                        print()
                        print(e)
                        print(url)
                        print("In get_network")
                        continue

        user_info = json.load(open(user_info_file_path, encoding='utf-8'))
        user_info["friends"] = tmp_friends

        try:
            # write user's friends
            with open(user_info_file_path, 'w', encoding='utf-8') as outfile:
                json.dump(user_info, outfile, indent=4)
                outfile.close()
            return 0
        except IOError:
            self.delete_file(user_info_file_path)
            return -1
