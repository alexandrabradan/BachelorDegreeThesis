import time
import urllib.request
import urllib.request
import urllib.error
import urllib.parse
import json
import tqdm
import os
import requests
import gzip
import shutil

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

    def compess_existing_file(self, path):
        """
           Compress the file passed by argument (.gz)
           :param path: path to compress
        """
        with open(path, 'rb') as f_in:
            path_compress = path + ".gz"
            with gzip.open(path_compress, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                f_out.close()
            f_in.close()

        # delete not compressed file
        self.delete_file(path)

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

    def check_if_file_is_empty(self, path):
        """
       Check if file is empty in order to ad an empty dict to it
       :param path: file's path to check
       :return True if file is empty
               False if file is not empty

       """
        exist = self.check_if_file_exist(path)
        if exist == -1:  # compress file doesn't exist
            return False
        else:  # compress file exist => decompress it => read its content => return content == "" (empty)
            with gzip.open(path, "rt") as infile:
                file_content = infile.read()
                infile.close()
                return file_content == ""

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

    def create_directory(self, path):
        """
       Create a new directory in the specified path
       :param path: path where to create the new directory
       """
        os.makedirs(path)

    def check_if_directory_exist(self, path):
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

    def check_if_directory_exist_and_create_it(self, path):
        """
          Check if a directory exist and if not craete it
          :param path: directory's path to check existance
          :return: 0 if the directory already exist
                   1 if the directory didn't exist and I just create it

       """
        exist = self.check_if_directory_exist(path)

        if exist == -1:
            self.create_directory(path)
            return 1
        else:
            return 0

    def empty_directories_files(self, path):
        """
            Empty all content of the files present in the directory passed by argument
            :param path directory to make empty
        """
        # check if directory exist
        exist = self.check_if_directory_exist(path)

        if exist == 0:

            # get all the files in the directory
            list_of_files = os.listdir(path)
            for file in list_of_files:
                # delete content of the file

                # check if directory's path ends with "/", otherwise attach "/" to it
                check = path.endswith("/")

                if check is False:
                    path = path + "/"

                file_path = path + file
                self.delete_content_of_a_file(file_path)


    def delete_directories_files(self, path):
        """
            Delete all the files present in the directory passed by argument
            :param path directory's path
        """
        # check if directory exist
        exist = self.check_if_directory_exist(path)

        if exist == 0:

            # get all the files in the directory
            list_of_files = os.listdir(path)
            for file in list_of_files:
                # delete content of the file

                # check if directory's path ends with "/", otherwise attach "/" to it
                check = path.endswith("/")

                if check is False:
                    path = path + "/"

                file_path = path + file
                self.delete_file(file_path)

    def delete_directory(self, path):
        """
           Function which delete the directory passed by argument and the files present in it.
           :param path directory's path to delete
        """
        shutil.rmtree(path)

    def clean_artist_name(self, artist):
        """
           Function which cleans an artist name from the following words:
           1. feat
           2. Feat
           3. ft.
           4. FT.
           :param artist: artist name to clean up
           :return artist's name cleaned up
        """

        artist_name = ""

        # delete the "featuring" of the artist, if present
        delim = "feat"
        delim2 = "Feat"
        delim3 = "ft."
        delim4 = "Ft."
        if delim in artist:
            artist_name = artist.partition(delim)[0]
            artist_name = artist_name.replace("(",
                                              "")  # remove remaining [ of feat., if present
            artist_name = artist_name.replace("[",
                                              "")  # remove remaining ( of feat., if present
            artist_name = artist_name.rstrip()  # remove all the ending white spaces
        elif delim2 in artist:
            artist_name = artist.partition(delim2)[0]
            artist_name = artist_name.replace("(",
                                              "")  # remove remaining [ of feat., if present
            artist_name = artist_name.replace("[",
                                              "")  # remove remaining ( of feat., if present
            artist_name = artist_name.rstrip()  # remove all the ending white spaces
        elif delim3 in artist:
            artist_name = artist.partition(delim3)[0]
            artist_name = artist_name.replace("(",
                                              "")  # remove remaining [ of feat., if present
            artist_name = artist_name.replace("[",
                                              "")  # remove remaining ( of feat., if present
            artist_name = artist_name.rstrip()  # remove all the ending white spaces
        elif delim4 in artist:
            artist_name = artist.partition(delim4)[0]
            artist_name = artist_name.replace("(",
                                              "")  # remove remaining [ of feat., if present
            artist_name = artist_name.replace("[",
                                              "")  # remove remaining ( of feat., if present
            artist_name = artist_name.rstrip()  # remove all the ending white spaces
        else:
            artist_name = artist  # artist's name already cleaned

        return artist_name

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

        user_info_file_path = "data/users_info/" + username + "_info.json.gz"

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
                        with gzip.open(user_info_file_path, 'wt', encoding='utf-8') as outfile:
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
            :return 0 if artist's info file was succesfully created and populated
                    -1 otherwise
        """
        # check if artist is present or is "<Unknown>" (don't collect data in this case)
        if artist == "<Unknown>":
            return

        # delete the "featuring" of the artist, if present
        art = self.clean_artist_name(artist)

        char_not_supported_by_file_name = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']

        tmp_artist_name_file = art
        for c in char_not_supported_by_file_name:
            # if the special character "c" is present in the file's name replace it with an underscore
            tmp_artist_name_file = tmp_artist_name_file.replace(c, '_')

        artist_info_file_path = "data/artists_info/" + tmp_artist_name_file + "_artist_info.json.gz"

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
                        with gzip.open(artist_info_file_path, 'wt', encoding='utf-8') as outfile:
                            json.dump(artist_profile, outfile, indent=4)
                            outfile.close()

                    except KeyError as e:
                        # print("KeyError " + str(e))
                        # print(url)
                        # print("In get_artist_info: " + "artist= " + art)
                        # delete file, because it is empty
                        self.delete_file(artist_info_file_path)
                        return -1
                    except Exception as e:
                        print(e)
                        print(url)
                        print("In get_artist_info: " + "artist= " + art)
                        # delete file, because it is empty
                        self.delete_file(artist_info_file_path)
                        return -1
                else:
                    # delete file, because it is empty
                    self.delete_file(artist_info_file_path)
                    return -1
            else:
                # delete file, because it is empty
                self.delete_file(artist_info_file_path)
                return -1
        elif result == 0:  # result == 0: file already exist and artist's info are already there
            return 0

    def return_artist_info(self, artist):
        """
            Return artist's info in dict format
            :param artist: artist's name for which we have to retrieve info
        """

        # ask artist's info to the API
        query = "&artist=%s" % (
            urllib.parse.quote_plus(artist))  # convert whitespace in + and utf-8 '\x..' char in '%'
        url = "http://ws.audioscrobbler.com/2.0/?method=artist.getinfo&api_key=%s&format=json" % self.lfm_apikey
        url = url + query

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        if data != {} and data is not None:

            if len(data) > 0 and 'artist' in data:
                try:
                    artist_profile = {
                        "name": data['artist']['name'],
                        "mbid": data['mbid'],
                        "stats": data['artist']['stats'],
                        "similar": [data['artist']['similar']['artist'][x]['name'] for x in
                                    range(0, len(data['artist']['similar']['artist']))],
                        "tags": [data['artist']['tags']['tag'][x]['name'] for x in
                                 range(0, len(data['artist']['tags']['tag']))],
                        "crawled albums": {}
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

        artist_info_file_path = "data/artists_info/" + tmp_artist_name_file + "_info.json.gz"

        # check artist's file existance
        if self.check_if_file_exist(artist_info_file_path) == -1:  # file doesn't exist
            return

        with gzip.open(artist_info_file_path, 'rt', encoding='utf-8') as infile:
            artist_info = json.load(infile)
            infile.close()

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
                        with gzip.open(artist_info_file_path, 'wt', encoding='utf-8') as outfile:
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

        user_info_file_path = "data/users_info/" + username + "_info.json.gz"
        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

        if data != {} and data is not None:

            if data == 0:
                # TO DO : delete user's file
                # return -1 to signal to the main.py that the user must be removed also from it's user_list array
                self.delete_file(user_info_file_path)
                return -1  # exit

            # check if user's info file exist
            exist = self.check_if_file_exist(user_info_file_path)

            if exist == -1:
                # create the user's info file and add info to it
                result = self.get_user_info(username)
                if result == -1:  # user info file can't be created
                    return -1

            # user file exist/created
            # check if user's listenings file exist and if not create it
            result = self.check_if_file_exist_and_create_it(user_listenings_file_path)
            if result == 1:  # I've just create user's listenings file and I have to put 'user_id' in it
                try:
                    # get user's registration date
                    with gzip.open(user_info_file_path, 'rt', encoding='utf-8') as infile:
                        d = json.load(infile)
                        country = d['country']
                        registration_date = d['registered_on']
                        infile.close()

                    # write user's username, country and registration date on the user's listenings file
                    with gzip.open(user_listenings_file_path, 'wt', encoding='utf-8') as outfile:
                        user_info = {'user_id': username,
                                     'country': country,
                                     'registered_on': registration_date
                                     }
                        json.dump(user_info, outfile, indent=4)
                        outfile.close()
                except FileNotFoundError:
                    self.delete_file(user_info_file_path)  # delete user's info file
                    return -1
                except Exception:
                    self.delete_file(user_info_file_path)  # delete user's info file
                    self.delete_file(user_listenings_file_path)  # delete user's listenings file
                    return -1

            # try to retrieve the user's artists listened
            try:
                # read jason file and decode it in order to translate the json object in a dict
                with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
                    user_info = json.load(infile)
                    infile.close()
            except FileNotFoundError:
                # if the FileNotFoundError is raised it means we have problems
                self.delete_file(user_info_file_path) # delete user's info file
                self.delete_file(user_listenings_file_path) # delete user's listenings file
                return -1

            # keep track of already crawled weeks
            try:
                value = user_info["crawled"]

                # insert in the "crawled_weeks" list, the weeks already examined (present in the user'info file)
                crawled_weeks = {(value["from"], value["to"]): None for key, value in user_info["crawled"].items()}
            except KeyError:
                # if the KeyError is raised it means that the user's listening file don't have the "crawled" key
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

            if len(selected_weeks) == 0:
                return

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

            user_info["crawled"] = tmp_charts

            try:
                # write uptaded crawled listened tracks
                with gzip.open(user_listenings_file_path, 'wt', encoding='utf-8') as outfile:
                    json.dump(user_info, outfile, indent=4)
                    outfile.close()
            except IOError:
                self.delete_file(user_info_file_path)  # delete user's info file
                self.delete_file(user_listenings_file_path)  # delete user's listenings file
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
            self.delete_file(user_listenings_file_path)
            return -1  # exit

    def get_user_weekly_track_chart(self, username, fr, to):
        """
        Collect weekly chart data of listened tracks for a given user
        :param username: lastfm username
        :param fr: crawled week start date
        :param to: crawled week end date
        """

        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

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

        with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
            user_info = json.load(infile)
            infile.close()
        user_info["crawled"][str(fr)]["listened tracks"] = tmp_listened_tracks

        # write uptaded crawled listened tracks
        with gzip.open(user_listenings_file_path, 'wt', encoding='utf-8') as outfile:
            json.dump(user_info, outfile, indent=4)
            outfile.close()

    def get_user_weekly_artist_chart(self, username, fr, to):
        """
           Collect weekly chart data of listened artists for a given user
           :param username: lastfm username
           :param fr: start date
           :param to: end date
        """

        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

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
                        # delete the "featuring" of the artist, if present
                        artist_name = self.clean_artist_name(tr['name'])

                        record = {
                            'artist_rank': tr['@attr']['rank'],
                            'artist': artist_name,
                            'artist_mbid': tr['mbid'],
                            'playcount': tr['playcount']
                        }

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

        with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
            user_info = json.load(infile)
            infile.close()

        user_info["crawled"][str(fr)]["listened artists"] = tmp_listened_artists

        # write uptaded crawled listened artists
        with gzip.open(user_listenings_file_path, 'wt', encoding='utf-8') as outfile:
            json.dump(user_info, outfile, indent=4)
            outfile.close()

    def get_user_weekly_album_chart(self, username, fr, to):
        """
           Collect weekly chart data of listened albums for a given user
           :param username: lastfm username
           :param fr: start date
           :param to: end date
        """

        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

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

        with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
            user_info = json.load(infile)
            infile.close()

        user_info["crawled"][str(fr)]["listened albums"] = tmp_listened_albums

        # write uptaded crawled listened albums
        with gzip.open(user_listenings_file_path, 'wt', encoding='utf-8') as outfile:
            json.dump(user_info, outfile, indent=4)
            outfile.close()

    def collect_artist_info_from_weekly_charts(self, username):
        """
        Collect details on artists listeded by a given user
        :param username: lastfm username
        """
        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

        # verify existance of user's file info
        res = self.check_if_file_exist(user_listenings_file_path)
        if res == -1:
            return
        else:

            with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
                user_info = json.load(infile)
                infile.close()

            # iterate over the crawled weeks
            for key, crawled_week in tqdm.tqdm(user_info["crawled"].items(), ncols=100, ascii=True,
                                               desc=username + " collect artists' info"):

                try:
                    # iterate over the tracks of a given crawled week to extract artists
                    for kkey, artists_chart in crawled_week["listened artists"].items():
                        artist_name = artists_chart['artist']
                        artist_mbid = artists_chart['artist_mbid']
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
        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

        # verify existance of user's file info
        res = self.check_if_file_exist(user_listenings_file_path)
        if res == -1:
            return
        else:
            with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
                user_info = json.load(infile)
                infile.close()

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

    def get_first_artist_listening_week(self, username, artist):
        """
        Return the first week in which the user has started to listen the artist (if he started)
        :param username: lastfm usename
         :param username: artist to check firts listenings
        """
        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

        # verify existance of user's file info
        res = self.check_if_file_exist(user_listenings_file_path)
        if res == -1:
            return -1
        else:

            first_listening_week = -1

            with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
                user_info = json.load(infile)
                infile.close()

            # iterate over the crawled weeks
            for key, crawled_week in user_info["crawled"].items():
                try:
                    check = crawled_week["listened artists"][artist]

                    # if no KeyError is raised it means the user has listened the artist in for the first time
                    # in this week
                    first_listening_week = key
                    break

                except KeyError as e:
                    continue  # continue week's iteration to find artist's listening
                except Exception as e:
                    print()
                    print(e)
                    print("In get_first_artist_listening")
                    return -1

            return int(first_listening_week)

    def get_last_artist_listening_week(self, username, artist):
        """
        Return the last week in which the user has started to listen the artist (if he started)
        :param username: lastfm usename
         :param username: artist to check last listening
        """
        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

        # verify existance of user's file info
        res = self.check_if_file_exist(user_listenings_file_path)
        if res == -1:
            return -1
        else:

            last_listening_week = -1

            with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
                user_info = json.load(infile)
                infile.close()

            reversed_weeks = []

            # iterate over the crawled weeks
            for key, crawled_week in user_info["crawled"].items():
                reversed_weeks.append(key)

            # reverse crawled weeks list
            reversed_weeks.reverse()

            for i in reversed_weeks:
                try:
                    check = user_info["crawled"][str(i)]["listened artists"][artist]

                    # if no KeyError is raised it means the user has listened the artist in for the first time
                    # in this week
                    last_listening_week = i
                    break

                except KeyError as e:
                    continue  # continue week's iteration to find artist's listening
                except Exception as e:
                    print()
                    print(e)
                    print("In get_first_artist_listening")
                    return -1

            return int(last_listening_week)


    def get_totalistening_periodlength(self, username, artist):
        """
         Function which counts the total playconuts of the artist passed by argument
         :return totallistenting: total playconuts of the artist
       """
        totallistenting = 0

        user_listenings_file_path = "data/users_listenings/" + username + "_listenings.json.gz"

        # verify existance of user's file info
        res = self.check_if_file_exist(user_listenings_file_path)
        if res == -1:
            return -1
        else:
            with gzip.open(user_listenings_file_path, 'rt', encoding='utf-8') as infile:
                user_info = json.load(infile)
                infile.close()


                # iterate over the crawled weeks
                for key, crawled_week in user_info["crawled"].items():
                    try:
                        playcount = crawled_week['listened artists'][artist]['playcount']

                        totallistenting = totallistenting + int(playcount)

                    except KeyError as e:
                        continue  # continue week's iteration to find other artist's playcount
                    except Exception as e:
                        print()
                        print(e)
                        print("In get_first_artist_listening")
                        return -1

                return int(totallistenting)


    def get_network(self, username):
        """
        Collect data on user's friends
        :param username: lastfm usename
        """

        user_info_file_path = "data/users_info/" + username + "_info.json.gz"

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

        with gzip.open(user_info_file_path, 'rt', encoding='utf-8') as infile:
            user_info = json.load(infile)
            infile.close()
        user_info["friends"] = tmp_friends

        try:
            # write user's friends
            with gzip.open(user_info_file_path, 'wt', encoding='utf-8') as outfile:
                json.dump(user_info, outfile, indent=4)
                outfile.close()
            return 0
        except IOError:
            self.delete_file(user_info_file_path)
            return -1

    def get_friends_of_friend(self, friend):
        """
        Get the friends of a user's friend
        :param friend: lastfm usename
        :return the set of the friends of the user's friend
        """

        tmp_friends = set()

        url = "http://ws.audioscrobbler.com/2.0/?method=user.getfriends&user=%s&api_key=%s&limit=500" \
              "&format=json" % \
              (friend, self.lfm_apikey)

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        if data != {} and data is not None:

            if len(data) > 0:
                for k in data['friends']['user']:
                    try:
                        tmp_friends.add(k['name'])
                    except KeyError as e:
                        print()
                        print("KeyError " + str(e))
                        print(url)
                        print("In  get_friends_of_friends")
                        continue
                    except Exception as e:
                        print()
                        print(e)
                        print(url)
                        print("In  get_friends_of_friends")
                        continue

        return tmp_friends

    def get_friend_weekly_artist_chart(self, friend, data_encoding, start_date, end_date):
        """
           Collect weekly chart data of listened artists for a given user
           :param friend: lastfm username
           :param self.data_encoding LastfmDataEncode istance for encoding friend and artist in the global files
           :param start_date starting crawling date
           :param end_date ending crawling date
           :return friend's listening data as a codified dict, in the format:
               {username: {  (week, artist): playcount  , ... , (week, artist): playcount}}
        """

        # get friend's crawling weeks
        # ask the friend's chart to the API
        url = "http://ws.audioscrobbler.com/2.0/?method=user.getweeklychartlist&user=%s&api_key=%s&format=json" % \
              (friend, self.lfm_apikey)

        max_num_attempts_read_url = 3
        data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

        # encode friend as integer
        friend_encoding = data_encoding.get_user_encoding(friend)

        tmp_listened_artists = {str(friend_encoding): {}}  # value of artists' listened by the friend

        if data != {} and data is not None:

            if data == 0:
                print("crawling == 0")
                return {str(friend_encoding): {}}

            selected_weeks = []

            # keep track of weeks to crawl
            for ch in data['weeklychartlist']['chart']:
                try:
                    fr = ch['from']
                    to = ch['to']

                    if int(fr) >= start_date and int(to) <= end_date:
                        selected_weeks.append((fr, to))

                except KeyError as e:
                    print()
                    print("KeyError " + str(e))
                    print(url)
                    print("In get_friend_weekly_artist_chart")
                    # skip iteration
                    continue
                except Exception as e:
                    print()
                    print(e)
                    print(url)
                    print("In get_friend_weekly_artist_chart")
                    # skip iteration
                    continue

            if len(selected_weeks) == 0:
                return {str(friend_encoding): {}}

            # get charts for the weeks
            for week in selected_weeks:
                fr = int(week[0])
                to = int(week[1])

                # ask which artists the user has listened during the crawled week passed by argument to the API
                url = "http://ws.audioscrobbler.com/2.0/?method=user.getweeklyartistchart&user=%s&api_key=%s&from=%s&to=%s" \
                      "&format=json" % \
                      (friend, self.lfm_apikey, fr, to)

                max_num_attempts_read_url = 3
                data = self.fetch_HTTP_response(url, max_num_attempts_read_url)

                if data != {} and data is not None:

                    # add tracks to the crawled week only if the friend has listened to
                    # some artist in this period
                    if len(data['weeklyartistchart']['artist']) > 0:

                        for tr in data['weeklyartistchart']['artist']:
                            artist_name = ""
                            playcount = 0
                            try:

                                # check if artist is present or is "<Unknown>" (don't collect data in this case)
                                if tr['name'] == "<Unknown>":
                                    continue  # next artist

                                # delete the "featuring" of the artist, if present
                                artist_name = self.clean_artist_name(tr['name'])
                                playcount = tr['playcount']

                                # get week's encoding
                                week_encoding = data_encoding.get_week_encoding(fr)

                                # get artists's ecoding
                                artist_encoding = data_encoding.get_artist_encoding(artist_name)

                                # add key = (week, artist) and valeu = playcount to the tmp_dict
                                tmp_listened_artists[str(friend_encoding)][str((week_encoding, artist_encoding))] =\
                                                                                                            playcount

                            except KeyError as e:
                                print()
                                print("KeyError " + str(e))
                                print(url)
                                print("In get_friend_weekly_artist_chart: " + "fr= " + str(fr) + "to= " + str(to))
                                # skip iteration
                                continue
                            except Exception as e:
                                print()
                                print(e)
                                print(url)
                                print("In get_friend_weekly_artist_chart: " + "fr= " + str(fr) + "to= " + str(to))
                                # skip iteration
                                continue

                continue   # friend didn't listened to something

        return tmp_listened_artists