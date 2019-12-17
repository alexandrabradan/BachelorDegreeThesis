import json
import os

import requests
import urllib.request
import urllib.error
import time
from bs4 import BeautifulSoup


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


def fetch_HTTP_response(url, max_num_attempts_read_url):
    """
       Try to read HTTP response from an URL, until reaching success
       or the maximum number of attempts
       :param url: url from which to read HTTP response
       :param max_num_attempts_read_url: max number of retires to read the HTTP response
    """
    # ask HTTP page
    try:
        response = requests.get(url, proxies={"http": "http://61.233.25.166:80"})
        requests.packages.urllib3.contrib.pyopenssl.extract_from_urllib3()
        return response.text
    except requests.exceptions.ConnectionError as e:
        # suspend requests for 5 seconds
        time.sleep(5)

        # try again to read HTTP response, if there are still attempts
        max_num_attempts_read_url -= 1
        if max_num_attempts_read_url > 0:
            fetch_HTTP_response(url, max_num_attempts_read_url)
        else:
            print(e)
            print("Connection refused: " + url)
            return {}  # failure, return empty json response
    except urllib.error.HTTPError as e:
        # try again to read HTTP response, if there are still attempts
        max_num_attempts_read_url -= 1
        if max_num_attempts_read_url > 0:
            fetch_HTTP_response(url, max_num_attempts_read_url)
        else:
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
            print(e)
            print("Problem caused by url: " + url)
            return {}  # failure, return empty json response

def crawl_tags():
    """
        Function which retrieve the most common music tags present at
        "https://www.musicgenreslist.com/". The tags' retrieval is possibile
        by parsing the web page's content. The result of the crawling can be found in the
        "music_tags_map.json" file, which is a JSON file containing:
        a) key => a music tag
        b) values => corresponding main music tag
    """

    check = check_if_file_exist("music_tags_map.json")

    if check == -1:
        # create "music_tags_map.json" file
        touch("music_tags_map.json")

    not_wanted_main_music_tags = ["Children's Music", "Commercial (thank you Sheldon Reynolds)", "Disney",
                                  "Enka", "French Pop", "German Folk", "German Pop", "Fitness & Workout",
                                  "Holiday", "Indie Pop", "Industrial", "Christian & Gospel",
                                  "Instrumental", "JPop", "KPop", "Karaoke", "Kayokyoku", "New Age",
                                  "Opera", "Singer/Songwriter", "Soundtrack", "Spoken Word", "Tex",
                                  "Vocal", "World", "Easy Listening", "Rockabilly", "Anime", "Contemporary Classical",
                                  "Comedy", "Easy Listening"]
    main_music_tags = []  # list of the main music tags

    tags_map = {}

    # ask HTTP page
    max_num_attempts_read_url = 3
    url = "https://www.musicgenreslist.com/"
    response = fetch_HTTP_response(url, max_num_attempts_read_url)

    # ask HTTP page
    if response != {} and response is not None:
        """
            BeautifulSoup => Python library for pulling data out of HTML files.
            It works with your favorite parser to provide idiomatic ways of navigating, 
            searching, and modifying the parse tree
        """
        soup = BeautifulSoup(response, 'html.parser')
        # print(soup.prettify())

        # every music tag in the web page is stored in a link <a> with "title"=" x - Music Genre"
        for link in soup.find_all('a'):

            a_title = link.get('title')

            if a_title is not None:

                res = a_title.find('-')

                if res != -1:

                    tmp_tag = a_title.split('-')

                    # music genre has 2 '-': one in its name and one as saparatore between the
                    # musci genre and the "- Mmusic genre" phrase
                    if len(tmp_tag) > 2:
                        # put again '-' between "Hip" and "Hop"
                        if tmp_tag[0] == "Hip" and tmp_tag[1] == "Hop/Rap ":
                            tag = (tmp_tag[0] + "-" + tmp_tag[1]).strip()  # remove starting and ending whitespaces
                        else:
                            tag = (tmp_tag[0] + tmp_tag[1]).strip()  # remove starting and ending whitespaces
                    else:
                        tag = tmp_tag[0].strip()  # remove starting and ending whitespaces

                    if tag not in not_wanted_main_music_tags:
                        tag = tag.lower()
                        main_music_tags.append(tag)

                        uls = []
                        for nextSibling in link.findNextSiblings():
                            if nextSibling.name.title not in not_wanted_main_music_tags and nextSibling.name == 'a':
                                break
                            if nextSibling.name == 'ul':

                                uls.append(nextSibling)

                                tmp_keys = str(nextSibling).replace('\n', '').replace("</li>", "").split("<li>")
                                # print(str(tmp_keys))

                                # don't consider the first keys array's element  (== '<ul>)'
                                for i in range(1, len(tmp_keys)):
                                    # substitue "&amp;" with "&"
                                    tmp_keys[i] = tmp_keys[i].replace("&amp;", "&")

                                    # check if the key starts with "a href=" => is an hyperlink
                                    if tmp_keys[i].startswith("<a href="):
                                        tmp_i = tmp_keys[i]
                                        tmp_i_array = tmp_i.split("</a>")
                                        tmp_i_array2 = tmp_i_array[0].split(">")

                                        tmp_keys[i] = tmp_i_array2[1].strip()

                                    # remove eventual parentesis present in the key [the parentesis aren't useful
                                    # for our mapping but consists only in thanks for some user]
                                    tmp_k = tmp_keys[i].split("(")
                                    k = tmp_k[0].strip()  # remove entual white spaces at the end
                                    final_key = k

                                    # check if the key is made up by "/". In this case the key is the sum of
                                    # 2 subkeys (with the exception of "Pop/Rock" which has to be splitted and
                                    # reconstructed as "Pop Rock")
                                    res = k.find('/')

                                    if res != -1:  # key is made up of 2 subkeys

                                        tmp_final_key = k.split("/")

                                        if k == "Pop/Rock":
                                            final_key = str(tmp_final_key[0]) + " " + str(tmp_final_key[1])
                                        else:

                                            # if the second subkey has a whitespace it means that we have to append
                                            # to the first subkey the last word of the second subkey
                                            second_subkey_splitted = tmp_final_key[1].split(" ")
                                            tmp_last_second_subkey_word = ""

                                            if len(second_subkey_splitted) > 1:
                                                tmp_last_second_subkey_word = \
                                                                second_subkey_splitted[(len(second_subkey_splitted) - 1)]
                                                first_subkey = str(tmp_final_key[0]) + tmp_last_second_subkey_word

                                                # assign first subkey to the tags map (the value is the corresponding main tag)
                                                first_subkey = first_subkey.replace("<ul>", "")
                                                first_subkey = first_subkey.replace("</ul>", "")
                                                first_subkey = first_subkey.replace("<em>", "and")
                                                first_subkey = first_subkey.replace("</em>", "and")
                                                first_subkey = first_subkey.replace("<src=\"https:", "")
                                                first_subkey = (first_subkey.replace("<", "")).strip()
                                                first_subkey = first_subkey.lower()

                                                tags_map[str(first_subkey)] = str(tag)

                                                final_key = tmp_final_key[1]  # second subkey is assigned outside the if block

                                            else:
                                                # assign first subkey to the tags map (the value is the corresponding main tag)
                                                f_k = tmp_final_key[0]
                                                first_subkey = f_k.replace("<ul>", "")
                                                first_subkey = first_subkey.replace("</ul>", "")
                                                first_subkey = first_subkey.replace("<em>", "and")
                                                first_subkey = first_subkey.replace("</em>", "and")
                                                first_subkey = first_subkey.replace("<src=\"https:", "")
                                                first_subkey = (first_subkey.replace("<", "")).strip()
                                                first_subkey = first_subkey.lower()

                                                tags_map[str(first_subkey)] = str(tag)

                                                final_key = tmp_final_key[1]  # second subkey is assigned outside the if block

                                    final_key = final_key.replace("<ul>", "")
                                    final_key = final_key.replace("</ul>", "")
                                    final_key = final_key.replace("<em>", "and")
                                    final_key = final_key.replace("</em>", "and")
                                    final_key = final_key.replace("<src=\"https:", "")
                                    final_key = (final_key.replace("<", "")).strip()
                                    final_key = final_key.lower()

                                    if final_key != "ul>" and final_key != "ul>p>script async=\"\" src=\"https:":
                                        # assign clean-up key to the tags map (the value is the corresponding main tag)
                                        tags_map[str(final_key)] = str(tag)

                                # print(str(tags_map))
                                # print(len(tags_map))

        # print(str(tags_map))
        # print()

        # write uptaded crawled listened tracks
        with open("music_tags_map.json", 'wt', encoding='utf-8') as outfile:
            json.dump(tags_map, outfile, indent=4)
            outfile.close()

        print(main_music_tags)


# crawl the music tags present at "https://www.musicgenreslist.com/"
crawl_tags()