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

def crawl_wikipedia_tags():
    """
        Function which crawles the music tags present in the "https://en.wikipedia.org/wiki/List_of_music_styles"
    """

    check = check_if_file_exist("music_wikipedia_tags_map.json")

    if check == -1:
        # create "music_tags_map.json" file
        touch("music_wikipedia_tags_map.json")

    not_wanted_main_music_tags = ["african", "arabic music", "asian", "east asian", "south and southeast asian",
                                  "avant-garde", "caribbean and caribbean-influenced", "easy listening",
                                  "other", "references", "external links", "bibliography", "see also"]
    main_music_tags = []  # list of the main music tags

    tags_map = {}

    # ask HTTP page
    max_num_attempts_read_url = 3
    url = "https://en.wikipedia.org/wiki/List_of_music_styles"
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
        for link in soup.find_all(class_="toctext"):

            span1 = (str(link)).split("<span class=\"toctext\">")
            span2 = span1[1].split("</span>")
            tag = span2[0]

            if tag == "R&amp;B and soul":
                tag = "r&b/soul"
            elif tag == "Hip hop":
                tag = "hip-hop/soul"
            elif tag == "Classical music":
                tag = "classical"
            elif tag == "Electronic music":
                tag = "electronic"
            elif tag == "Folk song":
                tag = "country"
            else:
                tag = tag.lower()

            if tag not in not_wanted_main_music_tags:
                main_music_tags.append(tag)

        count = 1

        # I get a list of main music genres and submusic genre that are under the main genres.
        # The pattern I get is the following:
        # 1) a div which contains the main genre and the list of corresponding subgenres:
        #    div = < div class ="div-col columns column-width" style="-moz-column-width: 20em; " \
        #            "-webkit-column-width: 20em; column-width: 20em;" >
        # 2) an hyperlink with points to the main genre and the starting of the subgenres list:
        #   <ul><li><a href="/wiki/Ambient_music" title="Ambient music">Ambient</a>
        # 3) the list of subgenres (ex. Ambient_dub stands under Ambient_music)
        #   <ul><li><a class="mw-redirect" href="/wiki/Ambient_dub" title="Ambient dub">Ambient dub</a></li>
        #   .....
        # 5) ending of div:
        #       </div>

        for div in soup.find_all(class_="div-col columns column-width",
                                  style="-moz-column-width: 20em; -webkit-column-width: 20em; column-width: 20em;"):

            # get div's content
            content_list = (str(div)).split("\n")

            # I div = African (excluded)
            # II div = South and southeast Asian (excluded)
            # III div = Blues
            # IV div = Caribbean and Caribbean-influenced (excluded)
            # V div = Electronic music
            # VI div = Hip hop
            # VII div = Jazz
            # VIII div = Latin
            # IX div = Pop
            # X div = R&B and soul
            # XI div = Rock

            # Not present:
            # a) Country
            # b) Easy listenings
            # c) Folk song
            for i in range(1, len(content_list) - 1):
                raw_subtag = content_list[i]

                # print("raw_subtag = " + str(raw_subtag))
                # print()

                # get subtag's name
                raw_subtag_list = raw_subtag.split(" title=\"")

                if len(raw_subtag_list) > 1:
                    raw_subtag_list2 = raw_subtag_list[1].split("\">")
                    tmp_subtag = raw_subtag_list2[0].split("(")  # remove parenthesis
                    subtag = tmp_subtag[0].strip()
                    subtag = subtag.lower()
                else:   # tag doesn't have the attribute 'title' but it's just a plain list
                    raw_subtag_list2 = raw_subtag_list[0].split("</li>")
                    raw_subtag_list3 = raw_subtag_list2[0].split("<li>")
                    tmp_subtag = raw_subtag_list3[1].split("(")  # remove parenthesis
                    subtag = tmp_subtag[0].strip()
                    subtag = subtag.lower()

                print("subrag = " + str(subtag))
                print()

                if count == 3:
                    tags_map[str(subtag)] = "blues"
                elif count == 5:
                    tags_map[str(subtag)] = "electronic"
                elif count == 6:
                    tags_map[str(subtag)] = "hip-hop/rap"
                elif count == 7:
                    tags_map[str(subtag)] = "jazz"
                elif count == 8:
                    tags_map[str(subtag)] = "latin"
                elif count == 9:
                    tags_map[str(subtag)] = "pop"
                elif count == 10:
                    tags_map[str(subtag)] = "r&b/soul"
                elif count == 11:
                    tags_map[str(subtag)] = "rock"

            count += 1  # increment counter for knowing at which tag I arrived

        # The div containing the "Country" music genre is differente from the others => I have to
        # write an adhoc function for retrieving its subtags. Also the div containing the "Folk music" is
        # equal to the div of the "Country" one and since I map the "folk" genre in the "country"
        # I keep the two soup finds attached
        for div2 in soup.find_all(class_="div-col columns column-width",
                                  style="-moz-column-width: 18em; -webkit-column-width: 18em; column-width: 18em;"):
            # get div's content
            content_list = (str(div2)).split("\n")

            # I don't consider:
            # a) first element = '<div class="div-col columns column-width" style="-moz-column-width: 30em;
            #                                                       -webkit-column-width: 30em; column-width: 30em;">'
            # b) last element = '</div>'

            # THe other elements:
            # a) start with <li><a href="/wiki/
            # b) end with </a></li>
            # c) I split at the ' title="' attributed and I get rid of the rest of the string( I split again ath the end
            # of the title attribute '">' )
            # The second element has a ''<ul>' preppended and the second-last has a </ul> appended, but their splitting
            # behave like the others
            for i in range(1, len(content_list) - 1):
                raw_subtag = content_list[i]

                if raw_subtag != "</ul>":
                    # get subtag's name
                    raw_subtag_list = raw_subtag.split(" title=\"")
                    raw_subtag_list2 = raw_subtag_list[1].split("\">")
                    tmp_subtag = raw_subtag_list2[0].split("(")  # remove parenthesis
                    subtag = tmp_subtag[0].strip()
                    subtag = subtag.lower()

                    tags_map[str(subtag)] = "country"  # I consider also "Folk music" as "classical"

        # The div containing the "Easy listening" music genre is differente from the others => I have to
        # write an adhoc function for retrieving its subtags
        for div2 in soup.find_all(class_="div-col columns column-width",
                            style="-moz-column-width: 30em; -webkit-column-width: 30em; column-width: 30em;"):

            # get div's content
            content_list = (str(div2)).split("\n")

            # I don't consider:
            # a) first element = '<div class="div-col columns column-width" style="-moz-column-width: 30em;
            #                                                       -webkit-column-width: 30em; column-width: 30em;">'
            # b) last element = '</div>'

            # THe other elements:
            # a) start with <li><a href="/wiki/
            # c) I split at the ' title="' attributed and I get rid of the rest of the string( I split again ath the end
            # of the title attribute '">' )
            # b) end with </a></li>
            # The second elemet has a ''<ul>' preppended and the second-last has a </ul> appended, but their splitting
            # behave like the others
            for i in range(1, len(content_list) - 1):
                raw_subtag = content_list[i]

                # get subtag's name
                raw_subtag_list = raw_subtag.split(" title=\"")
                raw_subtag_list2 = raw_subtag_list[1].split("\">")
                tmp_subtag = raw_subtag_list2[0].split("(")  # remove parenthesis
                subtag = tmp_subtag[0].strip()
                subtag = subtag.lower()

                tags_map[str(subtag)] = "classical"  # I consider "Easy listening" as "classical"

    # write uptaded crawled listened tracks
    with open("music_wikipedia_tags_map.json", 'wt', encoding='utf-8') as outfile:
        json.dump(tags_map, outfile, indent=4)
        outfile.close()

    print(main_music_tags)


crawl_wikipedia_tags()


