import requests
import urllib.request
import urllib.error
import os
from bs4 import BeautifulSoup
import time

__author__ = 'Giulio Rossetti', 'Alexandra Bradan'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com", "alexandrabradan@gmail.com"


class LastfmDataCrawler(object):
    """
    Constructor
    """
    def __init__(self):
        self.get_artists()

        self.get_top_listeners()

    def touch(self, path):
        """
            Create a new file in the specified path
            :param path: path where to create the new file
        """
        with open(path, 'a') as f:
            os.utime(path, None)
            f.close()

    def fetch_HTTP_response(self, url, max_num_attempts_read_url):
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
                self.fetch_HTTP_response(url, max_num_attempts_read_url)
            else:
                print(e)
                print("Connection refused: " + url)
                return {}  # failure, return empty json response
        except urllib.error.HTTPError as e:
            # try again to read HTTP response, if there are still attempts
            max_num_attempts_read_url -= 1
            if max_num_attempts_read_url > 0:
                self.fetch_HTTP_response(url, max_num_attempts_read_url)
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
                self.fetch_HTTP_response(url, max_num_attempts_read_url)
            else:
                print(e)
                print("Problem caused by url: " + url)
                return {}  # failure, return empty json response


    def delete_duplicate_from_file(self, infilepath, outfilepath):
        """
               Delete the duplicate elements from the file passed by argument
                :param infilepath: file's path with duplicates
                :param outfilepath: file's path without duplicates
        """

        lines_seen = set()  # holds lines already seen

        # delete content of the outfile (if present)
        open(outfilepath, "w").close()

        # open infile and outfile and write on the last one the non-duplicate elements of the first one
        infile = open(infilepath, "r")
        outfile = open(outfilepath, "a")
        for line in infile:
            if line not in lines_seen:  # not a duplicate
                # outfile.write(line)
                lines_seen.add(line)

        # write elements in alphabetical order
        outfile.writelines(sorted(lines_seen))
        outfile.close()

    def get_artists_or_users(self, url, num_pages, file, type):
        """
            Retrieve the top listeners' usernames of a given artist or the artists from the url passed as parameter
             :param url: web page where to find artist's top listeners or artists's names
             :param num_pages: in how many pages the artist's top listeners or artists's names occur
             :param file: file where to write the users or artists' usernames/names
             :param type: which type of usernames/names to write on the file
                              user => users' usernames
                              music => artist's names
          """

        for page in range(num_pages):

            tmp_url = url

            # from the second page I have to add "?page=x"
            if page > 0:
                tmp_url = str(tmp_url) + "?page=" + str(page + 1)  # +1 beacuse I start to count from 0

            # print(tmp_url)

            # ask HTTP page
            max_num_attempts_read_url = 3
            response = self.fetch_HTTP_response(tmp_url, max_num_attempts_read_url)

            # in case of succes, ask HTTP page
            if response != {} and response is not None:

                soup = BeautifulSoup(response, 'html.parser')
                # print(soup.prettify())

                # every username in the lastfm's webpage is stored in a link <a> with class "link-block-target",
                # inside a header <h3> with class "top-listeners-item-name", inside a <div> with class
                # " top-listeners-itemjs-link-blocklink-block"
                for link in soup.find_all(class_="link-block-target"):
                    """
                        soup.find_all returns 2 types of strings:
                        1) /user/username
                        2) /music/username
                        I retrieve the usernames only from the strings starting with "/user/"
                        and when I reach the strings starting with "/music/" I quit the loop
                    """
                    start_string = "/" + type + "/"
                    start_index = len(start_string)
                    if link.get('href').startswith(start_string):
                        link_len = len(link.get('href'))
                        # slice the string in order to get the username(string[i:j] =>
                        # get the string from index "i" to index "j")
                        username = link.get('href')[start_index:link_len]
                        # print(username)
                        file.write(username)
                        file.write("\n")
            else:
                # jump this iteration
                continue

    def get_top_listeners(self):
        """
               Get the top listeners of the artist present in the artists' file
        """

        # file from which to retrieve artists' names
        artists_file = open("data/artists.csv", "r")

        # create a tmp file where to write the artists' top listeners
        self.touch("data/tmp_users.csv")

        # file in which to write all the usernames retrieved from the artist's top listeners
        users_file = open('data/tmp_users.csv', 'a')

        # loop through the artists' file in order to retrieve his/her's
        # top listeners usernames
        for art in artists_file:
            # clean up the artist's name and convert every whitespace in +  and every "/" in a "%2F"
            tt = art[0:len(art)]
            aa = tt.replace(" ", "+")
            ss = aa.replace("/", "%2F")
            artist = ss.rstrip()

            # usually an artist's top listeners are shown in 9 pages
            top_listeners_pages = 9

            # partial url from which to retrieve artist's top listeners usernames
            url = "https://www.last.fm/music/" + artist + "/+listeners"

            # ask HTTP page
            max_num_attempts_read_url = 3
            response = self.fetch_HTTP_response(url, max_num_attempts_read_url)

            # ask HTTP page
            if response != {} and response is not None:

                """
                    BeautifulSoup => Python library for pulling data out of HTML files.
                    It works with your favorite parser to provide idiomatic ways of navigating, 
                    searching, and modifying the parse tree
                """
                soup = BeautifulSoup(response, 'html.parser')
                # print(soup.prettify())

                # check how many pages the artist's top listeners have
                num_pages = 1
                for i in range(1, top_listeners_pages):
                    # get all the <a> tag of the html page
                    for link in soup.find_all('a'):
                        # check if the page "i+1" exist
                        exist_page = "?page=" + str(i + 1)
                        # if the page "i+1" exists then an <a> tag with href="?page=x" exist
                        if str(link.get('href')).startswith(exist_page):
                            # update last surrely existing page
                            num_pages = i + 1

                # now that I have the number of page in which the top listeners occur,
                # I can parse that web pages and retrieve their usernames
                self.get_artists_or_users(url, num_pages, users_file, "user")
            else:
                # jump this iteration
                continue

        users_file.close()

        artists_file.close()

        # delete duplicate elements from users' file
        self.delete_duplicate_from_file("data/tmp_users.csv", "data/users.csv")

        # delete tmp file
        os.remove("data/tmp_users.csv")

    def get_artists(self):
        """
            Retrieve the artists present in the data/artists_charts_links file
            (the artsists_charts_links file contains lastfm new artists charts' urls
             from which to get artists' names)
        """

        # file of charts
        artists_link_file = open("data/artists_charts_links.csv", "r")

        # create a tmp file where to write the retrived artist from the charts' file
        self.touch("data/tmp_artists.csv")

        # file in which to write all the artists' names retrieved from the chart
        artists_file = open('data/tmp_artists.csv', 'a')

        # loop through the artists_charts_links' file in order to retrieve the artists
        # present in the chart present in that link
        for link in artists_link_file:

            # usually an artists present in a chart are shown in 50 pages
            chart_artists_pages = 50

            # partial url from which to retrieve artists's names from the chart
            url = str(link)
            # clean up the url
            ll = url[0:len(url)]
            uurl = ll.rstrip()

            # ask HTTP page
            max_num_attempts_read_url = 3
            response = self.fetch_HTTP_response(uurl, max_num_attempts_read_url)

            # ask HTTP page
            if response != {} and response is not None:

                """
                    BeautifulSoup => Python library for pulling data out of HTML files.
                    It works with your favorite parser to provide idiomatic ways of navigating, 
                    searching, and modifying the parse tree
                """
                soup = BeautifulSoup(response, 'html.parser')
                # print(soup.prettify())

                # check how many pages the chart have
                num_pages = 1
                for i in range(1, chart_artists_pages):
                    # get all the <a> tag of the html page
                    for linkk in soup.find_all('a'):
                        # check if the page "i+1" exist
                        exist_page = "?page=" + str(i + 1)
                        # if the page "i+1" exists then an <a> tag with href="?page=x" exist
                        if str(linkk.get('href')).startswith(exist_page):
                            # update last surrely existing page
                            num_pages = i + 1

                # now that I have the number of page in which the artists in the chart occur,
                # I can parse that web pages and retrieve their names
                self.get_artists_or_users(uurl, num_pages, artists_file, "music")

            else:
                # jump this iteration
                continue

        artists_file.close()

        artists_link_file.close()

        # delete duplicate elements from artists' file
        self.delete_duplicate_from_file("data/tmp_artists.csv", "data/artists.csv")

        # delete tmp file
        os.remove("data/tmp_artists.csv")