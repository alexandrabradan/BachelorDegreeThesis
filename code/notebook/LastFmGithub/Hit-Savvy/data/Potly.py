from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
from bs4 import BeautifulSoup
import requests
import urllib.request
import urllib.error
import os
import selenium
from selenium import webdriver
import sys
import pandas as pd
from matplotlib import colors
import json
import ast


def get_country_from_encoding(country_encoding):
    f = open("country_node_map1", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        country = line_array[0]
        c_encoding = int(line_array[1].replace("\n", ""))
        if c_encoding == int(country_encoding):
            return country
    f.close()
    return None


def compute_countries_mean_main_genre():
    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic", "hip-hop/rap", "jazz", "latin",
                 "pop", "hip-hop/rap", "reggae", "rock"]
    main_tags_encoding = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    tmp_continent_dict = {}
    data = pd.read_csv("filtered_node_list.csv", delimiter="\t", usecols=['Continent', 'Country', 'Main_genre'], encoding="utf-8")

    for row in data.itertuples():

        try:
            int(row.Continent)
        except ValueError:
            continue

        try:
             genres_dict = tmp_continent_dict[str(row.Continent)]
             tmp_counter = genres_dict[str(row.Main_genre)]
             tmp_counter += 1
             genres_dict[str(row.Main_genre)] = tmp_counter
             tmp_continent_dict[str(row.Continent)] = genres_dict
        except KeyError:
            tmp_continent_dict[str(row.Continent)] = {}
            # initialize continent's main music genres
            for tag in main_tags_encoding:
                tmp_continent_dict[str(row.Continent)][str(tag)] = 0
            # keep track of current user's main music genre
            tmp_continent_dict[str(row.Continent)][str(row.Main_genre)] = 1

    continents_main_genre = {}
    for continent, tags_dict in tmp_continent_dict.items():
        # order dict by descendet values
        sorted_dict = {k: v for k, v in sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)}

        if list(sorted_dict.values())[0] == list(sorted_dict.values())[1]:
            print("Continent = " + str(continent) + "has multiple main genres:")
            print(list(sorted_dict.values())[0])
            print(list(sorted_dict.values())[1])
            sys.exit(-1)
        else:
            continents_main_genre[str(continent)] = list(sorted_dict.keys())[0]

    print(continents_main_genre)
    print(len(continents_main_genre))

    return continents_main_genre


def compute_countries_mean_main_genre():
    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic", "hip-hop/rap", "jazz", "latin",
                 "pop", "hip-hop/rap", "reggae", "rock"]
    main_tags_encoding = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    tmp_country_dict = {}
    data = pd.read_csv("filtered_node_list.csv", delimiter="\t", usecols=['Continent', 'Country', 'Main_genre'], encoding="utf-8")

    for row in data.itertuples():

        try:
            int(row.Country)
        except ValueError:
            continue

        try:
             genres_dict = tmp_country_dict[str(row.Country)]
             tmp_counter = genres_dict[str(row.Main_genre)]
             tmp_counter += 1
             genres_dict[str(row.Main_genre)] = tmp_counter
             tmp_country_dict[str(row.Country)] = genres_dict
        except KeyError:
            tmp_country_dict[str(row.Country)] = {}
            # initialize country's main music genres
            for tag in main_tags_encoding:
                tmp_country_dict[str(row.Country)][str(tag)] = 0
            # keep track of current user's main music genre
            tmp_country_dict[str(row.Country)][str(row.Main_genre)] = 1

    countries_main_genre = {}
    for country_encoding, tags_dict in tmp_country_dict.items():

        country = get_country_from_encoding(int(country_encoding))

        # order dict by descendet values
        sorted_dict = {k: v for k, v in sorted(tags_dict.items(), key=lambda x: x[1], reverse=True)}

        if list(sorted_dict.values())[0] == list(sorted_dict.values())[1]:
            # print("Continent = " + str(country) + "has multiple main genres:")
            # print(list(sorted_dict.values())[0])
            # print(list(sorted_dict.values())[1])
            #  sys.exit(-1)
            countries_main_genre[str(country)] = list(sorted_dict.keys())[0]
        else:
            countries_main_genre[str(country)] = list(sorted_dict.keys())[0]

    # print(countries_main_genre)
    # print(len(countries_main_genre))

    return countries_main_genre


def collect_lat_and_long():
    countries_already_analyzed = []
    logfile = open("logfile", "r", encoding="utf-8")
    for line in logfile:
        countries_already_analyzed.append(line.replace("\n", ""))
    logfile.close()
    logfile = open("logfile", "a", encoding="utf-8")

    country_file = open("country_node_map1", "r", encoding="utf-8")
    lat_file = open("country_latitude.csv", "a", encoding="utf-8")
    long_file = open("country_longitude.csv", "a", encoding="utf-8")
    for line in country_file:
        line_array = line.split("\t")
        country = line_array[0]
        country_encoding = line_array[1].replace("\n", "")

        # check countries already analized
        if country in countries_already_analyzed:
            continue

        # get webbrowser (Firefox in my case)
        url = "https://www.latlong.net"
        browser = webdriver.Firefox("/home/alexandra//Downloads/geckodriver-v0.26.0-linux64")

        # login to latlong
        login_url = "https://www.latlong.net/user/login"
        browser.get(login_url)
        find_serial = browser.find_element_by_id("email")
        find_serial.clear()  # clear textholder
        find_serial.send_keys("example@gmail.com")  # write in textholder
        find_serial = browser.find_element_by_id("password1")
        find_serial.clear()  # clear textholder
        find_serial.send_keys("example123")  # write in textholder
        find_serial = browser.find_element_by_css_selector(".button")
        find_serial.click()

        # get webpage
        browser.get(url)
        find_serial = browser.find_element_by_id("place")
        find_serial.clear()  # clear textholder
        find_serial.send_keys("%s" % country)  # write in textholder

        find_serial = browser.find_element_by_id('btnfind')
        find_serial.click()  # click on find button

        # iterate over all the links (latitude and longitude are shown as result of one of this hyperlinks)
        for elem in browser.find_elements_by_xpath("//a[@href]"):
            link = elem.get_attribute("href")
            str_link = str(link)
            if "https://www.latlong.net/user/new-location?lname=" in str_link:
                res = str_link.split("&")
                lat = float(res[1].replace("lat=", ""))
                long = float(res[2].replace("lng=", ""))
                print(lat)
                print(long)

                lat_file.write(f"{country}\t{lat}\n")
                lat_file.flush()
                long_file.write(f"{country}\t{long}\n")
                long_file.flush()

        browser.close()  # close tab

        # keep track of the analyzed country
        countries_already_analyzed.append(country)
        logfile.write(country)
        logfile.write("\n")
        logfile.flush()

    country_file.close()
    lat_file.close()
    long_file.close()
    logfile.close()


def draw_genre_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    # m.drawlsmask()

    countries_main_genre = compute_countries_mean_main_genre()

    color_dict = {"0": "red", "1": "blue", "2": "orange", "3": "brown", "4": "yellow", "5": "lime", "6": "cyan",
                  "7": "gray", "8": "salmon", "9": "fuchsia", "10": "darkviolet", "11": "darkgreen", "12": "black"}
    countries_main_genre_colors = []

    lats = pd.read_csv("country_latitude.csv", usecols=['country_id', 'latitude'], encoding="utf-8")
    lons = pd.read_csv("country_longitude.csv", usecols=['country_id', 'longitude'], encoding="utf-8")

    for country in lats["country_id"]:
        color = color_dict[countries_main_genre[str(country)]]
        countries_main_genre_colors.append(color)

    # print main music genre statistics base on countries
    for color in color_dict.values():
        color_counter = countries_main_genre_colors.count(str(color))
        print(str(color) + "  = " + str(color_counter))

    x, y = m(lons["longitude"], lats["latitude"])
    for i in range(0, len(countries_main_genre_colors)):
        m.plot(x[i], y[i], 'o', color=countries_main_genre_colors[i], markersize=1)

def draw_basemap():
    # make sure the value of resolution is a lowercase L,
    #  for 'low', not a numeral 1
    fig = plt.figure(figsize=(8, 6), edgecolor='w')
    m = Basemap(projection='cyl', resolution=None,
                llcrnrlat=-90, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, )
    draw_genre_map(m)
    plt.show()

collect_lat_and_long()
compute_countries_mean_main_genre()
draw_basemap()
