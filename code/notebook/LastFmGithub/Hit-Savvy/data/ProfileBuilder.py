import os
import sys
import csv
import tqdm
import math
import json
import sqlite3
import operator
from math import pi
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from scipy.stats import iqr

import numpy as np
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')


class ProfileBuilder(object):

    def __init__(self, database_name, friendship_database):
        self.database_name = database_name
        self.friendship_db = friendship_database
        self.adopters = self.get_adopters()
        self.profile = {}  # {adopter: [PLDM_u]}
        self.artist_genre = self.get_artist_genre()  # {artist: main_genre}
        self.artist = {}  # {adopter: {good: counter}}}
        self.genre = {}
        self.slot = {}
        self.quantity = {}

    def get_adopters(self):
        """
            Function which retrieves the adopters in the database
        """
        conn = sqlite3.connect("%s" % self.database_name)
        curr = conn.cursor()
        curr.execute("""SELECT distinct adopter from adoptions""")
        res = curr.fetchall()

        adopters = []
        for ad in res:
            ad = ad[0]
            if ad not in adopters:
                adopters.append(ad)
        curr.close()
        conn.close()
        print("Adopters collected!")
        return adopters

    def get_artist_genre(self):
        """
            Function which collects into a dict the main genre of each artist
            :return: a dict {artist:main_genre}
        """
        # N.B. starting to count from 1
        main_genre_encoding = {"alternative": 1, "blues": 2, "classical": 3, "country": 4, "dance": 5, "electronic": 6,
                 "hip-hop/rap": 7, "jazz": 8, "latin": 9, "pop": 10, "r&b/soul": 11, "reggae": 12, "rock": 13}
        artist_genre = {}
        f = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
        for line in f:
            line_array = line.split("::")
            g = line_array[0]
            genre = int(main_genre_encoding[str((line_array[1].replace("\n", "")))])
            artist_genre[str(g)] = genre
        f.close()
        print("Artist music genres collected!")
        return artist_genre

    def get_user_followings(self, adopter):
        """
            Function which retrieves adopter's following
        """
        user_friends = []
        conn = sqlite3.connect("%s" % self.friendship_db)
        curr = conn.cursor()
        curr.execute("""SELECT target from friendship where source='%s'""" % adopter)
        res = curr.fetchall()

        for v in res:
            v = v[0]
            if v not in user_friends:
                user_friends.append(v)
        curr.close()
        conn.close()
        return user_friends

    def get_user_followers(self, adopter):
        """
            Function which retrieves adopter's followers
        """
        user_friends = []
        conn = sqlite3.connect("%s" % self.friendship_db)
        curr = conn.cursor()
        curr.execute("""SELECT source from friendship where target='%s'""" % adopter)
        res = curr.fetchall()

        for v in res:
            v = v[0]
            if v not in user_friends:
                user_friends.append(v)
        curr.close()
        conn.close()
        return user_friends

    def get_lists_same_size(self, list1, list2):
        """
            Function which returns teh two lists passed by argument having the same size
            (it adds zeros to the shortes list untill it reaches the length of the other list)
        """
        ml = max(len(list1), len(list2))
        new_list1 = np.concatenate((list1, np.zeros(ml - len(list1))))
        new_list2 = np.concatenate((list2, np.zeros(ml - len(list2))))
        return new_list1, new_list2

    def build_profile(self):
        """
            Function which for all users build their PLDM
        """
        conn = sqlite3.connect("%s" % self.database_name)
        # for each adopter build its  PLDM
        print("Building users PLDM:")
        for ad in tqdm.tqdm(self.adopters):
            curr = conn.cursor()
            curr.execute("""SELECT * from adoptions where adopter='%s' """ % ad)
            res = curr.fetchall()

            # initialize user's frequency dictionaries
            self.artist[str(ad)] = {}
            self.genre[str(ad)] = {}
            self.slot[str(ad)] = {}
            self.quantity[str(ad)] = {}

            for elem in res:
                g = elem[0]
                genre = self.artist_genre[str(g)]
                s = int(elem[2])
                q = int(elem[3])

                try:
                    self.artist[str(ad)][str(g)] += 1
                except KeyError:
                    self.artist[str(ad)][str(g)] = 1
                try:
                    self.genre[str(ad)][str(genre)] += 1
                except KeyError:
                    self.genre[str(ad)][str(genre)] = 1
                try:
                    self.slot[str(ad)][str(s)] += 1
                except KeyError:
                    self.slot[str(ad)][str(s)] = 1
                try:
                    self.quantity[str(ad)][str(q)] += 1
                except KeyError:
                    self.quantity[str(ad)][str(q)] = 1

            # craete adopter's PLDM
            self.profile[str(ad)] = {}
            self.profile[str(ad)]["nlistening"] = len(res)  # tot. adoptions made by the user
            self.profile[str(ad)]["nartists"] = len(self.artist[str(ad)])
            self.profile[str(ad)]["ngenres"] = len(self.genre[str(ad)])
            self.profile[str(ad)]["nslots"] = len(self.slot[str(ad)])
            self.profile[str(ad)]["nquantities"] = len(self.quantity[str(ad)])

            self.profile[str(ad)]["au"] = self.artist[str(ad)]  # {artist: counter} for current ad
            self.profile[str(ad)]["gu"] = self.genre[str(ad)]  # {genre: counter} for current ad
            self.profile[str(ad)]["su"] = self.slot[str(ad)]  # {slot: counter} for current ad
            self.profile[str(ad)]["qu"] = self.quantity[str(ad)]  # {quantity: counter} for current ad

            # get entropy (adopter behaviour predictability among artists/genres/slots/quantities counters)
            # ex. he listens to all the artists/genres the same number of times or he varies ...
            self.profile[str(ad)]["e_au"] = self.entropy(list(self.artist[str(ad)].values()), len(self.artist[str(ad)]))
            self.profile[str(ad)]["e_gu"] = self.entropy(list(self.genre[str(ad)].values()), len(self.genre[str(ad)]))
            self.profile[str(ad)]["e_su"] = self.entropy(list(self.slot[str(ad)].values()), len(self.slot[str(ad)]))
            self.profile[str(ad)]["e_qu"] = self.entropy(list(self.quantity[str(ad)].values()), len(self.quantity[str(ad)]))

            # get top (most supported item of a dictionary)
            self.profile[str(ad)]["hat_au"] = self.top(self.artist[str(ad)])
            self.profile[str(ad)]["hat_gu"] = self.top(self.genre[str(ad)])
            self.profile[str(ad)]["hat_su"] = self.top(self.slot[str(ad)])
            self.profile[str(ad)]["hat_qu"] = self.top(self.quantity[str(ad)])

            # get knee (most representative supported items of a dictionary)
            self.profile[str(ad)]["tilde_au"] = self.knee(self.artist[str(ad)])
            self.profile[str(ad)]["tilde_gu"] = self.knee(self.genre[str(ad)])
            self.profile[str(ad)]["tilde_su"] = self.knee(self.slot[str(ad)])
            self.profile[str(ad)]["tilde_qu"] = self.knee(self.quantity[str(ad)])

            self.profile[str(ad)]["followings"] = self.get_user_followings(str(ad))
            self.profile[str(ad)]["followers"] = self.get_user_followers(str(ad))

        conn.close()
        print("PLDM completes!")

    def entropy(self, x, classes=None):
        """
            Function which computes the normalized entropy of a list, defined as:
                entropy(x) = - sum{P(x_i)*log_b P(x_i)/log_bN} for i=1,...,N
            :param x: list for which to compute entropy
            :param classes: list length
        """
        # sort list by descending values
        sorted(x, reverse=True)
        val_entropy = 0
        n = np.sum(x)
        for freq in x:
            if freq == 0:
                continue
            p = 1.0 * freq / n
            val_entropy -= p * np.log2(p)

        if classes is not None and classes > 1:
            val_entropy /= np.log2(classes)
        return val_entropy

    def top(self, dict):
        """
            Function which returns the most supported item of a dictionary (key with the highest value)
        """
        return max(dict.items(), key=operator.itemgetter(1))[0]

    @staticmethod
    def __closest_point_on_segment(a, b, p):
        sx1 = a[0]
        sx2 = b[0]
        sy1 = a[1]
        sy2 = b[1]
        px = p[0]
        py = p[1]

        x_delta = sx2 - sx1
        y_delta = sy2 - sy1

        if x_delta == 0 and y_delta == 0:
            return p

        u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
        if u < 0:
            closest_point = a
        elif u > 1:
            closest_point = b
        else:
            cp_x = sx1 + u * x_delta
            cp_y = sy1 + u * y_delta
            closest_point = [cp_x, cp_y]

        return closest_point

    def __get_change_point(self, x, y):

        max_d = -float('infinity')
        index = 0

        for i in range(0, len(x)):
            c = self.__closest_point_on_segment(a=[x[0], y[0]], b=[x[len(x) - 1], y[len(y) - 1]], p=[x[i], y[i]])
            d = math.sqrt((c[0] - x[i]) ** 2 + (c[1] - y[i]) ** 2)
            if d > max_d:
                max_d = d
                index = i

        if len(y) >= 2:
            return min(index + 1, len(x) - 1), y[min(index + 1, len(x) - 1) - 1]
        else:
            return min(index + 1, len(x) - 1), y[min(index + 1, len(x) - 1)]

    def knee(self, frequencies):
        """
            Function which returns the most representative supported items of a dictionary
            (keys with the highest values, where the cardinality of the result is given by
            the elbow of the curve formed by all the values of the dict)
        """
        if len(frequencies.keys()) == 0:
            return frequencies
        a = list(frequencies.values())
        a = sorted(a)

        index, v_prev = self.__get_change_point(range(0, len(a)), a)
        new_freq = {}
        for k, v in frequencies.items():
            if v >= a[index]:
                new_freq[k] = frequencies[k]

        if len(new_freq) == 1 and len(a) >= 2:
            new_freq = {}
            for k, v in frequencies.items():
                if v >= v_prev:
                    new_freq[k] = frequencies[k]
        # sort dict by descending values
        sorted_new_freq = {k: v for k, v in sorted(new_freq.items(), key=lambda x: x[1], reverse=True)}
        return sorted_new_freq

    def compute_mu(self, dict_u, dict_v):
        """
            Function which computes the cosine similarity between the keys of the dicts
            passed by argument (trasforming first this dicts into integers since they are strings)
        """
        list1 = list(dict_u.keys())
        list2 = list(dict_v.keys())
        list1 = [int(x) for x in list1]
        list2 = [int(x) for x in list2]

        # fill shortest list with 0s
        new_list1, new_list2 = self.get_lists_same_size(list1, list2)
        # compute cosine similarity among the 2 lists (transformed in np.arrays)
        mu = cosine_similarity(new_list1.reshape(1, -1), new_list2.reshape(1, -1))
        return mu[0][0]  # access first element of a list of lists

    def get_user_mu(self, ad, following_or_followers_list, key_name):
        """
            Function which returns the mu associated to the adopter and the list passed by argument
        :param ad: adopter for which to get mu
        :param following_or_followers_list: adopter's list of followers/followings
        :param key_name: following_or_followers_list related name
        """
        mu_au = []
        mu_gu = []
        mu_su = []
        mu_qu = []

        for fr in following_or_followers_list:
            mu_au.append(self.compute_mu(self.profile[str(ad)]["tilde_au"], self.profile[str(fr)]["tilde_au"]))
            mu_gu.append(self.compute_mu(self.profile[str(ad)]["tilde_gu"], self.profile[str(fr)]["tilde_gu"]))
            mu_su.append(self.compute_mu(self.profile[str(ad)]["tilde_su"], self.profile[str(fr)]["tilde_su"]))
            mu_qu.append(self.compute_mu(self.profile[str(ad)]["tilde_qu"], self.profile[str(fr)]["tilde_qu"]))

        # InterQuartile mean
        """self.profile[str(ad)]["mu_au"] = iqr(np.array(mu_au), rng=(25, 75), interpolation='midpoint')
        self.profile[str(ad)]["mu_gu"] = iqr(np.array(mu_gu), rng=(25, 75), interpolation='midpoint')
        self.profile[str(ad)]["mu_su"] = iqr(np.array(mu_su), rng=(25, 75), interpolation='midpoint')
        self.profile[str(ad)]["mu_qu"] = iqr(np.array(mu_qu), rng=(25, 75), interpolation='midpoint')"""
        # just compute plain Mean
        key1 = "mu_au" + str(key_name)
        key2 = "mu_gu" + str(key_name)
        key3 = "mu_su" + str(key_name)
        key4 = "mu_qu" + str(key_name)
        self.profile[str(ad)][key1] = np.mean(np.array(mu_au))
        self.profile[str(ad)][key2] = np.mean(np.array(mu_gu))
        self.profile[str(ad)][key3] = np.mean(np.array(mu_su))
        self.profile[str(ad)][key4] = np.mean(np.array(mu_qu))

        key1 = "sigma_au" + str(key_name)
        key2 = "sigma_gu" + str(key_name)
        key3 = "sigma_su" + str(key_name)
        key4 = "sigma_qu" + str(key_name)
        self.profile[str(ad)][key1] = np.std(np.array(mu_au))
        self.profile[str(ad)][key2] = np.std(np.array(mu_gu))
        self.profile[str(ad)][key3] = np.std(np.array(mu_su))
        self.profile[str(ad)][key4] = np.std(np.array(mu_qu))

    def compute_mu_and_sigma(self, profile_file, readfile_flag):
        """
            Function which computes the cosine similarity and the standard deviation
            of the following features:
                - artists
                - genres
                - slots
                - quantities
            among all the pairs of adopters and respective friends
        """
        # check if for some error the profile had to be saved on file and now
        # I need to re-read it
        if readfile_flag is True:
            with open(profile_file, 'r', encoding='utf-8') as infile:
                self.profile = json.load(infile)

        # iterate over each adopter, retrieve his friends, retrieve adopters and friends tilde info
        # (most representative items of each set) and compare them using the cosine similarity
        # (InterQuartileMean (IQM) among all the cosine similarities found) and the standard deviation of the IQM
        print("Computing mu and sigma:")
        for ad in tqdm.tqdm(self.adopters):
            # retrive adopter's followings
            followings = self.profile[str(ad)]["followings"]
            self.get_user_mu(ad, followings, "_followings")

            # retrieve adopter's followers
            followers = self.profile[str(ad)]["followers"]
            self.get_user_mu(ad, followers, "_followers")
        print("Mu and sigma completed!")

    def compute_global_mu_and_sigma(self, profile_file, readfile_flag):
        """
            Function which computes the cosine similarity and the standard deviation
            of the following features:
                - artists
                - genres
                - slots
                - quantities
            among all the pairs of users in the database
        """
        # check if for some error the profile had to be saved on file and now
        # I need to re-read it
        if readfile_flag is True:
            with open(profile_file, 'r', encoding='utf-8') as infile:
                self.profile = json.load(infile)

        # TODO: I have to iterate over the splitted files in order to gather all users
        for i in range(0, len(self.adopters)):
            u = self.adopters[i]

            global_mu_au = []
            global_mu_gu = []
            global_mu_su = []
            global_mu_qu = []
            for j in range(0, len(self.adopters)):

                v = self.adopters[j]
                if u == v:  # u and v are the same adopter
                    continue

                global_mu_au.append(self.compute_mu(self.profile[str(u)]["tilde_au"], self.profile[str(v)]["tilde_au"]))
                global_mu_gu.append(self.compute_mu(self.profile[str(u)]["tilde_gu"], self.profile[str(v)]["tilde_gu"]))
                global_mu_su.append(self.compute_mu(self.profile[str(u)]["tilde_su"], self.profile[str(v)]["tilde_su"]))
                global_mu_qu.append(self.compute_mu(self.profile[str(u)]["tilde_qu"], self.profile[str(u)]["tilde_qu"]))

            # InterQuartile mean
            """self.profile["global_mu_au"] = iqr(np.array(global_mu_au), rng=(25,75), interpolation='midpoint')
            self.profile["global_mu_gu"] = iqr(np.array(global_mu_gu), rng=(25, 75), interpolation='midpoint')
            self.profile["global_mu_su"] = iqr(np.array(global_mu_su), rng=(25, 75), interpolation='midpoint')
            self.profile["global_mu_qu"] = iqr(np.array(global_mu_qu), rng=(25, 75), interpolation='midpoint')"""
            # just compute plain Mean
            self.profile[str(u)]["global_mu_au"] = np.mean(np.array(global_mu_au))
            self.profile[str(u)]["global_mu_gu"] = np.mean(np.array(global_mu_gu))
            self.profile[str(u)]["global_mu_su"] = np.mean(np.array(global_mu_su))
            self.profile[str(u)]["global_mu_qu"] = np.mean(np.array(global_mu_qu))

            self.profile[str(u)]["global_sigma_au"] = np.std(np.array(global_mu_au))
            self.profile[str(u)]["global_sigma_gu"] = np.std(np.array(global_mu_gu))
            self.profile[str(u)]["global_sigma_su"] = np.std(np.array(global_mu_su))
            self.profile[str(u)]["global_sigma_qu"] = np.std(np.array(global_mu_qu))

    def save_profile(self, profile_file):
        """
            Function which writes the results of the ProfileBuilder on file
        """
        with open(profile_file, 'w', encoding='utf-8') as outfile:
            json.dump(self.profile, outfile, indent=4)
            outfile.close()


class Prova(object):

    def make_spider(self, df, row, title, color):
        """
            Function which draw the radar charts of the clusters
            obtained with K-Means
        """
        # number of variable
        categories = list(df)[1:]
        N = len(categories)

        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the spider plot
        # fig, ax = plt.subplots(2, 2, row + 1, polar=True)
        fig = plt.figure()
        ax = fig.add_subplot(int(str(22) + str(row + 1)), polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='black', size=8)

        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([10, 20, 30], ["10", "20", "30"], color="black", size=7)
        plt.ylim(0, 40)

        # Ind1
        values = df.loc[row].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
        ax.fill(angles, values, color=color, alpha=0.4)

        # Add a title
        # plt.title(title, size=11, color=color, y=1.1)

        # ------- PART 2: Apply to all individuals
        # initialize the figure
        my_dpi = 96
        plt.figure(figsize=(1000 / my_dpi, 1000 / my_dpi), dpi=my_dpi)

        # Create a color palette:
        my_palette = plt.cm.get_cmap("Set2", len(df.index))

        # Loop to plot
        for row in range(0, len(df.index)):
            make_spider(row=row, title='group ' + df['group'][row], color=my_palette(row))
        plt.show()


if __name__ == "__main__":
    main_database = "lastfm.db"
    friendship_database = "friendship.db"

    profile_file = "profile.json"
    pb = ProfileBuilder(main_database, friendship_database)
    pb.build_profile()
    pb.save_profile(profile_file)
    pb.compute_mu_and_sigma(profile_file, True)
    profile_file = "profile_mu_and_sigma.json"
    pb.save_profile(profile_file)

    pb.compute_global_mu_and_sigma(profile_file, False)
    profile_file = "profile_mu_and_sigma_global.json"
    pb.save_profile(profile_file)
