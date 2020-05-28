import networkx as nx
import math
import sqlite3
import numpy as np
import operator
import tqdm
import sys
import json
import os
from itertools import (takewhile, repeat)

__author__ = 'rossetti'
__license__ = "GPL"
__email__ = "giulio.rossetti@gmail.com"


class EarlyAdoptersThreshold(object):

    def __init__(self, database_name):
        self.database_name = database_name

    def __identify_max(self, row, index):
        if float(row[index]) <= float(row[index + 1]) and index < len(row) - 2:
            return self.__identify_max(row, index + 1)
        else:
            return index

    def __build_db(self, database_name):
        """
        Function which builds a database (the main database or the ones relative to the 4 training sets)
        and populates the database with the table "adoptions"
        :param database_name: the name of the database to create
        """
        conn = sqlite3.connect(database_name)
        conn.execute("""DROP TABLE IF EXISTS adoptions;""")
        conn.commit()
        conn.execute("""CREATE TABLE IF NOT EXISTS adoptions
                       (good TEXT  NOT NULL,
                       adopter TEXT NOT NULL,
                       slot      INTEGER NOT NULL,
                       quantity  INTEGER
                       );""")

        conn.close()

    def __build_table_for_test_set(self, database_name):
        """
        Function which constructs in the database of one of the 4 training sets the table "adoptions_test"
        to use as training set
        :param database_name: the name of the database in which to create the table
        """
        conn = sqlite3.connect(database_name)
        conn.execute("""DROP TABLE IF EXISTS adoptions_test;""")
        conn.commit()
        conn.execute("""CREATE TABLE  adoptions_test
                       (good TEXT  NOT NULL,
                       adopter TEXT NOT NULL,
                       slot      INTEGER NOT NULL,
                       quantity  INTEGER
                       );""")

        conn.close()

    def insert_adoptions_into_training_set(self, main_db_name, training_set_db_name):
        """
            Function which populates the "adoptions" table present in the "training_set_db_name" database
            :param main_db_name: main database from which to retrieve the info
        """

        # retrieve training set's adoptions
        conn = sqlite3.connect(main_db_name)
        curr = conn.cursor()
        curr.execute("""SELECT * from adoptions""")
        adoption_logs = curr.fetchall()
        curr.close()
        conn.close()

        # retrieve training set's adoptions
        count = 0
        goods_inserted = []
        conn = sqlite3.connect(training_set_db_name)
        for g, a, s, q in tqdm.tqdm(adoption_logs):
            s = int(s) - 26  # convert week to 0
            conn.execute("""INSERT into adoptions (good, adopter, slot, quantity) VALUES ('%s', '%s', %d, %d)""" %
                         (g, a, int(s), int(q)))
            if int(g) not in goods_inserted:
                goods_inserted.append(int(g))
            count += 1
            if count % 10000 == 0:
                conn.commit()

        conn.commit()
        conn.execute('CREATE INDEX good_idx_training on adoptions(good)')
        conn.execute('CREATE INDEX adopter_idx_training on adoptions(adopter)')
        conn.execute('CREATE INDEX slot_idx_training on adoptions(slot)')
        conn.close()

        conn = sqlite3.connect(main_db_name)
        curr = conn.cursor()
        curr.execute("""SELECT distinct good from adoptions """)
        res = curr.fetchall()
        curr.close()
        conn.close()

        if len(res) == len(goods_inserted):
            print("All goods added to training set")
        else:
            print("Missing goods in training set")
            sys.exit(-1)

    def insert_adoptions_into_test_set_modified(self, main_db_name, training_set_db_name, test_set_start_edge,
                                       test_set_end_edge):
        """
            Function which populates the "adoptions_test" table present in the "training_set_db_name" database
            :param main_db_name: main database from which to retrieve the info
            :param training_set_db_name: database where the "adoptions_test" table if found
            :param test_set_start_edge: test set start week
            :param test_set_end_edge: test set end week
        """
        # retrieve training set's goods
        conn = sqlite3.connect(main_db_name)
        curr = conn.cursor()
        curr.execute("""SELECT * from adoptions where (slot >= '%d' AND slot <='%d') order by slot asc
                             """ % (test_set_start_edge, test_set_end_edge))
        adoption_logs = curr.fetchall()
        curr.close()
        conn.close()

        count = 0
        goods_inserted = {}
        conn = sqlite3.connect(training_set_db_name)
        for g, a, s, q in tqdm.tqdm(adoption_logs):
            first_week = 0
            try:
                first_week = goods_inserted[str(g)]
            except KeyError:
                first_week = int(s)
                goods_inserted[str(g)] = int(first_week)

            s = int(s) - int(first_week)  # convert week to 0
            conn.execute(
                """INSERT into adoptions_test (good, adopter, slot, quantity) VALUES ('%s', '%s', %d, %d)""" %
                (g, a, int(s), int(q)))
            count += 1
            if count % 10000 == 0:
                conn.commit()

        conn.commit()
        conn.execute('CREATE INDEX good_idx_test on adoptions_test(good)')
        conn.execute('CREATE INDEX adopter_idx_test on adoptions_test(adopter)')
        conn.execute('CREATE INDEX slot_idx_test on adoptions_test(slot)')
        conn.close()

    def load_splitted_db_data(self, main_db_name, training_set_db_name, test_set_start_edge, test_set_end_edge):
        """
            Function which constructs the sub-database, based on the differen training set edges that I choose
            :param main_db_name: main database's name
        """

        # build training set database
        # self.__build_db(training_set_db_name)
        # retrieve training set's adoptions
        # self.insert_adoptions_into_training_set(main_db_name, training_set_db_name)

        # build test set table (inside training set db)
        self.__build_table_for_test_set(training_set_db_name)
        # retrieve test set's adoptions
        self.insert_adoptions_into_test_set_modified(main_db_name, training_set_db_name, test_set_start_edge, test_set_end_edge)

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

    def check_if_directories_exist_otherwise_create_them(self):
        """
            Function which checks if the following directories exist, otherwise creates them:
            - simple classifier
            - prediction
            - hit_model
            - flop_model
        """

        # directories = ["simple classifier", "prediction", "hit_model", "flop_model"]
        directories = ["./prediction/", "./hit_model/", "./flop_model/"]
        for dirr in directories:
            exist = self.check_if_directory_exist(dirr)

            if exist == -1:
                self.create_directory(dirr)

    def compute_active_period(self, hits_train, flops_train, goods_to_threshold, success_def, hit):
        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        cur.execute("""SELECT * from HFscore """)
        res = cur.fetchall()

        # retrive current success definition HitSavvy
        adopters = []
        for elem in tqdm.tqdm(res):
            a = elem[0]
            hf = float(elem[1])
            if hit is True:
                if hf <= 0.0:  # check if HF-propensity is >0
                    continue
            else:
                if hf >= 0.0:  # check if HF-propensity is < 0
                    continue
            if a not in adopters:
                adopters.append(a)

        tmp_average_activity = {}
        for ad in adopters:
            cur.execute(
                """SELECT good, min(slot) as start from adoptions where adopter=%s group by good order by min(slot) asc;""" % ad)
            first_adoptions = cur.fetchall()
            f = 0
            hi = 0
            hl = 0
            hpropensities = {}

            # compute HF-propensity on weekly basis
            idx = 0
            for i in first_adoptions:
                if int(i[1]) > idx + 1:
                    # compute HF-score for the old week
                    try:
                        hf_week = float(hi + hl + f) / float(hi - hl - f)
                        hpropensities[str(idx)] = hf_week
                    except ZeroDivisionError:
                        hpropensities[str(idx)] = 0.0
                    except Exception:
                        hpropensities[str(idx)] = 0.0

                    # reset scores
                    f = 0
                    hi = 0
                    hl = 0
                    # increase old week (just analyzed)
                    idx += 1

                    while int(i[1]) > idx:
                        hpropensities[str(idx)] = 0.0  # all 0 since in this week the user has not listened anything
                        idx += 1

                elif int(i[1]) == idx + 1:  # new week encountered

                    # compute HF-score for the old week
                    try:
                        hf_week = float(hi + hl + f) / float(hi - hl - f)
                        hpropensities[str(idx)] = hf_week
                    except ZeroDivisionError:
                        hpropensities[str(idx)] = 0.0
                    except Exception:
                        hpropensities[str(idx)] = 0.0

                    # reset scores
                    f = 0
                    hi = 0
                    hl = 0

                    # update current week tracker
                    idx = int(i[1])  # == idx + 1

                if i[0] in flops_train:
                    f -= 1
                else:
                    if i[1] <= goods_to_threshold[i[0]]:
                        hi += 1
                    else:
                        hl -= 1

            # compute HF-score for the last week
            try:
                hf_week = float(hi + hl + f) / float(hi - hl - f)
                hpropensities[str(idx)] = hf_week
            except ZeroDivisionError:
                hpropensities[str(idx)] = 0.0
            except Exception:
                hpropensities[str(idx)] = 0.0

            # compute AVERAGE ACTIVE PERIOD
            tmp_consecutive_weeks = 0
            counter_consectutive_weeks = 0
            update = False
            for week, hf_week in hpropensities.items():
                if hit is True:
                    if hf_week > 0.0:
                        update = True
                        tmp_consecutive_weeks += 1
                    else:
                        if update is True:
                            update = False
                            counter_consectutive_weeks += 1
                else:
                    if hf_week < 0.0:
                        update = True
                        tmp_consecutive_weeks += 1
                    else:
                        if update is True:
                            update = False
                            counter_consectutive_weeks += 1
            # consider last element
            if update is True:
                counter_consectutive_weeks += 1

            # keep track of current HitSavvy active period
            try:
                average_activity = int(tmp_consecutive_weeks / counter_consectutive_weeks)
                tmp_average_activity[str(ad)] = average_activity
            except ZeroDivisionError:
                continue
            except ValueError:
                continue
        conn.close()

        # write HitSaivves's active period on file
        if hit is True:
            active_file = "active_period_hit_savvies_success_def" + str(success_def) + ".json"
        else:
            active_file = "active_period_flop_adopters_success_def" + str(success_def) + ".json"
        with open(active_file, 'w', encoding='utf-8') as out:
            json.dump(tmp_average_activity, out, indent=4)

    def set_hit_flop(self, hit_flop_map):
        """
        Configure the hit/flop training split

        :param hit_flop_map:
        :return:
        """
        f = open(hit_flop_map)
        hits_train, flops_train = [], []
        for l in f:
            l = l.rstrip().split(",")
            if int(l[1]) == -1:
                flops_train.append(l[0])
            else:
                hits_train.append(l[0])
        return hits_train, flops_train

    @staticmethod
    def __compute_distance(x1, y1, x2, y2, x3, y3):
        y4 = (y2 - y1) * (x3 - x1) / (x2 - x1) + y1
        return math.fabs(y4 - y3)

    def execute(self):
        """
        Identify the final slot of innovators adoptions.

        :param goods_adoption_trends: dictionary {good_id: [n_adoptions_0,..., n_adoptions_N]}
        :return: dictionary {good_id: id_slot}
        """

        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        cur.execute("""SELECT distinct good from adoptions""")
        goods = cur.fetchall()

        goods_thresholds = {}
        for good in goods:
            good = good[0]
            distance_max = 0.0
            slot_threshold = 0

            cur = conn.cursor()
            cur.execute(
                """SELECT slot, count(*) from adoptions where good='%s' group by slot order by slot asc""" % good)
            res = cur.fetchall()

            adoption_trend = []
            # idx = 1  # why skipping the first week ???
            idx = 0
            for x in res:
                if idx == x[0]:
                    adoption_trend.append(x[1])
                    idx += 1
                else:
                    while idx < x[0]:
                        adoption_trend.append(0)
                        idx += 1

                    # self-added because otherwise this AT is omitted from adoption_trend
                    if idx == x[0]:
                        adoption_trend.append(x[1])
                        idx += 1

            for slot in range(0, len(adoption_trend) - 1):
                if float(adoption_trend[slot]) <= float(adoption_trend[slot + 1]):
                    if slot < len(adoption_trend) - 2:
                        indice_max = self.__identify_max(adoption_trend, slot + 1)
                    else:
                        indice_max = len(adoption_trend) - 1
                    if slot != 1:
                        tmp_distance = self.__compute_distance(slot - 1, float(adoption_trend[slot - 1]), indice_max,
                                                               float(adoption_trend[indice_max]), slot,
                                                               float(adoption_trend[slot]))
                    else:
                        tmp_distance = self.__compute_distance(slot, float(adoption_trend[slot]), indice_max,
                                                               float(adoption_trend[indice_max]), slot + 1,
                                                               float(adoption_trend[slot + 1]))

                    if tmp_distance > distance_max:
                        distance_max = tmp_distance
                        slot_threshold = slot

            goods_thresholds[good] = slot_threshold
        return goods_thresholds


class HFPropensity(object):

    def __init__(self, database_name, hits_train, flops_train, goods_to_threshold):
        self.flops_train = flops_train
        self.hits_train = hits_train
        self.goods_to_threshold = goods_to_threshold
        self.database_name = database_name
        self.__build_table()

    def __build_table(self):
        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""DROP TABLE IF EXISTS HFscore""")
        conn.commit()
        conn.execute("""CREATE TABLE IF NOT EXISTS HFscore
                               (adopter TEXT NOT NULL,
                               value      REAL NOT NULL,
                               hits INTEGER NOT NULL DEFAULT 0,
                               flops INTEGER NOT NULL DEFAULT 0
                               );""")

        conn.execute("""DROP TABLE IF EXISTS Coverage""")
        conn.commit()
        conn.execute("""CREATE TABLE IF NOT EXISTS Coverage
                                       (adopter TEXT NOT NULL,
                                       good      TEXT NOT NULL,
                                       hit INTEGER NOT NULL DEFAULT 0
                                       );""")

        conn.close()

    def execute_old(self):

        adopters_to_hits = self.__build_adopter_to_innovator(self.hits_train, self.goods_to_threshold)
        conn = sqlite3.connect("%s" % self.database_name)

        adopters_to_flops = {}
        for flop in self.flops_train:
            cur = conn.cursor()
            cur.execute("""SELECT * from adoptions where good='%s';""" % flop)
            adoptions = cur.fetchall()
            for a in adoptions:
                if a[1] not in adopters_to_hits:
                    adopters_to_hits[a[1]] = {}
                    adopters_to_hits[a[1]]["innovator"] = 0
                    adopters_to_hits[a[1]]["notInnovator"] = 0
                    adopters_to_hits[a[1]]["insuccesso"] = 0
                if a[1] not in adopters_to_flops:
                    adopters_to_flops[a[1]] = {}
                    adopters_to_flops[a[1]]['insuccesso'] = 1
                    adopters_to_flops[a[1]]['goods'] = {flop: 0}
                else:
                    if flop not in adopters_to_flops[a[1]]['goods']:
                        adopters_to_flops[a[1]]['insuccesso'] += 1
                        adopters_to_hits[a[1]]["insuccesso"] += 1
                        adopters_to_flops[a[1]]['goods'][flop] = 0

        adopter_to_epsilon_mu = self.__compute_hf_propensity(adopters_to_hits, adopters_to_flops)

        for a in adopter_to_epsilon_mu:
            hits, flops = 0, 0
            if a in adopters_to_hits and 'goods' in adopters_to_hits[a]:
                hits = len(adopters_to_hits[a]['goods'])
            if a in adopters_to_flops and 'goods' in adopters_to_flops[a]:
                flops = len(adopters_to_flops[a]['goods'])

            conn.execute("""INSERT into HFscore (adopter, value, hits, flops) VALUES ('%s', %f, %d, %d)""" %
                         (a, adopter_to_epsilon_mu[a]["misura"], hits, flops))
        conn.commit()

        for a in tqdm.tqdm(adopters_to_flops):
            if adopter_to_epsilon_mu[a]["misura"] < 0:
                if 'goods' in adopters_to_flops[a]:
                    for g in adopters_to_flops[a]['goods']:
                        conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 0)""" %
                                     (a, g))
        conn.commit()

        for a in tqdm.tqdm(adopters_to_hits):
            if adopter_to_epsilon_mu[a]["misura"] > 0:
                if 'goods' in adopters_to_hits[a]:
                    for g in adopters_to_hits[a]['goods']:
                        conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 1)""" %
                                     (a, g))
        conn.commit()

        conn.close()

    def execute(self):
        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        cur.execute("""SELECT distinct(adopter) from adoptions;""")
        adopters = cur.fetchall()

        adopters_to_hit = {}
        adopters_to_flop = {}

        for ad in adopters:
            ad = ad[0]
            cur.execute(
                """SELECT good, min(slot) as start from adoptions where adopter=%s group by good order by min(slot) asc;""" % ad)
            first_adoptions = cur.fetchall()

            scores = {}
            f = 0
            hi = 0
            hl = 0

            if ad not in adopters_to_hit:
                adopters_to_hit[ad] = []
            if ad not in adopters_to_flop:
                adopters_to_flop[ad] = []

            last_it = None

            for i in first_adoptions:

                if last_it is None:
                    last_it = i[1]

                n = i[1]
                """"
                if n > last_it + 1:
                    last_it += 1
                    while last_it < n:
                        scores[last_it] = (0, scores[last_it-1][0])
                        last_it += 1
                """

                last_it = n

                if i[0] in self.flops_train:
                    adopters_to_flop[ad].append(i[0])
                    f -= 1
                else:
                    if i[1] <= self.goods_to_threshold[i[0]]:
                        adopters_to_hit[ad].append(i[0])
                        hi += 1
                    else:
                        hl -= 1

            try:
                hpropensity = float(hi + hl + f) / float(hi - hl - f)
            except ZeroDivisionError:
                continue
            except Exception:
                continue

            if hpropensity < 0:
                if f == 0:
                    continue
                if f == 0 and hl == 0:
                    continue
                if (-hl) < (-f):  # true Flop-adopter (more Flops than later Hits)
                    conn.execute("""INSERT into HFscore (adopter, value, hits, flops) VALUES ('%s', %f, %d, %d)""" %
                                 (ad, hpropensity, hi, -f))
                    conn.commit()
                    for g in adopters_to_flop[ad]:
                        conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 0)""" % (ad, g))
                    conn.commit()

            if hpropensity > 0:
                conn.execute("""INSERT into HFscore (adopter, value, hits, flops) VALUES ('%s', %f, %d, %d)""" %
                             (ad, hpropensity, hi, -f))
                conn.commit()
                for g in adopters_to_hit[ad]:
                    conn.execute("""INSERT into Coverage (adopter, good, hit) VALUES ('%s', '%s', 1)""" % (ad, g))
            conn.commit()
        conn.close()

    def __build_adopter_to_innovator(self, hits_train, goods_to_threshold):
        conn = sqlite3.connect("%s" % self.database_name)
        adopter_as_innovator = {}

        for hit in hits_train:
            cur = conn.cursor()
            cur.execute("""SELECT * from adoptions where good='%s';""" % hit)
            adoptions = cur.fetchall()
            for a in adoptions:
                if a[1] not in adopter_as_innovator:
                    adopter_as_innovator[a[1]] = {}
                    adopter_as_innovator[a[1]]["innovator"] = 0
                    adopter_as_innovator[a[1]]["notInnovator"] = 0
                    adopter_as_innovator[a[1]]["insuccesso"] = 0
                    adopter_as_innovator[a[1]]['goods'] = {}

                if hit not in adopter_as_innovator[a[1]]['goods']:
                    if int(goods_to_threshold[a[0]]) < int(a[2]):
                        adopter_as_innovator[a[1]]["notInnovator"] += 1
                    else:
                        adopter_as_innovator[a[1]]["innovator"] += 1
                        adopter_as_innovator[a[1]]['goods'][hit] = 0
        conn.close()
        return adopter_as_innovator

    def __compute_hf_propensity(self, adopters_to_hits, adopters_to_flops):
        goods_count = len(self.hits_train)
        adopters_to_epsilon_mu = {}
        for adopter in adopters_to_hits:
            adopters_to_epsilon_mu[adopter] = {}
            if adopter not in adopters_to_flops:
                adopters_to_epsilon_mu[adopter]["mu"] = 0.0
            else:
                adopters_to_epsilon_mu[adopter]["mu"] = float(adopters_to_flops[adopter]['insuccesso']) / float(
                    goods_count)
            if adopters_to_hits[adopter]["innovator"] + adopters_to_hits[adopter]["notInnovator"] != 0:
                adopters_to_epsilon_mu[adopter]["epsilon"] = float((adopters_to_hits[adopter]["innovator"] -
                                                                    adopters_to_hits[adopter]["notInnovator"])) / \
                                                             float((adopters_to_hits[adopter]["innovator"] +
                                                                    adopters_to_hits[adopter]["notInnovator"]))
            else:
                adopters_to_epsilon_mu[adopter]["epsilon"] = 0.0

        for adopter in adopters_to_epsilon_mu:
            adopters_to_epsilon_mu[adopter]["misura"] = adopters_to_epsilon_mu[adopter]["epsilon"] * (
                    1 - adopters_to_epsilon_mu[adopter]["mu"])

        return adopters_to_epsilon_mu


class WMSC(object):
    def __init__(self, database_name, innovators=True):
        self.database_name = database_name
        self.innovators = innovators

    def read_network(self):
        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        if self.innovators:
            cur.execute("""SELECT * from Coverage where hit=1;""")
        else:
            cur.execute("""SELECT * from Coverage where hit=0;""")
        edges = cur.fetchall()

        g = nx.DiGraph()
        for e in edges:
            g.add_edge(e[0], e[1])

        auth = {}
        converged = False
        limit = 100
        # Possible convergence issue
        while not converged:
            try:
                hub, auth = nx.hits(g, max_iter=limit, tol=1)
                converged = True
            except:
                limit += 50
                print("HITS Convergence issue: retry")

        adopters = sorted(auth.items(), key=operator.itemgetter(1))
        adopters.reverse()

        class_type = "hit"
        if not self.innovators:
            class_type = "flop"

        conn.execute("""CREATE TABLE IF NOT EXISTS HubsAuth_%s
                        (adopter TEXT NOT NULL
                         );""" % class_type)

        step = float(len(adopters)) / 10
        count = 0

        for covered in range(1, 11):
            for a in adopters:
                if count < covered * step:
                    conn.execute("""INSERT INTO HubsAuth_%s (adopter) VALUES (%s)""" % (class_type, a[0]))
                    count += 1
        conn.commit()
        conn.close()

    def __read_data(self):

        conn = sqlite3.connect("%s" % self.database_name)

        cur = conn.cursor()
        if self.innovators:
            cur.execute("""SELECT * from HFscore where value>0;""")
        else:
            cur.execute("""SELECT * from HFscore where value<0;""")
        adopters_hf = cur.fetchall()

        adopter_to_hf = {a[0]: a[1] for a in adopters_hf}

        cur = conn.cursor()
        if self.innovators:
            cur.execute("""SELECT * from Coverage where hit=1;""")
        else:
            cur.execute("""SELECT * from Coverage where hit=0;""")
        edges = cur.fetchall()

        node_covered = {}
        product_covered = {}
        for r in edges:
            product = r[1]
            adopter = r[0]

            if adopter not in adopter_to_hf:
                continue

            if product not in product_covered:
                product_covered[product] = [adopter]
            else:
                product_covered[product].append(adopter)

            if adopter not in node_covered:
                node_covered[adopter] = [[product, ], 1, float(adopter_to_hf[adopter])]
            else:
                old = node_covered[adopter]
                novel = old[0]
                novel.append(product)
                node_covered[adopter] = [novel, old[1] + 1, old[2]]

        conn.close()

        if self.innovators:
            return node_covered, sorted(node_covered,
                                        key=lambda k: (node_covered[k][1], node_covered[k][2])), product_covered
        else:
            return node_covered, sorted(node_covered,
                                        key=lambda k: (-node_covered[k][1], node_covered[k][2])), product_covered

    @staticmethod
    def __weight_coverage_test(sel, cov):
        """

        @param sel:
        @param cov:
        @return:
        """
        if len(sel) == 0:
            return False
        for i, v in sel.items():
            if v < cov[i]:
                return False
        return True

    def execute(self):

        class_type = "hit"
        if not self.innovators:
            class_type = "flop"

        # nodi coperti con relativi ht e hf, ordinamento nodi da selezionare, insieme degli hit da coprire (sono tutti perchè per ora sto considerando tutti gli HitSavvy)
        n, sn, products = self.__read_data()

        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""DROP TABLE IF EXISTS stats_%s """ % class_type)
        conn.commit()
        conn.execute("""CREATE TABLE IF NOT EXISTS stats_%s
                                               (min_redundancy REAL NOT NULL,
                                               min_coverage      REAL NOT NULL,
                                               actual_coverage REAL NOT NULL,
                                               goods INTEGER NOT NULL,
                                               adopters INTEGER NOT NULL
                                               );""" % class_type)
        conn.commit()

        for min_redundancy in tqdm.tqdm(range(1, 11)):
            min_redundancy = float(min_redundancy) / 10
            for min_products_coverage in range(1, 11):
                min_products_coverage = float(min_products_coverage) / 10
                conn.execute(
                    """DROP TABLE IF EXISTS res_%s_%s_%s """ % (class_type, str(min_redundancy).replace(".", ""),
                                                                str(min_products_coverage).replace(".", "")))
                conn.commit()
                query = "CREATE TABLE IF NOT EXISTS res_%s_%s_%s (adopter TEXT NOT NULL);" \
                        % (class_type, min_redundancy, min_products_coverage)

                query = query.replace(".", "")

                conn.execute(query)
                conn.commit()

                product_coverage = {}  # percentuale beta di incoming edges per ogni hit da coprire
                for p in products:
                    product_coverage[p] = int(math.ceil(
                        len(products[p]) * min_redundancy))  # len(products[p])=len(adopter_of_p) => incoming edges in p

                node_selected = []  # nodi coperti
                product_selected = {}  # insieme degli oggetti coperti
                for i in sn:  # itero sull'ordinamento dei nodi
                    # se non ho raggiunto la percentuale apha di oggetti da coprire oppure per l'ultimo oggetto
                    # coperto non ho raggiunto la percentuale beta di incoming edges
                    if len(product_selected) < (min_products_coverage * len(products)) or \
                            not self.__weight_coverage_test(product_selected, product_coverage):

                        # inserisco nell'insieme dei nodi coperti, il nodo corrente dall'ordinamento
                        node_selected.append(i)
                        query = """INSERT INTO res_%s_%s_%s (adopter) VALUES ('%s')""" % \
                                (class_type, min_redundancy, min_products_coverage, i)
                        query = query.replace(".", "")
                        conn.execute(query)

                        # inserisco  nell'insieme degli oggetti coperti tutto gli oggetti dal nodo corrente
                        for pp in n[i][0]:
                            if pp not in product_selected:
                                product_selected[pp] = 1
                            else:
                                product_selected[pp] += 1
                min_red = str(min_redundancy).replace(".", "")
                min_pc = str(min_products_coverage).replace(".", "")
                if len(products.keys()) > 0:
                    conn.execute("""INSERT INTO stats_%s (min_redundancy, min_coverage, actual_coverage, goods, adopters)
                              VALUES (%s, %s, %f, %d, %d)"""
                                 % (class_type, min_red, min_pc,
                                    float(len(product_selected.keys())) / len(products.keys()),
                                    len(product_selected.keys()), len(node_selected)))
                else:
                    conn.execute("""INSERT INTO stats_%s (min_redundancy, min_coverage, actual_coverage, goods, adopters)
                        VALUES (%s, %s, %f, %d, %d)"""
                                 % (class_type, min_red, min_pc,
                                    0,
                                    len(product_selected.keys()), len(node_selected)))
                conn.commit()
        conn.close()


class Indicators(object):

    def __init__(self, database_name, hits_train, flops_train, slots):
        self.database_name = database_name
        self.hits_train = hits_train
        self.flops_train = flops_train
        self.slots = slots

        self.parameter = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

        hits, flops = self.__adoption_volumes()  # tot_playcounts_hit tot_playcounts_flops
        tot = hits + flops  # tot playcounts
        self.start_hits = int(float(hits) / tot * 10)
        self.start_flops = int(float(flops) / tot * 10)
        diff = max(self.start_hits, self.start_flops) - min(self.start_hits, self.start_flops)
        if self.start_hits < self.start_flops:
            self.start_hits = 5
            self.start_flops = 5 + diff
        elif self.start_hits > self.start_flops:
            self.start_flops = 5
            self.start_hits = 5 + diff
        else:
            self.start_hits = 5
            self.start_flops = 5

        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""DROP TABLE IF EXISTS model""")
        conn.commit()
        conn.execute("""CREATE TABLE IF NOT EXISTS model
                      (category TEXT NOT NULL,
                      table_name TEXT NOT NULL,
                      threshold REAL NOT NULL
                      );""")
        conn.commit()
        conn.close()

    def hitters(self):

        conn = sqlite3.connect("%s" % self.database_name)

        cur = conn.cursor()
        adoptions = cur.execute("""SELECT * from adoptions ORDER BY slot ASC;""")  # [(g0,a0,s,q) ... , (gN,aM,s,q)]

        good_to_adoprs = {}  # {g0:{a0:None, ... , aM:None}... , gN:{aY:None, ..., aX:None}}}
        for adoption in adoptions:
            if int(adoption[2]) <= self.slots:  # se l'oggetto e' stato adottato durante la durata del TRS
                if adoption[0] in good_to_adoprs:
                    good_to_adoprs[adoption[0]][adoption[1]] = None
                else:
                    good_to_adoprs[adoption[0]] = {adoption[1]: None}

        good_to_adoptions = {}  # {g0: #_adopters_TRS , ... , gN: #_adopters_TRS}
        for g, k in good_to_adoprs.items():
            good_to_adoptions[g] = len(k)

        good_to_adopters = {}  # {h0: [a0, ..., aM] , ... , hN: [aX, ... , aY]}
        for row in self.hits_train:
            if row not in good_to_adopters:
                good_to_adopters[row] = []
                results = conn.execute("""select * from adoptions where good='%s' and slot<=%d""" % (row, self.slots))
                for r in results:
                    if r[1] not in good_to_adopters[row]:  # self-addes otherwise the same adopter is counted muliple
                        good_to_adopters[row].append(r[1])

        out_file, selected_threshold = self.__alpha_beta(good_to_adopters, good_to_adoptions, hits=True)

        conn.execute("""INSERT INTO model (category, table_name, threshold) VALUES ('hit', '%s', %f);""" %
                     (out_file, selected_threshold))
        conn.commit()
        conn.close()

    def floppers(self):

        conn = sqlite3.connect("%s" % self.database_name)

        cur = conn.cursor()
        adoptions = cur.execute("""SELECT * from adoptions ORDER BY slot ASC;""")

        good_to_adoprs = {}
        for adoption in adoptions:
            if int(adoption[2]) <= self.slots:
                if adoption[0] in good_to_adoprs:
                    good_to_adoprs[adoption[0]][adoption[1]] = None
                else:
                    good_to_adoprs[adoption[0]] = {adoption[1]: None}

        good_to_adoptions = {}
        for g, k in good_to_adoprs.items():
            good_to_adoptions[g] = len(k)

        good_to_adopters = {}
        for row in self.flops_train:
            if row not in good_to_adopters:
                good_to_adopters[row] = []
                results = conn.execute("""select * from adoptions where good='%s' and slot<=%d""" % (row, self.slots))
                for r in results:
                    if r[1] not in good_to_adopters[row]: # self-addes otherwise the same adopter is counted muliple
                        good_to_adopters[row].append(r[1])

        out_file, selected_threshold = self.__alpha_beta(good_to_adopters, good_to_adoptions, hits=False)

        conn.execute("""INSERT INTO model (category, table_name, threshold) VALUES ('flop', '%s', %f);""" %
                     (out_file, selected_threshold))
        conn.commit()
        conn.close()

    def __alpha_beta(self, good_to_adopters, good_to_adoptions, hits=True):

        conn = sqlite3.connect("%s" % self.database_name)

        selected_threshold = 0
        best_fit = ""

        for alpha in self.parameter:
            for beta in self.parameter:
                adopters = {}

                if hits:
                    table_name = "res_hit_%s_%s" % (alpha, beta)
                else:
                    table_name = "res_flop_%s_%s" % (alpha, beta)

                table_name = table_name.replace(".", "")
                filtered_adopters = conn.execute("""SELECT * from %s;""" % table_name)

                for adopter in filtered_adopters:
                    adopters[adopter[0]] = None

                detail_table_name = "%s_goods" % table_name
                detail_table_name.replace(".", "")
                conn.execute("""DROP TABLE IF EXISTS  %s""" % detail_table_name)
                conn.commit()
                query = "CREATE TABLE IF NOT EXISTS %s (good TEXT NOT NULL, adoptions INTEGER NOT NULL DEFAULT 0);" % detail_table_name

                conn.execute(query)

                goods = {}  # {h_0: #_Indicators_first_month, ... , h_m:#_Indicators_first_month}
                for good in good_to_adopters:  # iterate over the hit's adopters in the first month
                    control = False
                    for adopter in good_to_adopters[good]:
                        if adopter in adopters:  # if current indicator has adopted the hit in the first month
                            control = True
                            if good not in goods:
                                goods[good] = 1
                            else:
                                goods[good] += 1
                    if control:
                        if good_to_adoptions[good] == 0:
                            conn.execute(
                                """INSERT INTO %s (good, adoptions) VALUES ('%s', 0)""" % (detail_table_name, good))
                        else:
                            conn.execute(
                                """INSERT INTO %s (good, adoptions) VALUES ('%s', %f)""" % (
                                    detail_table_name, good, float(goods[good]) / good_to_adoptions[good]))
            conn.commit()

        error = sys.maxsize
        for alpha in self.parameter:
            for beta in self.parameter:

                if hits:
                    detail_table_name = "res_hit_%s_%s_goods" % (alpha, beta)
                else:
                    detail_table_name = "res_flop_%s_%s_goods" % (alpha, beta)

                goods = conn.execute("""SELECT * from %s""" % detail_table_name)

                app = {}  # {h_0: %_Indicators_TRS, ... , h_m: %_Indicators_TRS}
                for good in goods:
                    app[good[0]] = float(good[1])

                if len(app) != 0:
                    arr = np.array(list(app.values()))

                    threshold, mean_error = self.__percentile_selection(arr, 10, hits=hits)
                    print(mean_error, error, detail_table_name)

                    if mean_error <= error:
                        error = mean_error
                        if hits:
                            best_fit = "res_hit_%s_%s" % (alpha, beta)
                        else:
                            best_fit = "res_flop_%s_%s" % (alpha, beta)
                        best_fit = best_fit.replace(".", "")
                        selected_threshold = threshold
        conn.close()

        return best_fit, selected_threshold

    def __percentile_selection(self, vals, k, hits=True):
        lv = len(vals)
        x = int(float(lv) / k)

        thresholds = {}

        if hits:
            start = self.start_hits
        else:
            start = self.start_flops

        values = [float("{0:.2f}".format(l)) for l in vals]

        for p in range(start - 1 + 5, 90, start):
            z = x
            thresholds[p] = []
            while z < lv:
                if x > 0:
                    train = values[:x]
                else:
                    train = [values[0]]
                test = values[x:]
                pt = float("{0:.2f}".format(np.percentile(train, p)))

                FP = 0
                for v in test:
                    if v < pt:
                        FP += 1

                    thresholds[p].append((float(FP) / len(test)) - p)

                if x > 0:
                    z += x
                else:
                    z += 1

        for k, v in thresholds.items():
            thresholds[k] = np.mean(v)

        thresholds = sorted(thresholds.items(), key=operator.itemgetter(1))
        t = float("{0:.2f}".format(np.percentile(values, thresholds[0][0])))
        return t, float("{0:.2f}".format(thresholds[0][1]))


    def __adoption_volumes(self):
        conn = sqlite3.connect("%s" % self.database_name)

        flop_adoptions_volume = 0
        hits_adoptions_volume = 0
        for g in self.hits_train:
            cur = conn.cursor()
            cur.execute("""SELECT adopter, quantity FROM adoptions WHERE good='%s' and slot <= %d""" %
                        (g, self.slots))
            res = cur.fetchall()
            if res is not None:
            	for elem in res:
            		hits_adoptions_volume += int(elem[1])

        for g in self.flops_train:
            cur = conn.cursor()
            cur.execute("""SELECT adopter, quantity FROM adoptions WHERE good='%s' and slot <= %d""" %
                        (g, self.slots))
            res = cur.fetchall()
            if res is not None:
            	for elem in res:
            		flop_adoptions_volume += int(elem[1])

        cur.close()
        conn.close()
        return hits_adoptions_volume, flop_adoptions_volume


class PredictAndEvaluate(object):

    def __init__(self, database_name):
        self.database_name = database_name
        self.thresholds, self.adopters = self.__get_model()  # thresholds={hits:m , flops:n} adopters={hits:{hitters}, flops:{floppers}}

    @staticmethod
    def __rawincount(filename):
        f = open(filename, 'rb')
        buf_gen = takewhile(lambda x: x, (f.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in buf_gen)

    @staticmethod
    def get_hit_flop(hit_flop_map):
        """
        Configure the hit/flop training split

        :param hit_flop_map:
        :return:
        """
        f = open(hit_flop_map)
        hits_test, flops_test = [], []
        for l in f:
            l = l.rstrip().split(",")
            if int(l[1]) == -1:
                flops_test.append(l[0])
            else:
                hits_test.append(l[0])
        return hits_test, flops_test

    def __get_model(self):
        conn = sqlite3.connect("%s" % self.database_name)
        curr = conn.cursor()
        curr.execute("""SELECT * from model""")
        models = curr.fetchall()

        thresholds = {}
        adopters = {}
        for m in models:
            thresholds[m[0]] = m[2]
            curr.execute("SELECT adopter from %s" % m[1])
            ads = curr.fetchall()
            adopters[m[0]] = {a[0]: None for a in ads}

        curr.close()
        conn.close()

        return thresholds, adopters

    def simple_classifier(self, test_set_start_edge_this_function, slots):
        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""DROP TABLE IF EXISTS predictions;""")
        conn.commit()
        conn.execute("""CREATE TABLE predictions
               (good TEXT  NOT NULL,
               success INTEGER NOT NULL DEFAULT 0
               );""")  # FUTURE_HIT == 1,  FUTURE_FLOP == -1, NOT_POSSIBLE_TO_PREDICT == 0
        conn.commit()

        cur = conn.cursor()
        cur.execute("""SELECT distinct adopter from adoptions""")  # I'm using TRS table
        adopters = cur.fetchall()

        indicators = []
        for ad in adopters:
            ad = ad[0]
            curr = conn.cursor()
            curr.execute("""SELECT min(slot) from adoptions where adopter='%s'""" % ad)  # I'm using TRS table
            first_week = curr.fetchall()

            # check if adopter its in Training Set (has made his first adoption in the TRS)
            # and so we can use its HF-propensity to make predictions
            if int(first_week[0][0]) < int(test_set_start_edge_this_function):
                if ad not in indicators:
                    indicators.append(ad)

        cur = conn.cursor()
        cur.execute("""SELECT distinct good from adoptions""")  # I'm using TRS table
        goods = cur.fetchall()

        goods_in_TS = []
        for good in tqdm.tqdm(goods):
            curr = conn.cursor()
            curr.execute("""SELECT min(slot) from adoptions where good='%s'""" % good[0])  # I'm using TRS table
            first_week = curr.fetchall()

            if int(test_set_start_edge_this_function) <= int(first_week[0][0]):  # good its in Test Set
                if good[0] not in goods_in_TS:
                    goods_in_TS.append(good[0])

        for g in tqdm.tqdm(goods_in_TS):
            curr = conn.cursor()
            # retrieve good's adoptions during the predictive period (expressed by slots)
            curr.execute(
                """SELECT distinct adopter from adoptions_test where good='%s' and slot<='%d' """ % (g, int(slots)))
            good_adopters_in_TS = curr.fetchall()

            good_hf_counter = 0.0
            counter = len(good_adopters_in_TS)
            for ad in good_adopters_in_TS:
                ad = ad[0]
                # check if the adopeter is an indicator present in TRS or not
                if ad in indicators:
                    # retrive infocator's HF-propensity (> 0.0 he is a Hitter, < 0.0 he is a Flopper, = 0.0 neutral)
                    curr = conn.cursor()
                    curr.execute("""SELECT value from HFscore where adopter='%s'""" % ad)  # I'm using TRS table
                    res = curr.fetchall()
                    try:
                        ad_hf = float(res[0][0])
                        good_hf_counter += ad_hf
                    except Exception:
                        continue  # indicator is neutral => omitt him

            try:
                preditiction = good_hf_counter / counter
            except ZeroDivisionError:
                preditiction = 0.0
            except Exception:
                preditiction = 0.0

            if preditiction > 0.0:
                conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % g)
            elif preditiction < 0.0:
                conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % g)
            else:  # preditiction == 0.0
                conn.execute("""INSERT INTO predictions (good, success) values ('%s', 0)""" % g)
        conn.commit()
        conn.close()

    def predict(self, test_set_start_edge_this_function, slots):
        conn = sqlite3.connect("%s" % self.database_name)
        conn.execute("""DROP TABLE IF EXISTS predictions;""")
        conn.commit()
        conn.execute("""CREATE TABLE predictions
            (good TEXT  NOT NULL,
            success INTEGER NOT NULL DEFAULT 0
            );""")  # FUTURE_HIT == 1,  FUTURE_FLOP == -1, NOT_POSSIBLE_TO_PREDICT == 0
        conn.commit()

        goods_in_TS = []
        predictions = {}

        cur = conn.cursor()
        cur.execute("""SELECT distinct good from adoptions""")  # I'm using TRS table
        goods = cur.fetchall()

        for good in goods:
            curr = conn.cursor()
            curr.execute("""SELECT min(slot) from adoptions where good='%s'""" % good[0]) # I'm using TRS table
            first_week = curr.fetchall()

            if int(test_set_start_edge_this_function) <= int(first_week[0][0]):  # good its in Test Set
                if good[0] not in goods_in_TS:
                    goods_in_TS.append(good[0])

        for good in goods_in_TS:
            curr = conn.cursor()
            # retrieve good's adoptions during the predictive period (expressed by slots)
            curr.execute("""SELECT * from adoptions_test where good='%s' and slot<='%d' """ % (good, int(slots)))
            good_adoptions = curr.fetchall()
            for ad in good_adoptions:
                if ad[0] not in predictions:  # keep track of every item in TS if it is adoptes by a hitters or a floppers
                    predictions[ad[0]] = {}
                    predictions[ad[0]]["numero_adozioni_tot"] = 0
                    predictions[ad[0]]["adozioni_innovatori"] = 0
                    predictions[ad[0]]["adozioni_flop"] = 0
                if ad[1] in self.adopters['hit']:
                    predictions[ad[0]]["adozioni_innovatori"] += 1
                elif ad[1] in self.adopters['flop']:
                    predictions[ad[0]]["adozioni_flop"] += 1
                predictions[ad[0]]["numero_adozioni_tot"] += 1

        count = 0
        for g in predictions:
            if predictions[g]["adozioni_innovatori"] > 0 or predictions[g]["adozioni_flop"] > 0:
                print(predictions[g], count)
                count += 1

        for good in predictions:
            # print predictions[good]
            if float(predictions[good]["numero_adozioni_tot"]) >= 0:
                percentage_hitters = float("{0:.3f}".format(float(predictions[good]["adozioni_innovatori"]) / \
                                                            float(predictions[good]["numero_adozioni_tot"])))
                percentage_floppers = float("{0:.3f}".format(float(predictions[good]["adozioni_flop"]) / \
                                                             float(predictions[good]["numero_adozioni_tot"])))

                if percentage_hitters >= self.thresholds['hit'] and percentage_floppers < self.thresholds['flop']:
                    conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                elif percentage_floppers >= self.thresholds['flop'] and percentage_hitters < self.thresholds['hit']:
                    conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                elif percentage_floppers >= self.thresholds['flop'] and percentage_hitters >= self.thresholds['hit']:
                    if self.thresholds['flop'] > 0 and self.thresholds['hit'] > 0:
                        dist_flop = percentage_floppers - self.thresholds['flop']
                        dist_hit = percentage_hitters - self.thresholds['hit']
                        if dist_flop >= dist_hit:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                        else:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                    elif self.thresholds['flop'] == 0 and self.thresholds['hit'] == 0:
                        if percentage_floppers > percentage_hitters:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                        elif percentage_floppers < percentage_hitters:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                        else:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 0)""" % good)
                    else:
                        if (self.thresholds['flop'] == 0 and
                            percentage_floppers >= (percentage_hitters - self.thresholds['hit'])) or \
                                (self.thresholds['hit'] == 0 and percentage_hitters < (
                                        percentage_floppers - self.thresholds['flop'])):
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', -1)""" % good)
                        elif (self.thresholds['flop'] == 0 and percentage_floppers < (
                                percentage_hitters - self.thresholds['hit'])) or (
                                self.thresholds['hit'] == 0 and percentage_hitters >= (
                                percentage_floppers - self.thresholds['flop'])):
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 1)""" % good)
                        else:
                            conn.execute("""INSERT INTO predictions (good, success) values ('%s', 0)""" % good)
                else:
                    conn.execute("""INSERT INTO predictions (good, success) values ('%s', 0)""" % good)

        conn.commit()
        conn.close()

    def evaluate(self, ground_truth_file):

        gt_hits, gt_flops = self.get_hit_flop(ground_truth_file)

        conn = sqlite3.connect("%s" % self.database_name)
        cur = conn.cursor()
        cur.execute("""SELECT * from predictions""")
        predictions = cur.fetchall()

        TP, FP, FN, TN = 0, 0, 0, 0  # TP=tot_hit_predicted FP=hit_predicted_but_NOT_hit, TN=tot_flop_predicted, FN=flop_predicted_but_NOT_flop
        unclassified = 0
        total = 0

        for res in predictions:

            total += 1
            if int(res[1]) == 0:
                unclassified += 1
            elif int(res[1]) == 1:
                if res[0] in gt_hits:
                    TP += 1
                else:
                    FP += 1
            else:  # int(res[1]) == -1
                if res[0] in gt_flops:
                    TN += 1
                else:
                    FN += 1

        if TP + FP != 0:
            precision = float(TP) / float(TP + FP)
        else:
            precision = 0.0

        if TP + FN + unclassified != 0:
            frecall = float(TP) / float(TP + FN + unclassified)
        else:
            frecall = 0.0

        if float(total) != 0:
            accuracy = float(TP + TN) / float(total)
        else:
            accuracy = 0.0

        if FN + TN != 0:
            NPV = float(TN) / float(FN + TN)
        else:
            NPV = 0.0

        if TP + FN != 0:
            recall = float(TP) / float(TP + FN)
        else:
            recall = 0.0

        if FP + TN != 0:
            specificity = float(TN) / float(FP + TN)
        else:
            specificity = 0.0

        if FP + TN != 0:
            un_specificity = float(TN) / float(FP + TN + unclassified)
        else:
            un_specificity = 0.0

        if TP + FP + FN != 0:
            F1 = 2 * float(TP) / float(2 * TP + FP + FN)
        else:
            F1 = 0

        conn.execute("""DROP TABLE IF EXISTS prediction_results;""")
        conn.execute("""CREATE TABLE prediction_results (score TEXT NOT NULL, value REAL NOT NULL);""")
        conn.commit()

        conn.execute("""INSERT INTO prediction_results (score, value) VALUES ('precision', %f)""" % precision)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('recall (with unclassified)', %f)""" % frecall)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('recall (without unclassified)', %f)""" % recall)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('accuracy', %f)""" % accuracy)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('specificity (with unclassified)', %f)""" % un_specificity)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('specificity (with unclassified)', %f)""" % specificity)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('NPV', %f)""" % NPV)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('F1', %f)""" % F1)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('TP', %d)""" % TP)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('FP', %d)""" % FP)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('TN', %d)""" % TN)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('FN', %d)""" % FN)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('unclassified', %d)""" % unclassified)
        conn.execute(
            """INSERT INTO prediction_results (score, value) VALUES ('total', %d)""" % total)

        conn.commit()
        conn.close()


if __name__ == "__main__":

    success_def = -1  # insert success definition's number
    ground_truth_file = "info_data/"  # insert success definition's ground truth file
    
    main_database = "lastfm.db"
    database_name = "success_def" + str(success_def) + ".db"

    print("Computing early adopters")
    e = EarlyAdoptersThreshold(database_name)
    e.load_splitted_db_data(main_database, database_name, 26, 103)
    hits_train_set, flops_train_set = e.set_hit_flop(ground_truth_file)
    goods_thresholds = e.execute()
    e.check_if_directories_exist_otherwise_create_them()
    
    # Compute HFpropensity scores
    print("Computing-HF Propensity")
    hf = HFPropensity(database_name, hits_train_set, flops_train_set, goods_thresholds)
    hf.execute()

    # Coverage
    print("WMSC hits")
    w = WMSC(database_name, True)
    w.execute()

    print("WMSC flops")
    w = WMSC(database_name, False)
    w.execute()

    conn = sqlite3.connect("%s" % database_name)
    curr = conn.cursor()
    curr.execute("""SELECT * from HFscore""")
    res = curr.fetchall()

    hit_savvies = {}
    flop_adopters = {}
    for elem in res:
        ad = elem[0]
        hf = float(elem[1])
        if hf > 0.0:
            hit_savvies[str(ad)] = {"value": hf}
            hit_savvies[str(ad)]["hits"] = int(elem[2])
            hit_savvies[str(ad)]["flops"] = int(elem[3])
            curr = conn.cursor()
            curr.execute("""SELECT good from Coverage where adopter='%s'""" % ad)
            result = curr.fetchall()
            goods_list = []
            for g in result:
                g = g[0]
                if g not in goods_list:
                    goods_list.append(g)
            hit_savvies[str(ad)]["goods"] = goods_list
        else:
            flop_adopters[str(ad)] = {"value": hf}
            flop_adopters[str(ad)]["hits"] = int(elem[2])
            flop_adopters[str(ad)]["flops"] = int(elem[3])
            curr = conn.cursor()
            curr.execute("""SELECT good from Coverage where adopter='%s'""" % ad)
            result = curr.fetchall()
            goods_list = []
            for g in result:
                g = g[0]
                if g not in goods_list:
                    goods_list.append(g)
            flop_adopters[str(ad)]["goods"] = goods_list

    curr.close()
    conn.close()
    hit_savvies_file = "hit_savvies_success_def" + str(success_def) + ".json"
    with open(hit_savvies_file, 'w', encoding='utf-8') as out:
        json.dump(hit_savvies, out, indent=4)
    flop_adopter_file = "flop_adopters_success_def" + str(success_def) + ".json"
    with open(flop_adopter_file, 'w', encoding='utf-8') as out:
        json.dump(flop_adopters, out, indent=4)

    e.compute_active_period(hits_train_set, flops_train_set, goods_thresholds, success_def, True)
    e.compute_active_period(hits_train_set, flops_train_set, goods_thresholds, success_def, False)

    for i in range(1, 6):
        training_set_start_edge = 12*i  # TRS starts from 3 months (3*4 weeks) => in reality it is training_set_end_edge
        test_set_start_edge = training_set_start_edge + 1

        TRS_encoding = ""
        if training_set_start_edge == 12:
            TRS_encoding = "a"
        elif training_set_start_edge == 24:
            TRS_encoding = "b"
        elif training_set_start_edge == 36:
            TRS_encoding = "c"
        elif training_set_start_edge == 48:
            TRS_encoding = "d"
        elif training_set_start_edge == 60:
            TRS_encoding = "e"
        else:
            print("error with training_set_start_edge = " + str(training_set_start_edge))
            sys.exit(-1)

        slots = training_set_start_edge  # I consider only goods in TRS's HF-scores

        # Model construction
        print("Indicators selection")
        ids = Indicators(database_name, hits_train_set, flops_train_set, slots)
        print("Hitters")
        ids.hitters()

        print("Floppers")
        ids.floppers()
        print("Model generated")

        conn = sqlite3.connect("%s" % database_name)
        curr = conn.cursor()
        curr.execute("""SELECT * from model""")
        models = curr.fetchall()

        adopters = {}
        adopters_info = {}
        for m in models:
            curr.execute("SELECT adopter from %s" % m[1])
            ads = curr.fetchall()
            adopters[str(m[0])] = {a[0]: None for a in ads}

            # keep track of trashhold
            adopters_info[str(m[0])] = {"threshold": m[2]}

            # retrive alpha and beta: m[1] = res_hit_alpha_beta or res_flop_alpha_beta
            str_array = str(m[1]).split("_")
            alpha = int(str_array[2])
            beta = int(str_array[3])
            adopters_info[str(m[0])]["alpha"] = alpha
            adopters_info[str(m[0])]["beta"] = beta

            # keep track of general info
            stats_table = "stats_" + str(m[0])
            curr.execute("""SELECT * from '%s' where min_redundancy='%d' and min_coverage='%d'"""
                                                                % (stats_table, alpha, beta))
            res = curr.fetchall()
            for elem in res:
                adopters_info[str(m[0])]["actual_coverage"] = elem[2]
                adopters_info[str(m[0])]["goods"] = elem[3]
                adopters_info[str(m[0])]["adopters"] = elem[4]
                # check if I'm taking the right info
                if int(elem[4]) != len(list(adopters[m[0]].keys())):
                    print("Missmatch with alpha and beta, model |" + str(m[0]) + "| = " + str(m[1]))
                    sys.exit(-1)
            # keep track of covered goods
            if int(alpha) <= 9:
                new_alpha = str(0) + str(alpha)
            else:
                new_alpha = str(alpha)
            if int(beta) <= 9:
                new_beta = str(0) + str(beta)
            else:
                new_beta = str(beta)
            goods_table = "res_" + str(m[0]) + "_" + new_alpha + "_" + new_beta + "_goods"
            curr.execute("""SELECT * from '%s'""" % (goods_table))
            res = curr.fetchall()
            adopters_info[str(m[0])]["goods"] = []
            adopters_info[str(m[0])]["adoptions"] = []
            for elem in res:
                adopters_info[str(m[0])]["goods"].append(elem[0])
                adopters_info[m[0]]["adoptions"].append(elem[1])
        curr.close()
        conn.close()
        hit_model_file = "hit_model/hit_model_success_def" + str(success_def) + "_" + str(TRS_encoding) + ".json"
        with open(hit_model_file, 'w', encoding='utf-8') as out:
            json.dump(adopters["hit"], out, indent=4)
        flop_model_file = "flop_model/flop_model_success_def" + str(success_def) + "_" + str(TRS_encoding) + ".json"
        with open(flop_model_file, 'w', encoding='utf-8') as out:
            json.dump(adopters["flop"], out, indent=4)

        hit_model_file = "hit_model/hit_model_info_success_def" + str(success_def) + "_" + str(TRS_encoding) + ".json"
        with open(hit_model_file, 'w', encoding='utf-8') as out:
            json.dump(adopters_info["hit"], out, indent=4)
        flop_model_file = "flop_model/flop_model_info_success_def" + str(success_def) + "_" + str(TRS_encoding) + ".json"
        with open(flop_model_file, 'w', encoding='utf-8') as out:
            json.dump(adopters_info["flop"], out, indent=4)

        # Load Adoption log Data
        print("Loading data for prediction")
        l = PredictAndEvaluate(database_name)

        """print("Simple Classifier")
        for i in tqdm.tqdm(range(1, 13)):
            slots = 4 * i

            # it doesn't make sense to analyze good's adoptions for a period longer than TS itself
            # (TS = 78 - training_set_start_edge) with 78 tot. num. weeks
            if slots > (78 - training_set_start_edge):
                break

            # Simple Classifier (a new item is a hit if the sum of its adopters' HF-propensity is > 0)
            l.simple_classifier(test_set_start_edge, slots)
            l.evaluate(ground_truth_file)

            conn = sqlite3.connect("%s" % database_name)
            cur = conn.cursor()
            cur.execute(SELECT * from prediction_results)
            predict_results = cur.fetchall()
            info = {}
            for score, value in predict_results:
                info[str(score)] = value

            info_file = "simple_classifier/simple_classifier_success_def" + str(success_def) + "_" + str(TRS_encoding) \
                        + "_" + str(slots) + ".json"
            with open(info_file, 'w', encoding='utf-8') as out:
                json.dump(info, out, indent=4)"""

        print("Predict")
        for i in tqdm.tqdm(range(1, 13)):
            slots = 4 * i

            # it doesn't make sense to analyze good's adoptions for a period longer than TS itself
            # (TS = 78 - training_set_start_edge) with 78 tot. num. weeks
            if slots > (78 - training_set_start_edge):
                break

            l.predict(test_set_start_edge, slots)
            l.evaluate(ground_truth_file)

            conn = sqlite3.connect("%s" % database_name)
            curr = conn.cursor()
            curr.execute("""SELECT * from prediction_results""")
            predict_results = curr.fetchall()
            info = {}
            for score, value in predict_results:
                info[str(score)] = value

            info_file = "prediction/prediction_success_def" + str(success_def) + "_" + str(TRS_encoding) \
                        + "_" + str(slots) + ".json"
            with open(info_file, 'w', encoding='utf-8') as out:
                json.dump(info, out, indent=4)
        curr.close()
        conn.close()
