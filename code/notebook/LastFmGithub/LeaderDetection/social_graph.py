from pathlib import Path
import json
import gzip
from tqdm import tqdm
from collections import defaultdict

"""
    LEGEND:
        1. data/network_full1.edgelist => global network file
        2. data/new_ids => users not crawled in DataSets  (NOT USED)
        3. data/network_filtered1.edgelist => network with only friendship's bond between the users analyzed in the Datasets
        4. data/node_map1 => usernames' encoding file
        5. data/network_filtered_int1.edgelist =>  data/network_filtered1.edgelist encoded
        6. data/unknown_ranks => users not crawled in DataSets and which are at least friends with 100 users we crawled
"""

"""
   Function which construct the global network (friendship bond) file.
   To construct the global file we have to iterate over the Datasets and for each
   user analyzed here, we have to extract their friends' data from their
   user_info.file
"""
def graph_extracion():
    with open("data/network_full1.edgelist", 'w') as out:
        # iterate over the 56 DataSets
        for fl in range(0, 56):
            # get all the users' files present in the DataSet{fl}/data/users_info/'s directory
            # glob = module which finds all the pathnames matching a specified pattern
            pathlist = Path(f"/data/users/bradan/Lastfm_Bradan/DataSets/DataSet{fl}/data/users_info/").glob("*gz")
            # iterate over the retrieved users' files
            for filename in tqdm(pathlist):
                #user_name = str(filename).split("/")[-1].split(".")[0]
                with gzip.GzipFile(filename, 'r') as fin:
                    data = json.loads(fin.read().decode('utf-8'))
                try:
                    # retrieve user's friends
                    friends = data['friends'].keys()
                    # iterate over user's friends and for each friend write
                    #the user's name and the friend's name on the "data/network_full1.edgelist" file
                    # seprated by a tab ("\t") space
                    for friend in friends:
                        out.write(f"{data['user_id']}\t{friend}\n")
                except:
                    pass
            out.flush()

"""
    Function which filter the global network file, in order to get a network
    with only friendship bond betweek the users analyzed in the Datasets
"""
def filter_edges():
    expanded = {}  # user's analyzed in the DataSets

    # iterate over the global newtork file, in order to retrieve all the
    # usernames analyzed in tha DataSets and to memorize them in the expanded dict
    with open("data/network_full1.edgelist") as f:
        for l in f:
            l = l.rstrip().split("\t")  # get first username of the pair separate by "\t"
            expanded[l[0]] = None

    print(len(expanded))

    unseen = {}  # users not analyzed in the DataSets (they are friends of the user's analyzed)

    with open("data/network_filtered1.edgelist", "w") as o:

        # iterate again over the expanded "data/network_filtered1.edgelist", this time
        # in order to discriminate over the friend:
        # 1. if the friend is an user analyzed in the Datasets, update "data/network_filtered1.edgelist" file
        # 2. elsewhere, add friend to the "unseen" list
        with open("data/network_full1.edgelist") as f:
            for l in f:
                u, v = l.rstrip().split("\t")
                # if both user and friend
                if v in expanded:
                    o.write(f"{u}\t{v}\n")
                else:
                    unseen[v] = None

    # write unseen friends in the "data/new_ids"
    with open("data/new_ids", "w") as o:
        for i in unseen:
            o.write(f"{i}\n")

"""
     Function which encode "data/network_filtered1.edgelist" file (every username and friend from the  network file is 
     encoded with an integer). The file with the mapping (username, integer) is "data/node_map1" and the
     equivalent file of"data/network_filtered1.edgelist",  encoded, is  "data/network_filtered1.edgelist".
"""
def remap_graph():
    nodes = {}
    i = 0

    with open("data/network_filtered1.edgelist") as f:
        with open("data/node_map1", "w") as o:

            for l in f:
                u, v = l.rstrip().split("\t")
                if u not in nodes:
                    nodes[u] = i  # assign current encoding to user "u"
                    o.write(f"{u}\t{i}\n")
                    i+=1
                if v not in nodes:
                    nodes[v] = i  # assign current encoding to friend "v"
                    o.write(f"{v}\t{i}\n")
                    i+=1

    # re-write "data/network_filtered_int1.edgelist" file encoded in the "data/network_filtered1.edgelist"
    with open("data/network_filtered_int1.edgelist", "w") as o:
        with open("data/network_filtered1.edgelist") as f:
            for l in f:
                u, v = l.rstrip().split("\t")
                o.write(f"{nodes[u]}\t{nodes[v]}\n")

"""
    Function which populates the "data/unknown_ranks" file, with the friend's names and their enconuter frequency
    (they have to  be friend with at least 100 users we crawled))
    
"""
def rank_external_ids():
    known_nodes = {}  # hold track of how many encoded users

    with open("data/node_map1") as f:
        for l in f:
            u, _ = l.rstrip().split("\t")
            known_nodes[u] = 0  # assign to crawled username the value 0 (value not used)

    unknown_freq = defaultdict(int)    # deafult dict = have a default value if that key has not been set yet

    # iterate over the global network file, in order to track how many pair (username, friend) contains friends
    # not crawled
    with open("data/network_full1.edgelist") as f:
        for l in f:
            _, v = l.rstrip().split("\t")  # get the pair (username, friend) from the global network file

            # if the friend is not crawled, add him to the unknown_freq and increment his encountered frequency
            if v not in known_nodes:
                unknown_freq[v] += 1  # increment the number of the times we seen him

    # sort unknown_freq dict with:
    # a) keys = not crawled friends
    # b) values = enconutered_frequency
    res = sorted(unknown_freq.items(), key=lambda x: -x[1])

    # write on the "data/unknown_ranks" file only the friend encoutered at list 100 times (he is friend with at least
    # 100 users we crawled)
    with open("data/unknown_ranks", "w") as o:
        for n, v in res:
            if v >= 100:
                o.write(f"{n}\t{v}\n")

#graph_extracion()
#filter_edges()
#remap_graph()

#import networkx as nx
#g = nx.read_edgelist("data/network_filtered_int1.edgelist")
#print(g.number_of_nodes())
#print(nx.is_connected(g), len(list(nx.connected_components(g))[0]), len(list(nx.connected_components(g))[0])/g.number_of_nodes())

rank_external_ids()
