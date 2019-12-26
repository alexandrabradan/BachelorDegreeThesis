import math
import sys
import zipfile

import networkx as nx


class Diffusion(object):
    """
        Leader detection algorithm, based on the three dimensions of social prominence measurements:
        1) Width: ratio of the neighbours of a node, that follows the node's actions
        2) Depth: how many degrees of separation there are between a node and the other nodes
                  that followed its actions
        3) Strenght: intensity of the action performed by some nodes after the leader
    """

    def __init__(self, graph_file):
        """
        constructor
        :param graph_file: edgelist format (network_filtered_int1.edgelist)
        """
        self.__load_data(graph_file)  # construct SOCIAL GRAPH
        self.node2weight = {}  # temporary dicts which contains the weights fort a given action

    def get_listeners_for_artist(self, artist_encoding):
        """
            Function which returns the users which listened for the first time to the artist passed by argument
            :param artist_encoding: artist for which to retireve the users which listened to him for the first time
                                    and how many playcounts in this first time they made
            :return: dict of of users who listened to the artist, their first time of listening and how many playcounts
                            they made
                     empty dict otherwise
        """

        first_week_directory = "first_week/"
        artist_file_name = first_week_directory + str(artist_encoding) + "first_week_artist_count"

        nodes2time = {}
        # re-initialize temporary dict (it contains the data of the previous action in the iteration)
        self.node2weight = {}

        try:
            with open(artist_file_name, 'r') as infile:
                for line in infile:
                    line_array = line.split("::")

                    week_encoding = line_array[0]
                    user_encoding = line_array[1]
                    art_encoding = line_array[2]
                    playcount = line_array[3].replace("\n", "")

                    if int(art_encoding) == int(artist_encoding):
                        nodes2time[str(user_encoding)] = week_encoding
                        self.node2weight[str(user_encoding)] = playcount

        except FileNotFoundError:
            print(str(artist_file_name) + " doesn't exist")
            sys.exit(-999)

        return nodes2time

    def get_total_listenings_for_artist(self, user_encoding, artist_encoding):
        """
            Function which retrieves the total number of listenings that the user passed by argument has made
            for the artist passed by argument
            :param user_encoding: user for which to retrieve artist's total playcounts
            :param artist_encoding: artist for which to retrive total playcounts
            return >= 1 artist's total playcopunts made by the user
                   -1 if the user didn't listen to the artist (strange, it shouldn't happen)
        """
        total_listenings_directory = "total_listenings/"
        user_file_name = total_listenings_directory + str(user_encoding) + "user_artist_totalistening_periodlength"

        total_playcounts = -1

        try:
            with open(user_file_name, 'r') as infile:
                for line in infile:
                    line_array = line.split("::")

                    usr_encoding = line_array[0]
                    art_encoding = line_array[1]
                    tot_playcount = line_array[2]
                    period_lenght = line_array[3]

                    if int(usr_encoding) == int(user_encoding) and int(art_encoding) == int(artist_encoding):
                        total_playcounts = tot_playcount
                        break
        except FileNotFoundError:
            print(str(user_file_name) + " doesn't exist")
            sys.exit(-999)

        return total_playcounts

    def __load_data(self, graph_file):
        """
        Load the social graph graph

        :param graph_file: edgelist format [username \t friend]
        """

        # it's a DIRECTED graph because Last.fm has introduced the distinction between following and followers
        # (I've collected for eah user his followings, so the directed graph rapresents the following graph of the
        # users I have scrapped)
        self.G = nx.DiGraph()

        with zipfile.ZipFile('networks.zip') as z:
            for filename in z.namelist():
                if filename == graph_file:
                    edges = z.open(filename)

                    for e in edges:
                        e = e.decode("utf-8")
                        part = e.split()
                        u = int(part[0])  # username
                        v = int(part[1].strip())  # friend

                        self.G.add_edge(u, v)

                    edges.close()

        self.G.remove_edges_from(self.G.selfloop_edges())  # remove self-loops
        self.G.remove_nodes_from(list(nx.isolates(self.G)))  # remove isolated nodes

    def build_action_subgraph(self, action, delta):
        """
        Compute the DIRECTED induced subgraph for the given action

        :param action: action id (artist's name)
        :param delta: temporal displacement
        """
        nodes2time = self.get_listeners_for_artist(action)  # user's which listened to the artist (for the first time)
        nodes = list(nodes2time.keys())  # get usernames
        for i in range(0, len(nodes)):
            nodes[i] = int(nodes[i])

        # induced subgraph for action
        isg = self.G.subgraph(nodes)

        # build the digraph (DIRECTED graph) from "isg"
        disg = nx.DiGraph()

        for e in isg.edges():  # action's subgraph pair edge list

            seed_user = str(e[0])
            friend = str(e[1])

            # difference between artist's first week of listening (encoded) made by the user and the user he follows
            diff = abs((int(nodes2time.get(seed_user)) - int(nodes2time.get(friend))))

            # among the 2 users, one of them has listened to the artist first (<= the influencial keep alive time)
            if diff != 0 and diff <= delta:
                # if the user that e[0] follows (e[1]) has listened to the artist first,
                #  I add the edge, REVERSED, in order to take trace of the diffusion path
                if int(nodes2time.get(friend)) < int(nodes2time.get(seed_user)):
                    disg.add_edge(int(friend), int(seed_user))

        disg.remove_edges_from(disg.selfloop_edges())  # remove self-loops
        disg.remove_nodes_from(list(nx.isolates(disg)))  # remove isolated nodes

        return disg  # return DIRECTED induced subgraph for the given action

    def compute_action_leaders(self, disg):
        """
        Compute the leaders for the given action

        :param disg: directed graph induced for the specific action
        """
        # compute the leaders for this action
        leaders = []

        for n in disg.nodes():  # action's DIRECTED graph nodes list

            n_ind = disg.in_degree(n)  # in-degree = number of edges pointing in to the node "n"
            n_odg = disg.out_degree(n)  # out-degree = number of edges outgoing from the node "n"

            # node "n" not influenced by anyone, but influencer at least for one user
            if n_ind == 0 and not (n_odg == 0):
                leaders.append(n)  # node "n" is a leader for the action/artist

        return leaders

    def compute_max_depth(self, tree, leader, frontier):
        """
        Compute the maximal depth for the given diffusion tree, going to check each shortest_path from the root to the
        frontier's nodes

        :param tree: minimum action diffusion tree
        :param leader: root of the tree (leader)
        :param frontier: leafs of the tree
        """
        max_depth = 0  # root's depth

        for f in frontier:
            # get shortest path between leader(root) and frontier node f
            l = len(nx.shortest_path(tree, leader, f))  # count num. of nodes (including root)
            if l > max_depth:
                max_depth = l

        return max_depth  # return leader's maximum depth

    def compute_width(self, tree, leader):
        """
        Compute the ratio between leaders neighbors in the full graph and the ones in the diffusion tree

        :param tree: the diffusion tree
        :param leader: leader of the tribe
        """
        leader_neighbors = list(self.G.neighbors(leader))  # get leaders neighbors in the full graph
        tribe_restricted_neighbors = list(tree.neighbors(leader))  # leaders neighbors in the diffusion tree

        # if the leader is a friend of a seed user, for which we didn't collect network info =>
        # we reject him as a leader because he could have incoming-edges that we didn't consider
        if len(leader_neighbors) == 0:
            return -1

        ratio = float(len(tribe_restricted_neighbors)) / float(len(leader_neighbors))

        # return leader's width ration between its neighbors in the diffusion tree
        # and its neighbors in the full graph
        return ratio 

    def compute_strength(self, tree, leader, distance_factor, action):
        """
        Compute the strength for the given diffusion tree, leader and distance factor

        weight = sum_{n \in tree} strength_{n}^{-distanceFactor * shortestPath(leader, n)}

        :param tree: the diffusion tree
        :param leader: root of the tree (leader)
        :param distance_factor: dumping factor
        :param action: artist
        """
        strength = 0
        t_nodes = tree.nodes()  # nodes influenced by the leader

        for n in t_nodes:
            if n is not leader:
                l = len(nx.shortest_path(tree, leader, n))
                w = self.node2weight[action][n]  # get user's "n" artist's weight
                strength += float(w) * math.exp(-distance_factor * (l - 1))

        return strength  # return leader's strenght for this action

    def compute_level_strength(self, tree, leader, distance_factor, action):
        """
        Compute the strength for the given diffusion tree, leader and distance factor

        :param tree: the diffusion tree
        :param leader: root of the tree (leader)
        :param distance_factor: dumping factor
        :param action: artist

        N.B The leader's strenght is computed as:
           sum (distance_factor ^ i * sum(node2weight[action][n] / (sum ( node2weight[n][*])) ) )
                                                            for all nodes n at distance i from l, for i ∈ [0, depth(l))

        """
        strength = 0
        t_nodes = tree.nodes()  # nodes influenced by the leader

        level_to_weight = {}

        for n in t_nodes:
            if n is not leader:
                l = len(nx.shortest_path(tree, leader, n))  # min distance of node "n" from "leader" (root)

                # try to get user's artist playcount in the first week
                try:
                    w = self.node2weight[str(n)]  # n's artist playcount in first week 

                    total_listens = float(
                        self.get_total_listenings_for_artist(n, action))  # n's tot. artist playcount

                    if total_listens == -1:  # n's artist total playcount is missing
                        w = 0.0
                        total_listens = 0.0

                except KeyError:  # the user didn't listen to the artist (it should not happen)
                    w = 0.0
                    total_listens = 0.0

                if not l in level_to_weight:
                    if w == 0.0 or total_listens == 0.0:
                        level_to_weight[l] = 0.0
                    else:
                        level_to_weight[l] = float(w) / total_listens
                else:
                    if w != 0.0 and total_listens != 0.0:  # I skip summing + 0
                        level_to_weight[l] += float(w) / total_listens

        for l in level_to_weight:
            strength += (distance_factor ** l) * level_to_weight[l]

        return strength
