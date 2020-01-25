import os
import shutil
import networkx as nx
import itertools
import matplotlib.pyplot as plt


def semantic_feasibility(subg):
    """
        Function which checks if the induced subgraph passed by argument has as first node, a
        node which have the attribute 'status' equal to "leader"
        :param subg: induced subgraph from a diffusion tree
        :return: True: if the induced subgraph's first node has as atribute 'status' = "leader"
                 False: otherwise
    """
    subg_status_dict = nx.get_node_attributes(subg, 'status')

    try:
        subg_first_value = next(iter(subg_status_dict.values()))
    except StopIteration:  # attribute 'status' not present
        return False

    if subg_first_value == "leader":
        return True
    else:
        return False


def get_vf2_algorithm_result(T, target, type_of_pattern):
    """
        Function which uses the VF2 algorithm (implemented through Networkx's function
        <<nx.is_isomorphic(subg, target)>>>), which given a data graph and a query graph searches
        the presence of the query graph (as a subset) in the data graph in polynomial time
        :param T: data graph
        :param target: query graph
        :param type_of_pattern: type of pattern to search
        :return: True: if the query graph is present in the data graph
                 False: otherwise
    """

    t_nodes = list(T.nodes)
    leader_node = int(t_nodes[0])

    nodes_to_examine = []
    nodes_to_examine.append(leader_node)

    # get leader's neighbours
    leader_neightbours = list(T.neighbors(leader_node))
    for node in leader_neightbours:
        nodes_to_examine.append(int(node))

        if type_of_pattern != "star":
            # get leader's second generation of friends
            leader_second_generation_neightbours = list(T.neighbors(int(node)))
            for s_node in leader_second_generation_neightbours:
                nodes_to_examine.append(int(s_node))

                if type_of_pattern == "chain":
                    # get leader's third generation of friends
                    leader_third_generation_neightbours = list(T.neighbors(int(s_node)))
                    for t_node in leader_third_generation_neightbours:
                        nodes_to_examine.append(int(t_node))

    for sub_nodes in itertools.combinations(nodes_to_examine, len(target.nodes())):
        subg = T.subgraph(sub_nodes)

        #  I have to check if the induced subgraph's first node has the attribute 'status' = "leader",
        #  in order to discriminate only the induced subgraphs which involve the leader
        if nx.is_isomorphic(subg, target) and semantic_feasibility(subg):
            return True  # I need to find just one graphlet (if present) in the diffusion tree

    return False


def compute_star_pattern(T):
    """
        Function which given the diffusion tree passed by argument, computes using the VF2 algorithm,
        the following pattern: presence in the directed graph of three neighbors influenced by the
        leader
        :param T: diffusion leader (which belongs to the first node which appears in the nodes list)
        :return True: if the pattern is present
                False: otherwise
    """
    target = nx.DiGraph()
    target.add_edge('a', 'b')
    target.add_edge('a', 'c')
    target.add_edge('a', 'd')

    return get_vf2_algorithm_result(T, target, "star")


def compute_chain_pattern(T):
    """
        Function which given the diffusion tree passed by argument, computes using the VF2 algorithm,
        the following pattern: presence if the directed graph of a chain, where each node is prominent
        for (at least) one neighbor
        :param T: diffusion leader (which belongs to the first node which appears in the nodes list)
        :return True: if the pattern is present
                False: otherwise
    """
    target = nx.DiGraph()
    target.add_edge('a', 'b')
    target.add_edge('b', 'c')
    target.add_edge('c', 'd')

    return get_vf2_algorithm_result(T, target, "chain")


def compute_split_pattern(T):
    """
        Function which given the diffusion tree passed by argument, computes using the VF2 algorithm,
        the following pattern: presence if the directed graph of a split, where the leader is prominent
        for a node, which itself is prominent for two other neighbors
        :param T: diffusion leader (which belongs to the first node which appears in the nodes list)
        :return True: if the pattern is present
                False: otherwise
    """
    target = nx.DiGraph()
    target.add_edge('a', 'b')
    target.add_edge('b', 'c')
    target.add_edge('b', 'd')

    return get_vf2_algorithm_result(T, target, "split")

""""
def compute_diffusion_trees_for_main_tag(main_tag):
    
        Function which computes the total number of diffusion trees for the main tag and
        present in the "leaders_DIRECTED" file. The function counts all the diffusion trees
        without restriction of the number of nodes (the directory "leaders_diffusion_trees/main_tag/"
        contains only the diffusion trees with >= 4 nodes).
        :param: main_tag: tag for which to count the total number of diffusion trees
        :return: the total number of diffusion trees
    

    # I collect the artists assigned with the main tag
    artists_main_tag = []
    artists_with_main_tag_file = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in artists_with_main_tag_file:
        line_array = line.split("::")
        artist_encoding = line_array[0]
        artist_tag = line_array[1].replace("\n", "")
        if str(artist_tag) == str(main_tag):
            artists_main_tag.append(artist_encoding)
    artists_with_main_tag_file.close()

    # I count the number of diffusion trees that have as action the artists collected above
    total_number_of_diffusion_trees_for_main_tag = 0
    f = open("leaders_DIRECTED", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        artist_encoding = line_array[0]

        if artist_encoding in artists_main_tag:
            total_number_of_diffusion_trees_for_main_tag += 1
    f.close()

    return total_number_of_diffusion_trees_for_main_tag
"""

def count_files_inside_directories(directory):
    """
        Function which counts the files present in the subdirectories (one for every main music genre)
        of the directory passed by argument
    :return: a list with the following format:
            [(main_music_genre_1, num_files_1) , ... , (main_music_genre_n, num_files_n)]
    """

    main_tags = ["alternative", "blues", "classical", "country", "dance", "electronic", "hip-hop/rap",
                 "jazz", "latin", "pop","r&b/soul", "reggae", "rock"]

    list_of_directories = []

    for main_tag in main_tags:
        main_tag = main_tag.replace("/", "_")
        complete_directory_name = directory + main_tag + "/"

        num_files = len([f for f in os.listdir(complete_directory_name)
                         if os.path.isfile(os.path.join(complete_directory_name, f))])
        pair = (main_tag, num_files)
        list_of_directories.append(pair)

    return list_of_directories


def get_str_to_write(main_tag, total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes, tmp_x_pattern):
    """
        Format divided by "::":
            main music genre
            num. diffusion tree with x pattern in the main music genre
            num. main music genre's diffusion trees
            num. diffusion tree with x pattern in main music genre / num. main music genre's diffusion trees percentage
            num. main music genre's diffusion trees with >=4 nodes
            num. diffusion tree with x pattern in  main music genre  / num. main music genre's diffusion trees
            with >=4 nodes percentage
    :param main_tag: main music genre
    :param tmp_x_pattern: number of diffusion diffusion tree for the main music genre that present the pattern x
    :param total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes: total number of diffusion trees for the
                                                                given main tag with at least 4 nodes
    """

    # the number of lines of the "leaders_DIRECTED"
    # file that contains the artisrts classified with the main tag
    # total_number_diffusion_trees_for_main_tag = compute_diffusion_trees_for_main_tag(main_tag)

    # the total number of diffusion trees for the main tag is given by the sum of the diffusion trees present in
    # the corresponding subdirectory inside "leaders_diffusion_trees/" and "less_then_4_nodes/" directories
    total_number_diffusion_trees_for_main_tag_with_less_then_4_nodes = 0
    less_then_4_nodes_directory = "less_then_4_nodes/"
    list_of_directories = count_files_inside_directories(less_then_4_nodes_directory)
    for key, value in list_of_directories:
        if str(key) == str(main_tag):
            total_number_diffusion_trees_for_main_tag_with_less_then_4_nodes = int(value)
            break

    total_number_diffusion_trees_for_main_tag = total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes + \
                                                    total_number_diffusion_trees_for_main_tag_with_less_then_4_nodes

    str_to_write = str(main_tag) + "::" + str(tmp_x_pattern) + "::"

    fraction1 = float(int(tmp_x_pattern) / total_number_diffusion_trees_for_main_tag)
    percentage1 = fraction1 * 100

    fraction2 = float(int(tmp_x_pattern) / total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes)
    percentage2 = fraction2 * 100

    str_to_write = str_to_write + "::" + str(total_number_diffusion_trees_for_main_tag) + "::" + str(percentage1) \
                   + "::" + str(total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes) + "::" + str(
        percentage2) + "\n"

    return str_to_write


def compute_total_number_of_diffusion_trees():
    """
        Function which computes the total number of diffusion trees present in the "leaders_DIRECTED"
        file, counting all the diffusion trees without restriction of the number of nodes (the
        directory "leaders_diffusion_trees" contains only the diffusion trees with >= 4 nodes) and
        excluding from the computation all the leaders for the artist that don't have a main
        music tag assigned (artists present in the "artists_withou_main_tag")
        :return: the total number of diffusion trees
    """

    artists_without_main_tag = []

    artists_without_main_tag_file = open("filtered_artists_with_main_tag", "r", encoding="utf-8")
    for line in artists_without_main_tag_file:
        line_array = line.split("::")
        artist_encoding = line_array[0]
        artists_without_main_tag.append(artist_encoding)
    artists_without_main_tag_file.close()

    total_number_of_diffusion_trees = 0

    f = open("leaders_DIRECTED", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("::")
        artist_encoding = line_array[0]

        if artist_encoding not in artists_without_main_tag:
            total_number_of_diffusion_trees += 1
    f.close()

    return total_number_of_diffusion_trees


def search_patterns():
    """
        Function which iterates over the diffusion trees present in the "leaders_diffusion_trees/"
        directory in order to check the presence of the following patterns:
        1. star pattern
        2. chain pattern
        3. split pattern
        The computation is done for every main music genre in order to get an insgiht of
        this patterns among them.
    """
    leaders_diffusion_trees_directory = "leaders_diffusion_trees/"

    star_path = leaders_diffusion_trees_directory + "star_pattern"
    chain_path = leaders_diffusion_trees_directory + "chain_pattern"
    split_path = leaders_diffusion_trees_directory + "split_pattern"

    star_file = open(star_path, "a")
    chain_file = open(chain_path, "a")
    split_file = open(split_path, "a")

    # I have to iterate over the main tags directories present in the "leaders_diffusion_trees" directory,
    # in order to compute for each main tag the topological pattern statistics of the diffusion trees under
    # each main tag
    list_of_directories = count_files_inside_directories(leaders_diffusion_trees_directory)

    print(list_of_directories)

    # get global number of diffusion trees
    global_number_diffusion_tress = compute_total_number_of_diffusion_trees()

    print("global_number_diffusion_tress = " + str(global_number_diffusion_tress))

    # compute the global number of diffusion trees with >= 4 nodes
    global_number_diffusion_trees_with_at_least_4_nodes = 0
    for main_tag, num_files_in_directory in list_of_directories:
        global_number_diffusion_trees_with_at_least_4_nodes += int(num_files_in_directory)

    print("global_number_diffusion_trees_with_at_least_4_nodes = " +  \
                                                            str(global_number_diffusion_trees_with_at_least_4_nodes))

    for main_tag, total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes in list_of_directories:
        main_tag = main_tag.replace("/", "_")
        complete_directory_name = leaders_diffusion_trees_directory + main_tag + "/"

        tmp_main_tag_star_pattern = 0
        tmp_main_tag_chain_pattern = 0
        tmp_main_tag_split_pattern = 0

        analyzed_files = 0

        list_of_files = os.listdir(complete_directory_name)
        for file in list_of_files:

            file_path = complete_directory_name + file
            f = open(file_path, "r")

            # re-construct diffusion tree
            g = nx.DiGraph()
            count = 0

            for line in f:
                fields = line.strip().split()
                u = int(fields[0])
                v = int(fields[1])

                count += 1
                if count == 1:
                    #  add the attribute 'leader' to the first node (root of the diffusion tree), to
                    # distinguish it from the others
                    g.add_node(u, status='leader')

                g.add_edge(u, v)
            f.close()

            analyzed_files += 1
            print("file " + str(analyzed_files) + " : " + file_path)

            star = compute_star_pattern(g)
            if star is True:
                tmp_main_tag_star_pattern += 1
            print("star = " + str(star))

            chain = compute_chain_pattern(g)
            if chain is True:
                tmp_main_tag_chain_pattern += 1
            print("chain = " + str(chain))

            split = compute_split_pattern(g)
            if split is True:
                tmp_main_tag_split_pattern += 1
            print("split = " + str(split))

            # remove node, edges and attributes (not sure of garbage collector, here is why)
            g.clear()

        # keep track of the main tag's diffusion trees on file
        main_tag = main_tag.replace("_", "/")
        str_to_write = get_str_to_write(main_tag, total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes,
                                                                                        tmp_main_tag_star_pattern)
        star_file.write(str_to_write)
        star_file.flush()

        str_to_write = get_str_to_write(main_tag, total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes,
                                                                                            tmp_main_tag_chain_pattern)
        chain_file.write(str_to_write)
        chain_file.flush()

        str_to_write = get_str_to_write(main_tag, total_number_diffusion_trees_for_main_tag_with_at_least_4_nodes,
                                                                                            tmp_main_tag_split_pattern)
        split_file.write(str_to_write)
        split_file.flush()

    star_file.close()
    chain_file.close()
    split_file.close()

search_patterns()