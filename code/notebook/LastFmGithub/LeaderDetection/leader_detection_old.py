import os
import networkx as nx

from Diffusion_old import Diffusion_old


def touch(path):
    """
    Create a new file in the specified path
    :param path: path where to create the new file
    """
    with open(path, 'a', encoding='utf-8') as f:
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


def create_directory(path):
    """
   Create a new directory in the specified path
   :param path: path where to create the new directory
   """
    os.makedirs(path)


def check_if_directory_exist(path):
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


def check_if_directory_exist_and_create_it(path):
    """
      Check if a directory exist and if not craete it
      :param path: directory's path to check existance
      :return: 0 if the directory already exist
               1 if the directory didn't exist and I just create it

   """
    exist = check_if_directory_exist(path)

    if exist == -1:
        create_directory(path)
        return 1
    else:
        return 0


if __name__ == "__main__":
    distance_factor = 0.5  # damping factor
    delta = 3 * 604800  # max influencial keep alive time (1 week = 604800 unixtime)

    check = check_if_file_exist("target_artists_with_no_leaders_DIRECTED")

    if check == -1:
        touch("target_artists_with_no_leaders_DIRECTED")

        check = check_if_file_exist("leaders_DIRECTED")

        if check == -1:
            touch("leaders_DIRECTED")

            target_artists_with_no_leaders = open("target_artists_with_no_leaders_DIRECTED", "a")
            machine_out_file = open("leaders_DIRECTED", "a")

            # print("Starting to construct the social graph")
            LFD = Diffusion_old("network_filtered_int1.edgelist", "first_week_user_artist_count.gz",
                                "user_artist_totalistening_periodlength.gz")

            check_if_directory_exist_and_create_it("leaders_diffusion_trees/")
            check_if_directory_exist_and_create_it("less_then_4_nodes/")
            if check == -1:
                create_directory("less_then_4_nodes/")

            actions = open("filtered_target_artists", 'r')
            for line in actions:
                tmp_act = line.split("::")
                act = int(tmp_act[2])
                asg = LFD.build_action_subgraph(act, delta)  # DIRECTED induced subgraph for the given action
                leaders = LFD.compute_action_leaders(asg)  # leaders for the given action

                # keep track of target artists without leaders
                if len(leaders) == 0:
                    target_artists_with_no_leaders.write(str(act))
                    target_artists_with_no_leaders.write("\n")

                for l in leaders:
                    # get leader's minimum diffusion tree
                    l_t = nx.dfs_tree(asg, l)
                    print("Diffusion  tree of the leader |" + str(l) + "| of the action |" + str(act) + "| completed!")
                    tribe = l_t.nodes()  # users' influenced by the leader

                    # memorize minimum diffusion tree's frontier (leaf nodes)
                    frontier = []
                    for l_n in tribe:
                        if l_t.out_degree(l_n) == 0:  # out-degree = number of edges outgoing from the node "l_n"
                            frontier.append(l_n)
                    print("Minimum diffusion tree of the leader |" + str(l) + "| of the action |" + str(
                        act) + "| completed!")

                    # CHARACTERIZE LEADER
                    width = LFD.compute_width(l_t, l)  # get width ratio between friends in diffusion tree and full graph

                    if width == -1:
                        # I reject the leader found because it's a friend of a seed user for which we didn't collect
                        # users info (it could have incoming edges that we didn't consider not analyzing its network)
                        print("Leader |" + str(l) + "| rejected")
                        continue

                    depth = LFD.compute_max_depth(l_t, l, frontier)  # get leeader's diffusion depth
                    mean_depth = float(depth) / len(frontier)  # get mean depth
                    l_strength = LFD.compute_level_strength(l_t, l, distance_factor, act)  # get leader's strenght

                    machine_out_file.write(
                        "%d::%d::%d::%1.9f::%1.9f::%1.9f::%1.9f\n" \
                        % (act, l, len(tribe), depth, mean_depth, width, l_strength))

                    # write leader's diffusion tree on file, if it has at leas 4 nodes
                    edges = l_t.edges()

                    if len(edges) >= 3:  # pair of edges
                        leader_diffusion_tree_file =  "leaders_diffusion_trees/" + str(l) + "_for_action_" + str(
                            act) + "_diffusion_tree"
                        ldtf = open(leader_diffusion_tree_file, "a")
                        for e in edges:
                            ldtf.write(f"{e[0]}\t{e[1]}\n")
                        ldtf.close()
                    else:
                        less_then_4_nodes_file = "less_then_4_nodes/" + str(l) + "_for_action_" + str(
                            act) + "_diffusion_tree"
                        ldtf = open(less_then_4_nodes_file, "a")
                        for e in edges:
                            ldtf.write(f"{e[0]}\t{e[1]}\n")
                        ldtf.close()

                    print("End characterization of the leader |" + str(l) + "| of the action |" + str(
                        act) + "| completed!")

            actions.close()  # close "target_artists" file

            target_artists_with_no_leaders.close()  # close "artists_with_no_leaders" file
            machine_out_file.close()  # close "leaders" file

