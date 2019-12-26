import os
import zipfile

import networkx as nx

from Diffusion import Diffusion


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

            print("Starting to construct the social graph")
            LFD = Diffusion("network_filtered_int1.edgelist",  "first_week_user_artist_count.gz",
                                                                             "user_artist_totalistening_periodlength.gz")
            print("Social graph constructed!")

            unique_artists_with_no_leaders = []  # target artists without leaders (no duplicates)

            actions = open("target_artists", 'r')
            for line in actions:
                tmp_act = line.split("::")
                act = int(tmp_act[2])
                asg = LFD.build_action_subgraph(act, delta)  # DIRECTED induced subgraph for the given action
                leaders = LFD.compute_action_leaders(asg)  # leaders for the given action

                # keep track of target artists without leaders
                if len(leaders) == 0 and act not in unique_artists_with_no_leaders:
                    target_artists_with_no_leaders.write(str(act))
                    target_artists_with_no_leaders.write("\n")
                    unique_artists_with_no_leaders.append(act)

                for l in leaders:
                    # get leader's minimum diffusion tree (from induced DIRECTED subgraph action) =>
                    # return oriented tree constructed from a depth-first-search from source (restrict "asg" to only
                    # the user's influenced by the leader, cutting out the others)
                    l_t = nx.dfs_tree(asg, l)
                    tribe = l_t.nodes()  # users' influenced by the leader

                    # memorize minimum diffusion tree's frontier (leaf nodes)
                    frontier = []
                    for l_n in tribe:
                        if l_t.out_degree(l_n) == 0:  # out-degree = number of edges outgoing from the node "l_n"
                            frontier.append(l_n)

                    # CHARACTERIZE LEADER
                    # print("Characterize the leader |" + str(l) + "| of the action |" + str(act) + "|")
                    width = LFD.compute_width(l_t, l)  # get width ratio between friends in diffusion tree and full graph

                    if width == -1:
                        # I reject the leader found because it's a friend of a seed user for which we didn't collect
                        # users info (it could have incoming edges that we didn't consider not analyzing its network)
                        print("Leader |" + str(l) + "| rejected for action <<" + str(act) + ">>")
                        continue

                    depth = LFD.compute_max_depth(l_t, l, frontier)  # get leeader's diffusion depth
                    mean_depth = float(depth) / len(frontier)  # get mean depth
                    l_strength = LFD.compute_level_strength(l_t, l, distance_factor, act)  # get leader's strenght
                    print("NEW leader |" + str(l) + "| for action <<" + str(act) + ">>")

                    machine_out_file.write(
                        "%d::%d::%d::%1.9f::%1.9f::%1.9f::%1.9f\n" \
                        % (act, l, len(tribe), depth, mean_depth, width, l_strength))

            actions.close()  # close "target_artists" file

            target_artists_with_no_leaders.close()  # close "artists_with_no_leaders" file
            machine_out_file.close()  # close "leaders" file
