import re
import sys
import zipfile


def check_if_user_is_encoded(username):
    """
        Function which checks if a username is econded in our  "node_map1" file (in case we don't find him, it means
        that is a friends with less then 100 users analyzed in our DataSets)
        :param username to search in the  "node_map1" file
        :return >= 0 the encoding, if the username is encoded
                -1 otherwise
    """

    if username == "[brain]":
        return 24068
    elif username == "[Mike]":
        return 3743
    elif username == "[Ninja] Killer":
        return 23758

    # Compile a regular expression pattern into a regular expression object, which can be used
    # for matching using its match(), search() and other methods
    b = bytes(username, 'utf-8')  # convert username to bytes
    my_regex = re.compile(b)

    with zipfile.ZipFile('networks.zip') as z:
        for filename in z.namelist():
            # check is user is encoded (if not jump to next iterationn beacuse we don't need his listenings)
            if filename == "node_map1":
                # open  "node_map1" file
                with z.open(filename) as f:
                    for line in f:
                        match = my_regex.match(line)

                        if match is not None:
                            # retrive user's encoding
                            username_encoding = line.split(b)
                            # get encoding
                            u_e = username_encoding[1].decode("utf-8")  # convert bytes to string
                            u_e2 = re.sub(r"[\n\t\s]*", "", u_e)
                            f.close()
                            z.close()

                            # verify if the encoding is an int
                            try:
                                int(u_e2)
                                return u_e2
                            except Exception:
                                # print("username = " + str(username))
                                # print("username_encoding =" + str(u_e2))
                                # sys.exit(-999)
                                return -1

    # signal to the caller that the username is not encoded in our file
    return -1


def gather_mutual_friends():

    u_dict = {}

    with zipfile.ZipFile('networks.zip') as z:
        for filename in z.namelist():
            if filename == "network_filtered_int1.edgelist":
                edges = z.open(filename)

                for e in edges:
                    e = e.decode("utf-8")
                    part = e.split()
                    u = part[0]  # username
                    v = part[1].strip()  # friend

                    try:
                       u_list = u_dict[str(u)]

                       if v not in u_list:
                           u_list.append(v)

                       u_dict[str(u)] = u_list

                    except KeyError:
                        u_dict[str(u)] = [v]

        mutual_friends_u = []
        mutual_friends_v = []
        not_encoded_keys = []

        for key, value in u_dict.items():
            # check for every (key, value[i]) edge if there exists a corresponding (value[i], key) edge)
            for key_friend in value:

                try:
                    key_value_list = u_dict[str(key_friend)]

                    if key in key_value_list:   # mutual friendship between (key, value[i])

                        """
                        # get key's encoding
                        key_encoding = check_if_user_is_encoded(key)
                        key_friend_encoding = check_if_user_is_encoded(key_friend)

                        if key_encoding == -1:
                            not_encoded_keys.append(key)

                        if key_friend_encoding == -1:
                            not_encoded_keys.append(key_friend)
                        """

                        # get key_friend's encoding
                        mutual_friends_u.append(key)
                        mutual_friends_v.append(key_friend)

                except KeyError:
                    continue  # no data for value[i]

        # print(mutual_friends)
        print(len(mutual_friends_u))
        print(len(mutual_friends_v))

        print(mutual_friends_u[len(mutual_friends_u) - 1])
        print(mutual_friends_v[len(mutual_friends_v) - 1])

        print("------------------")
        # print(not_encoded_keys)
        # print(len(not_encoded_keys))


gather_mutual_friends()