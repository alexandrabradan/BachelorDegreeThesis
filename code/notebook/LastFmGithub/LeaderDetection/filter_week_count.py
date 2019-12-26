import gzip
import os
import re
import string
import sys
import zipfile

from tqdm import tqdm


def touch(path):
    """
    Create a new file in the specified path
    :param path: path where to create the new file
    """
    with open(path, 'a', encoding='utf-8') as f:
        os.utime(path, None)
        f.close()


def verify_well_encoding(encoding_to_check):
    try:
        int(encoding_to_check)
        return 0  # ok
    except ValueError:
        return -1

def reverse(text):
    a = ""
    for i in range(1, len(text) + 1):
        a += text[len(text) - i]
    return a

def get_username_from_encoding(user_encoding):

    b = bytes(".*" + "\t" + user_encoding + "\n", 'utf-8')
    my_regex = re.compile(b)

    with zipfile.ZipFile('networks.zip') as z:
        for filename in z.namelist():
            # check is week_encoding has a corresponding unixtime week
            if filename == "node_map1":

                with z.open(filename) as f:
                    for line in f:
                        match = my_regex.match(line)

                        if match is not None:
                            # retrieve username
                            # line's format : username \t user_encoding
                            line_array = line.decode("utf-8").split('\t')
                            username = line_array[0].strip('\t')
                            username2 = username.strip('\n')
                            f.close()
                            z.close()
                            return username2

    return ""  # no corresponding encoding


def get_correct_encoding(wrong_encoding):
    username_encoding = wrong_encoding.split("\t")

    print("wrong_encoding = " + wrong_encoding)
    print("username_encoding = " + username_encoding[0])

    u_e2 = ""

    if len(username_encoding) == 1:  # wrong_encoding is attach to username
        # reverse user_encoding
        username_encoding_reversed = reverse(username_encoding[0])

        tmp_u_e2 = ""
        last_letter_found = False

        # iterate over the single encoding word
        for i in range(len(username_encoding_reversed)):
            if username_encoding_reversed[i].isdigit() and last_letter_found is False:
                tmp_u_e2 += username_encoding_reversed[i]
            else:
                last_letter_found = True
                break

        u_e2 = reverse(tmp_u_e2)
    else:  # wrong_encoding has space

        u_e2 = re.sub(r"[\n\t\s]*", "", username_encoding[1])

    if u_e2 == "":
        print("encoding_got =" + str(u_e2))
        sys.exit(-999)

    # verify if the encoding is an int
    try:
        int(u_e2)

        username = ""
        remove_first_char = "xxx"

        while username == "" and len(remove_first_char) > 0:

            # check if the encoding corresponds to a user of our DataSets
            username = get_username_from_encoding(u_e2)

            if username == "":
                remove_first_char = u_e2[1:]
                # retry to retrieve username from encoding
                username = get_username_from_encoding(u_e2)

        if username == "":
            print("CAN'T GET USERNAME FROM ENCODING = " + u_e2)
            sys.exit(999)

        return u_e2  # return user's encoding
    except Exception:
        print("wrong_encoding = " + str(wrong_encoding))
        print("encoding_got =" + str(u_e2))
        sys.exit(-999)


if __name__ == "__main__":

    # create "week_user_artist_count2.gz" file
    touch("week_user_artist_count.gz")

    # iterate over the "leader_detection.zip/week_user_artist_count.gz" file and filter it
    with gzip.open("week_user_artist_count.gz", 'at', encoding='utf-8') as outfile:
        with zipfile.ZipFile('leader_detection.zip') as z:

            for filename in z.namelist():
                if filename == "week_user_artist_count.gz":
                    with z.open(filename, 'r') as f:
                        # needed to open "week_user_artist_count.gz" (compressed file, inside a zip archieve)
                        gzip_fd = gzip.GzipFile(fileobj=f)

                        right_encoding_dict = {}

                        for line in gzip_fd:
                            l = line.decode("utf-8")
                            print("LINE = " + l)

                            # get data from the line
                            data = l.split("::")

                            week_encoding = data[0]
                            user_encoding = data[1]
                            artist_encoding = data[2]
                            # remove all white spaces and newlines
                            playcount = data[3].translate({ord(c): None for c in string.whitespace})

                            correct_user_encoding = ""

                            # verify if week is well encoded
                            if verify_well_encoding(week_encoding) != 0:
                                print("week_encoding WRONG ENCODED = " + str(week_encoding))
                                sys.exit(-999)

                            # verify if user is bad encoded
                            if user_encoding in right_encoding_dict:
                                correct_user_encoding = right_encoding_dict[user_encoding]
                            else:
                                # verify if user is well encoded
                                if verify_well_encoding(user_encoding) != 0:
                                    correct_user_encoding = get_correct_encoding(str(user_encoding))
                                    right_encoding_dict[user_encoding] = correct_user_encoding
                                else:
                                    correct_user_encoding = user_encoding

                            # verify if artist is well encoded
                            if verify_well_encoding(artist_encoding) != 0:
                                print("artist_encoding WRONG ENCODED = " + str(artist_encoding))
                                sys.exit(-999)

                            # verify if playcount is well encoded
                            if verify_well_encoding(playcount) != 0:
                                print("playcount_encoding WRONG ENCODED = " + str(playcount))
                                sys.exit(-999)
                        
                            str_to_write = str(week_encoding) + "::" + str(correct_user_encoding) + "::" + str(
                                artist_encoding) + "::" + str(playcount)

                            # write correct string on new file
                            outfile.write(str_to_write)
                            outfile.write("\n")

        outfile.close()

