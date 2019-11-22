This directory contains the data used in our data-driven study. The data is divided in the following subdirectories used in different processes of our predictioction analysis. It is omited the users' and artists' data files due to sensible informations that they contain and due to high storage space needed:

a) **networks**: directory which collects the following files and correlated infos:
   - **node_map1**: file which contains the usernames of our DataSets encoded as integers;
   - **network_full1.edgelist**: file which contains the social graph extracted from Last.Fm. It contains two space-separated                columns with the usernames of two nodes, connected if they are friend in the Last.Fm social graph. The graph is unweighted and undirected;
   - **network_filtered1.edgelist**: network with only friendship's bond between the users analyzed in the Datasets;
   - **network_filtered_int1.edgelist**: network with only friendship's bond between the users analyzed in the Datasets, encoded as integers.

b) **leader_detection**: directory which collects the following files and correlated infos:
  - **weeks_map**: file which contains the weeks of our study encoded as integer (our study goes from 16/07/2017 to 16/07/2019);
  - **artists_map**: file which contains the artists listend by the users of our DataSets encoded as integers;
  - **week_user_artist_count**: the main action table used by the ExtractLeader procedure. It has 4 columns separated by the token “::” (without quotes). The first column is the timestamp (week id) of the action; the second column is the user id (correspondent to the nodes id in the file “network”); the third column is the artist id and the fourth column contains the number of listenings made by the user of that particular artist in that particular period;
   - **user_artist_totalistening_periodlength**: the complementary action table for the calculation of the Strength measure. It has 4 columns separated by the token “::” (without quotes). The first column is the user id; the second column is the artist id; the third column is the total number of listenings that the user made of the artist and the fourth column is the length of the period in which the user listened to the artist (last week minus first week);
   - **artist_tag_id**: a dictionary with three columns separated by the token “::” (without quotes). The first column is the artist's is id to cross the information with the action table in the file “week_user_artist_count” and the second column is its majority tag (music genre) and the third column .  
