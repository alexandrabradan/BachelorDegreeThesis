This folder contains the following scripts:
1. **social_graph.py**: script which:
    - extracts the network of the study from the 55 Datasets and stores it in the *leader_detection/network_full1.edgelist* file;
    - filters the network, keeping only friendship's bond between the users analyzed in the 55 Datasets and storing the result in the *leader_detection.zip/network_filtered1.edgelist* file;
    - encodes the users of the filetered network as integers in the *leader_detection.zip/node_map1*;
    - encodes the *leader_detection.zip/network_filtered1.edgelist* using the *leader_detection.zip/node_map1* file, creating the *leader_detection.zip/network_filtered_int1.edgelist;
2. **weeks_encoder.py**: script which encodes as integer the 104 weeks of the study (16/07/17 - 16/07/19). The result of the encoding is stores in the *week_map1* file;
3. **artists&week_count_encoder.py**: script which makes a backup of all the listenings made by the users analyzed and stores the result in the *week_user_artist_count.gz* file. The scipt takes care also in encoding all the artists encountered in the *artists_map1* file: 
4. **total_listenings_encoder.py**: script which take care of constructing the *first_week_artist_count.gz* and *user_artist_total_listenings_period_length.gz* files;
5. **convert_first_week_file_into_directory.py**: script which splits the *first_week_artist_count.gz* (1.5 GB) file into the directory *first_week*, which stores in its files, for every artist, the first week of listening made by the users who performed an action on it. This is done to speed up, given an artist and an user, the finding of the first week of listening made by the user to the artist;
6. **convert_total_listenings_file_into_directory.py**: script which splits the *user_artist_total_listenings_period_length.gz* (1.4 GB) file into the directory *total_listenings*, which stores in its files, for every user, the total number of listenings made by him. This is done to speed up, given an user and an artist, the finding of the total number of playcounts made by the user to the artist;
7. **Gather_target_artists.py**: script which collects in the *target_artists* file all the artists candidated to help in leader detection strategy. This target artistst are "new", in the sense that their absolute first listenings is recorded SIX MONTHS after the beginning of our observation period (16/07/17).  If an artist was in activity before our observationtime window, there is no way to know if a user has listened to it before, therefore nullifying our leader detection strategy. 
8. **leader_detetction.py**: script which for every action present in the *target_artists* file computes its leaders (users who first adopted it and influenced their neighborhood), storing the result in the *leaders_DIRECTED* file. The actions without any leader are stored in the *target_artists_with_no_leaders_DIRECTED*;
9. **Diffusion.py**: script which implements the leader detection strategy.
and following data files:
1. **leader_detection.zip**: archieve which contains the following data:
  - *network_full1.edgelist*: global network extracted from the 55 Datasets;
  - *network_filtered1.edgelist*: global network filtered, keeping only the friendship's bond between the users analyzed in   the 55 Datasets;
  - *node_map1*: file which encodes as integers the filtered network's users;
  - *network_filtered_int1.edgelist*: file which encodes the filtered network;
2. **week_map1**: file which encodes as integers the weeks of the study (16/07/17 - 16/07/19);
3. **artist_map1**: file which encodes as integers the artists of the study;
4. **first_week_artist_count.gz**:  file which encodes as integers the first week of listening of every artist of the study in gurumine format:
                        week_encoding::user_encoding::artist_encoding::first_week_playcount
5. **user_artist_total_listenings_period_length.gz**: file which encodes as integers the total playcounts of every artist of the study in gurumine format:
                      user_encoding::artist_encoding::total_playcounts::period_length
6. **target_artists**: file which contains the "new" artists (new in the sense that their absolute first listenings is recorded SIX MONTHS after the beginning of our observation period (16/07/17));
7. **leaders_DIRECTED**: file which contains the leaders found for the artists present in the *target_artists* file;
8. **target_artists_with_no_leaders_DIRECTED**: file which contains the artists present in the *target_artists* but not having a leader in  *leaders_DIRECTED*.
