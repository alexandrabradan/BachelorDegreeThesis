This directory contains the following scripts:

and following files:


which allowed me to study the dynamics in the spread of artists belonging to different music genres among our social graph.
First of all I had to assign to every interest artist (artist listened by a leader / hit-savvy) a music genre of belonging.
This was possible thanks to the info returned by the Last.fm API *artist.getTags*, which returned a list of music tags assigned  to the artist. My main work was to clean-up all this tags and among them choose the most representative one. Moreover, the choosen representative tag shall belong to a finite number of main tags, namely:

**['Alternative', 'Blues', 'Classical', 'Country', 'Dance', 'Electronic', 'Hip-Hop/Rap', 'Jazz', 'Latino', 'Pop', 'R&B/Soul', 'Reggae', 'Rock']**

The main tags were retrieve from https://www.musicgenreslist.com/ , a website which tries to be a web-database of all the music genres on the Internet. 
Since Last.fm API *artist.getTags* returned me a tag list, I had to develop an heuristc in order to choose an artist's most representative tag. This heuristc is very simple:
* for every artist analyzed, I iterated over the tag list and for every tag I consulted the *music_tags_map.json* file, in order to retrieve (if present) the corresponding mapping of the tag;
* if the corresponding mapping wasn't found I wrote the tag on the "not_mapped_tags" file and continued the iteration over the list, otherwise I just continued the iteration;
* after this first retrieval process I checked the "not_mapped_tags" file in order to understand why the tags present in it weren't mapped in the *music_tags_map.json* file. I manually assigned to this tags the most adequate main tag and updated the *music_tags_map.json* file with the new assignments;

I repetaed the process described above untill the "not_mapped_tags" file was empty (every tag was correctly mapped) and then for every artist I proceded as following:
* I iterated over his tag list and for every tag I consulted the *music_tags_map.json* file, in order to retrieve the corresponding mapping (main tag);
* with every main tag I updated a temporary counter list (indexed by the the main tags);
* consulting the counter list I chose as the artist's most representative tag the index (main tag) of the slot with the highest values and I wrote this association on the *artist_tag_map1* file. If two or more values were equally the highest, I retrieved artist's similar singers through the Last.fm API *artist.getSimilar* and with a similar retrieval process I retrieved the similar artist's main tags. I assigned as main tag to the analyzed artist the most frequent main tag among his/her similar artists.
