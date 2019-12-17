This directory contains the following scripts:
1. TagsCrawler.py : script which retrieves the tags present at https://www.musicgenreslist.com/;
2. WikipediaTagsCrawler: script which retrieves the tags present at https://en.wikipedia.org/wiki/List_of_music_styles#Avant-garde;
3. MergeTagsFiles.py: script which merges the tags retrieved from https://www.musicgenreslist.com/ and https://en.wikipedia.org/wiki/List_of_music_styles#Avant-garde, keeping track of the colliding tags in the "data/collided_music_tags_map.json" file;
4. MergeLauraTags.py: script which checks if the tags present in the "data/merged_music_tags_map.json" file  are the same present in the "data/laura_music_genre_list.json" file (this last file is the one used for the music tag analysis done in the "The italian music superdiversity" paper and so the guideline I followd in the mapping process);
5. TagsCollector.py: script run multiple times in  order to empty the "data/not_mapped_tags" file and give them a mapping  (this is done mainly manually).

and following data files, gathered in the "data/" subdirectory:
1. laura_music_genre_list.json: tags mapping used in the "The italian music superdiversity" paper and used as a guideline;
2. music_tags_map.json: file which contains the tags mapping present at https://www.musicgenreslist.com/;
3. music_wikipedia_tags_map.json: file which contains the tags mapping present at https://en.wikipedia.org/wiki/List_of_music_styles#Avant-garde;
4. collided_music_tag_map.json: file which contains the colliding mapping in the "music_tags_map.json" and "music_wikipedia_tags_map.json" files;
5. MusicalInstruments: a list of musical instruments to add to the "merged_music_tags_map.json" mapped as "classical";
6. NationAndNationalities: a list of nations and nationalities to add to "not_considered_tags" in order to avoid to map this key;
7. not_considered_tags: file which contains the tags excluded from the mapping;
8. not_mapped_tags: file which contains the tags not already mapped;
9. to_add.json: file which contains the tags to add to the merged_music_tags_map.json fi
10. collided_music_tag_map_with_laura.json: file which contains the colliding mapping in the "merged_music_tags_map.json" and "laura_music_genre_list.json" files;
11. merged_music_tags_map.json: final mapping file

which allowed me to study the dynamics in the spread of artists belonging to different music genres in our social graph.
First of all I had to assign to every target artist (artist listened by a leader / hit-savvy) a music genre of belonging.
This was possible thanks to the info returned by the Last.fm API *artist.getTags*, which returned a list of music tags assigned to the artist. My main work was to clean-up all this tags and among them choose the most representative one. Moreover, the choosen representative tag shall belong to a finite number of main tags, namely:

**['alternative', 'blues', 'classical', 'country', 'dance', 'electronic', 'hip-hop/rap', 'jazz', 'latin', 'pop', 'r&b/soul', 'reggae', 'rock']**

The main tags were retrieve from https://www.musicgenreslist.com/ , a website which tries to be a web-database of all the music genres on the Internet. Moreover, I merged the retrieved tags with the tags present at https://en.wikipedia.org/wiki/List_of_music_styles#Avant-garde, in order to have a larger tags mapping.
Since Last.fm API *artist.getTags* returned me a tag list, I had to develop an heuristc in order to choose an artist's most representative tag. This heuristc is very simple:
* for every artist analyzed, I iterated over the tag list and for every tag I consulted the *music_tags_map.json* file, in order to retrieve (if present) the corresponding mapping of the tag;
* if the corresponding mapping wasn't found I wrote the tag on the "not_mapped_tags" file and continued the iteration over the list, otherwise I just continued the iteration;
* after this first retrieval process I checked the "not_mapped_tags" file in order to understand why the tags present in it weren't mapped in the *music_tags_map.json* file. I manually assigned to this tags the most adequate main tag and updated the *music_tags_map.json* file with the new assignments;

I repetaed the process described above untill the "not_mapped_tags" file was empty (every tag was correctly mapped) and then for every artist I proceded as following:
* I iterated over his tag list and for every tag I consulted the *music_tags_map.json* file, in order to retrieve the corresponding mapping (main tag);
* with every main tag I updated a temporary counter list (indexed by the the main tags);
* consulting the counter list I chose as the artist's most representative tag the index (main tag) of the slot with the highest values and I wrote this association on the *artist_tag_map1* file. If two or more values were equally the highest, I retrieved artist's similar singers through the Last.fm API *artist.getSimilar* and with a similar retrieval process I retrieved the similar artist's main tags. I assigned as main tag to the analyzed artist the most frequent main tag among his/her similar artists.
