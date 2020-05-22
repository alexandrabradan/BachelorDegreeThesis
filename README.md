
# Lastfm Leader detection and Hit-Savvy prediction

Here you can find the work that I've done during my internship at University of Pisa. The aim of the internship was to use a leader detection algorithm combined with a predictive model, to validate the following question:

*"Can we exploit influential nodes to predicte the success of new, succesfull artists in a musical network ?"*

My study shows how in a sparse network (low friendship density and fragmentary adoption log) influential and predictive nodes are uncorrelated (if a node is able to influence his peer neighbors, most of the time he is not useful to spot success. The same is true for nodes used as indicators to measure unseen artists' future success, which in most cases don't espress influential power). My work focuses in detecting these key nodes, discussing the predictive results (new artists' success predictions) and undergoing some adjustments to better tune the scores. Lastly, my study confirms nodes' preferntial attachment using a Personal Listening Data Model, used to classify and cluster the nodes and better explain my analyzed newtork's sparsity.

The code as well as the results and the final report that I've produced, can be found in the directory **./code/notebook/LastfmGithub/**. The directory is organized in the following sub-directories:
1. **/Leader_detection/**: contains the scripts, the data and the results of the leader detection;
2. **/Hit_Savvy/**: contains the scripts, the data and the results of the predictive model and the personal listenings data model;
3. **/Thesis/**: contains my report, in which I explain every undergone step and result.

The leader detection algorithm, the predictive model, the personal listening data model and the communities detection library used in my work and modified to fit a direct, sparse graph, are tuned-up by the Knowledge Discovery Data Mining Laboratory(KDD Lab), the joint research initiative of ISTI Insitute of CNR and the Departement of Computer Science of the University of Pisa. Their original code can be found here:

- https://github.com/GiulioRossetti/leader_detect
- https://github.com/GiulioRossetti/hit-savvy
- https://github.com/GiulioRossetti/LastfmProfiler
- https://github.com/GiulioRossetti/cdlib

while, if you are interested in the theoretical part, the paper explainig them are the following (in my report you can found a small summarry, as well):

- https://link.springer.com/chapter/10.1007/978-3-319-03260-3_28
- https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0189096
- http://www.giuliorossetti.net/about/ongoing-works/publications/
- https://arxiv.org/abs/1206.3552

