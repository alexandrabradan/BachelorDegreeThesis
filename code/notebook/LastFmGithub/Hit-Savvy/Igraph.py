from igraph import *
import pandas as pd
import sys


def get_country_encoding(country):
    f = open("country_node_map1", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        c = line_array[0]
        c_encoding = int(line_array[1].replace("\n", ""))
        if country == c:
            return c_encoding
    f.close()
    return -1


def construct_vertices_map(g):
    # print(g.vs[0])  # get vertix's 0 attributes
    f = open("igraph_vertices_node_map1", "a", encoding="utf-8")
    vseq = g.vs  # get graph's vertices
    v_index = 0  # vertices are numbered sequentially
    for v in g.vs:
        user_encoding = v["user_encoding"]
        str_to_write = str(v_index) + "\t" + str(user_encoding) + "\n"
        v_index += 1
        f.write(str_to_write)
        f.flush()
    f.close()


def get_country_color(country):
    color_traduction = {"0": "green", "1": "red", "2": "black", "3": "yellow", "4": "blue", "5": "gray"}
    f = open("country_color_map1", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        c = line_array[0]
        c_color = int(line_array[1].replace("\n", ""))
        if country == c:
            return color_traduction[str(c_color)]
    f.close()
    return -1


def get_color_dict():
    dict = {}
    f = open("", "r", encoding="utf-8")
    for line in f:
        line_array = line.split("\t")
        c = line_array[0]
        c_encoding = int(line_array[1].replace("\n", ""))
        try:
            color = dict[str(c_encoding)]
        except KeyError:
            # assign color to the country base on its continent
            dict[str(c_encoding)] = get_country_color(c)
    f.close()
    return dict


def plot_graph(g):
    # g.to_undirected()
    color_dict = get_color_dict()
    g.vs["color"] = [color_dict[str(country)] for country in g.vs["country"]]
    g.es["edge_size"] = 1
    plot(g, "social_network2.png", layout="drl", colors=color_dict)


def plot_graph2(g):
    # g.to_undirected()
    color_dict = {"0":"red", "1":"blue", "2":"orange", "3":"brown", "4":"yellow", "5":"lime", "6":"cyan", "7":"gray", "8":"salmon", "9":"fuchsia", "10":"darkviolet", "11":"darkgreen", "12":"black"}
    g.vs["color"] = [color_dict[str(music_tag)] for music_tag in g.vs["music_genre"]]
    g.es["edge_size"] = 0.01
    plot(g, "social_network4.png", layout="drl", colors=color_dict)



def get_user_main_genre(user_encoding):
	data = pd.read_csv("filtered_node_list.csv", delimiter="\t", usecols=["Id", "Main_genre"])
	# iterate ove dataframe's rows
	for row in data.itertuples():
		if row.Id == "Id":
			continue
		u = int(row.Id)
		v = int(row.Main_genre)
		if int(u) == int(user_encoding):
			return v
	return -1

def load_the_social_graph():
    print("Loading the social graphload")
    data = pd.read_csv("filtered_edge_list.csv", delimiter="\t", usecols=["Source", "Target"])
    # print(data)

    vertices = []
    countries = []
    main_genres = []
    edges = []
    count = 0

    # iterate ove dataframe's rows
    for row in data.itertuples():
        u = int(row.Source)
        v = int(row.Target)

        if u not in vertices:
            vertices.append(u)
            """country_encoding = get_user_country(u)
            if country_encoding == -1:
                print("user = " + str(v) + "dont' have country")
                sys.exit(-1)
            countries.append(country_encoding)"""
            music_genre_encoding = get_user_main_genre(u)
            if music_genre_encoding == -1:
                print("user = " + str(u) + "dont' have main music genre")
                music_genre_encoding = 5
            main_genres.append(music_genre_encoding)
        if v not in vertices:
            vertices.append(v)
            country_encoding = get_user_country(v)
            if country_encoding == -1:
                # print("friend = " + str(v) + "dont' have country")
                countries.append(5)  # None country
            else: countries.append(country_encoding)
            music_genre_encoding = get_user_main_genre(v)
            if music_genre_encoding == -1:
                print("friend = " + str(v) + "dont' have main music genre")
                music_genre_encoding = 5
            main_genres.append(music_genre_encoding)

        edges.append((u, v))
        print(count)
        count += 1

    # N.B : vertices are added and numbered sequentially, starting from 0. I add to each vertex two
    # attributes (they are 2 dicts):
    # a) "user_encoding": username's encoding
    # b) "country": username's country
    g = Graph(vertex_attrs={"user_encoding": vertices, "country": countries, "music_genre":music_genres}, edges=edges, directed=True)

    summary(g)  # print summary of the current graph
    print("Graph loaded")

    plot_graph(g)
    plot_graph2(g)
    construct_vertices_map(g)
    compute_structural_proprieties_of_the_graph(g)


def write_on_file(list_to_write, filename):
    index = 0
    f = open(filename, "a", encoding="utf-8")
    for elem in list_to_write:
        str_to_write = str(index) + "::" + str(elem) + "\n"
        index += 1
        f.write(str_to_write)
        f.flush()
    f.close()


def compute_structural_proprieties_of_the_graph(g):
    # DEGREE
    in_degree = g.degree(mode="in")
    write_on_file(in_degree, "stats/in_degree")
    print("in_degree")
    out_degree = g.degree(mode="out")
    write_on_file(out_degree, "stats/out_degree")
    print("out_degree")
    neighbors_avg_degree = g.knn()
    # returns ([average degree of neighbors for each vertex],
    #                                                   [average degree of neighbors as a function of vertex degree]])
    write_on_file(neighbors_avg_degree[0], "stats/knn_0")
    write_on_file(neighbors_avg_degree[1], "stats/knn_1")
    print("neighbors_avg_degree")

    # TRIANGLES
    clustering = g.transitivity_local_undirected()  # directed graph is considered undirected
    write_on_file(clustering, "stats/clustering")
    print("clustering")

    # PATHS
    closenness_centr = g.closeness(cutoff=3)   # consider paths <= 3
    write_on_file(closenness_centr, "stats/closenness")
    print("closenness_centr")
    betweenness_centr = g.edge_betweenness(cutoff=3)  # consider paths <= 3
    write_on_file(betweenness_centr, "stats/betweenness")
    print("betweenness_centr")

    # DEGREE
    # instead of eigenvector_centrality, which for directed and acyclic is always 0
    pagerank = g.personalized_pagerank()
    write_on_file(pagerank, "stats/pagerank")
    print("pagerank")
    hubs = g.hub_score()  # A node is a hub if it links to authorities (it hasn't incoming edges)
    write_on_file(hubs, "stats/hubs")
    print("hubs")
    authorities = g. authority_score()  # A node is am authority if it is linked to by hubs (it has many incoming edges)
    write_on_file(authorities, "stats/authorities")
    print("authorities")

    jaccard = g.similarity_jaccard()
    write_on_file(jaccard, "stats/jaccard")
    print("jaccard")

    # HOMOPHILy / ASSORTATIVE MIXING
    assortativity = g.assortativity(types1="country")
    write_on_file(assortativity, "stats/assortativity")
    print("assortativity")


# get_color_dict()
load_the_social_graph()
