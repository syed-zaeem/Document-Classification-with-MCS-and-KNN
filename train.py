import docx
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import networkx as nx
import os
import pickle
from gspan_mining import gSpan
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')  


folder_path = "allDocuments"

documents_graphs = []

documents_graphs_gspan = []

for filename in os.listdir(folder_path):
    if filename.endswith(".docx"):
        doc = docx.Document("allDocuments/"+filename)
        category = filename.split('.')[0]
        category = filename.split('_')[1]
        record = {}
        stop_words = set(stopwords.words('english'))
        ps = PorterStemmer()
        preprocessed_text = []

        text = " ".join([paragraph.text for paragraph in doc.paragraphs])
        tokens = word_tokenize(text.lower())

        for word in tokens:
            if word.isalpha() and word not in stop_words:
                stemmed_word = ps.stem(word)
                preprocessed_text.append(stemmed_word)

        G = nx.DiGraph() #Directed graph
        for i in range(len(preprocessed_text) - 1):
            word1 = preprocessed_text[i]
            word2 = preprocessed_text[i + 1]
            G.add_edge(word1, word2)
        # print("This is catagory" , category, len(documents_graphs))
        # print("This is graph " , G)
        record['category'] = category
        record['graph'] = G
        documents_graphs.append(record)

#          # Convert graph to gSpan input format (list of nodes)
#         nodes = list(G.nodes())
#         documents_graphs_gspan.append(nodes)

# # Frequent subgraph mining using gSpan
# min_support = 5  # Minimum support count for a subgraph to be considered frequent
# g = gSpan(documents_graphs_gspan)
# frequent_subgraphs = g.run(support=min_support)  

# # Convert frequent subgraphs to NetworkX graphs
# frequent_subgraphs_nx = [nx.DiGraph(subgraph) for subgraph in frequent_subgraphs]

# # Initialize a dictionary to store common subgraphs
# common_subgraphs = {}

# # Iterate through each document graph
# for document_graph in documents_graphs:
#     category = document_graph['category']
#     graph = document_graph['graph']
    
#     # Initialize dictionary to store subgraph presence for the current document
#     subgraph_presence = {subgraph: 0 for subgraph in frequent_subgraphs}
    
#     # Iterate through each frequent subgraph
#     for i, frequent_subgraph in enumerate(frequent_subgraphs_nx):
#         # Check if the frequent subgraph is isomorphic to any subgraph of the document graph
#         for subgraph in nx.algorithms.isomorphism.subgraph_isomorphisms(frequent_subgraph, graph):
#             subgraph_presence[i] = 1
    
#     # Add subgraph presence dictionary to common subgraphs
#     if category not in common_subgraphs:
#         common_subgraphs[category] = []
#     common_subgraphs[category].append(subgraph_presence)

# # Save common subgraphs to a file
# with open('common_subgraphs.pkl', 'wb') as f:
#     pickle.dump(common_subgraphs, f)
















































def compute_graph_distance(graph1, graph2):
    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())
    common_nodes = len(nodes1.intersection(nodes2))

    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common_edges = len(edges1.intersection(edges2))

    max_nodes = max(len(nodes1), len(nodes2))
    max_edges = max(len(edges1), len(edges2))

    distance = 1 - (common_nodes + common_edges) / (max_nodes + max_edges)
    return distance


def knn_classification(test_graph, documents_graphs, k):
    distances = []
    # print("These are all graphs of the all the documents: " , documents_graphs)
    # Compute distances between the test graph and all training graphs
    for record in documents_graphs:
        train_graph = record['graph']
        distance = compute_graph_distance(test_graph, train_graph)
        distances.append((distance, record['category']))

    distances.sort(key=lambda x: x[0])

    nearest_neighbors = distances[:k]

    categories = [neighbor[1] for neighbor in nearest_neighbors]
    category_counts = {category: categories.count(category) for category in set(categories)}
    predicted_category = max(category_counts, key=category_counts.get)

    return predicted_category

test_graph = documents_graphs[0]['graph']  
k = 5  # Choose the number of nearest neighbors
# print("Total Length of all the graphs of all the documents: " , len(documents_graphs))
predicted_category = knn_classification(test_graph, documents_graphs[1:], k)  # Exclude the test document from training data
print("Predicted category:", predicted_category)
with open('training_data.pkl', 'wb') as f:
    pickle.dump(documents_graphs, f)