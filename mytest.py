import docx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')


true_labels = []
predicted_labels = []

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    preprocessed_text = []

    tokens = word_tokenize(text.lower())

    for word in tokens:
        if word.isalpha() and word not in stop_words:
            stemmed_word = ps.stem(word)
            preprocessed_text.append(stemmed_word)

    return preprocessed_text

def create_document_graph(text):
    G = nx.DiGraph()

    for i in range(len(text) - 1):
        word1 = text[i]
        word2 = text[i + 1]
        G.add_edge(word1, word2)

    return G

with open('training_data.pkl', 'rb') as f:
    documents_graphs = pickle.load(f)

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

test_folder_path = "testDocuments"

total_test_documents = 0
correct_predictions = 0

for filename in os.listdir(test_folder_path):
    if filename.endswith(".docx"):
        test_doc = docx.Document(os.path.join(test_folder_path, filename))
        true_category = filename.split('_')[1]
        true_labels.append(true_category)

        test_text = " ".join([paragraph.text for paragraph in test_doc.paragraphs])
        preprocessed_test_text = preprocess_text(test_text)
        test_graph = create_document_graph(preprocessed_test_text)
        test_graph = create_document_graph(preprocessed_test_text)

        k = 5 
        predicted_category = knn_classification(test_graph, documents_graphs, k)
        predicted_labels.append(predicted_category)

        total_test_documents += 1
        if predicted_category == true_category:
            correct_predictions += 1

accuracy = correct_predictions / total_test_documents * 100
print("Accuracy:", accuracy)


conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=sorted(set(true_labels)), yticklabels=sorted(set(true_labels)))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(true_labels, predicted_labels)
print("Classification Report:")
print(class_report)
