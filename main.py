'''
This is the main file that will be executed to run the software.
'''
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.express as px
import numpy as np
from scipy.spatial.distance import cdist
import openai

# Function to read the excel file and discard other columns
def read_excel_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    data_pd = pd.read_excel(file_path, usecols=[0, 1, 2])
    data_pd.to_parquet('Data_pd.parquet')

# Function to get embeddings using Sentence Transformers
def get_embeddings():
    data_pd = pd.read_parquet('Data_pd.parquet')
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    embeddings = model.encode(data_pd['Task Name'].tolist())
    data_pd['Embedding'] = embeddings.tolist()
    data_pd.to_parquet('Data_pd.parquet')

# Function to cluster embeddings using K-means
def cluster_embeddings():
    data_pd = pd.read_parquet('Data_pd.parquet')
    embeddings = np.array(data_pd['Embedding'].tolist())
    kmeans = KMeans(n_clusters=20)
    kmeans.fit(embeddings)
    data_pd['Cluster'] = kmeans.labels_
    data_pd.to_parquet('Data_pd.parquet')

# Function to reduce embedding dimensions using Linear Discriminant Analysis
def reduce_dimensions():
    data_pd = pd.read_parquet('Data_pd.parquet')
    embeddings = np.array(data_pd['Embedding'].tolist())
    lda = LinearDiscriminantAnalysis(n_components=2)
    reduced_embeddings = lda.fit_transform(embeddings, data_pd['Cluster'])
    data_pd['Reduced Embedding'] = reduced_embeddings.tolist()
    data_pd.to_parquet('Data_pd.parquet')

# Function to display scatter plot of reduced embeddings
def display_scatter_plot():
    data_pd = pd.read_parquet('Data_pd.parquet')
    fig = px.scatter(data_pd, x=[x[0] for x in data_pd['Reduced Embedding']],
                     y=[x[1] for x in data_pd['Reduced Embedding']],
                     color=data_pd['Cluster'])
    fig.show()

# Function to find cluster centroids
def find_centroids():
    data_pd = pd.read_parquet('Data_pd.parquet')
    centroids = []
    for cluster in range(20):
        cluster_data = data_pd[data_pd['Cluster'] == cluster]
        centroid = np.mean(np.array(cluster_data['Embedding'].tolist()), axis=0)
        centroids.append(centroid.tolist())
    clusters_pd = pd.DataFrame({'Cluster': range(20), 'Centroid': centroids})
    clusters_pd.to_parquet('Cluster_pd.parquet')

# Function to find closest embeddings to each cluster's centroid
def find_closest_embeddings():
    data_pd = pd.read_parquet('Data_pd.parquet')
    clusters_pd = pd.read_parquet('Cluster_pd.parquet')
    closest_pd = pd.DataFrame()
    for cluster in range(20):
        cluster_data = data_pd[data_pd['Cluster'] == cluster]
        centroid = np.array(clusters_pd[clusters_pd['Cluster'] == cluster]['Centroid'].tolist()[0])
        distances = cdist(np.array(cluster_data['Embedding'].tolist()), [centroid], 'cosine')
        closest_indices = np.argsort(distances, axis=0)[:10]
        closest_embeddings = np.array(cluster_data['Embedding'].tolist())[closest_indices.flatten()]
        closest_task_names = np.array(cluster_data['Task Name'].tolist())[closest_indices.flatten()]
        closest_cluster_pd = pd.DataFrame({'Cluster': cluster_data['Cluster'][:10].tolist(), 'Task Name': closest_task_names.tolist(), 'Embedding': closest_embeddings.tolist()})
        closest_pd = pd.concat([closest_pd, closest_cluster_pd])
    closest_pd.to_parquet('Closest_pd.parquet')

# Function to create prompts for GPT-3.5-turbo and get task name summaries
def get_task_name_summaries():
    closest_pd = pd.read_parquet('Closest_pd.parquet')
    cluster_pd = pd.read_parquet('Cluster_pd.parquet')
    cluster_pd['Task Name Summary']=''
    for cluster in cluster_pd['Cluster'].tolist():
        closest_data = closest_pd[closest_pd['Cluster'] == cluster]
        task_names = closest_data['Task Name'].tolist()
        prompt = f"Summarize the following task names in just 8 words: {task_names}"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=8,
            n=1,
            stop=None,
            temperature=0.5
        )
        summary = response.choices[0].text.strip()
        cluster_pd.loc[cluster_pd['Cluster'] == cluster, 'Task Name Summary'] = summary
    cluster_pd.to_parquet('Cluster_pd.parquet')

# Function to display table of Cluster_pd
def display_cluster_table():
    cluster_pd = pd.read_parquet('Cluster_pd.parquet')
    data_pd = pd.read_parquet('Data_pd.parquet')
    cluster_counts = data_pd['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    cluster_table = pd.merge(cluster_pd, cluster_counts, on='Cluster')
    cluster_table = cluster_table[['Cluster', 'Task Name Summary', 'Centroid', 'Count']]
    print(cluster_table)

# Function to display scatter plot of cluster centroids with labels
def display_centroid_scatter_plot():
    cluster_pd = pd.read_parquet('Cluster_pd.parquet')
    fig = px.scatter(cluster_pd, x=[x[0] for x in cluster_pd['Centroid']],
                     y=[x[1] for x in cluster_pd['Centroid']],
                     text=cluster_pd['Task Name Summary'])
    fig.show()

# GUI implementation
def main():
    root = tk.Tk()
    root.title("ChatDev Software")
    root.geometry("400x400")
    read_excel_button = tk.Button(root, text="Read Excel File", command=read_excel_file)
    read_excel_button.pack()
    get_embeddings_button = tk.Button(root, text="Get Embeddings", command=get_embeddings)
    get_embeddings_button.pack()
    cluster_embeddings_button = tk.Button(root, text="Cluster Embeddings", command=cluster_embeddings)
    cluster_embeddings_button.pack()
    reduce_dimensions_button = tk.Button(root, text="Reduce Dimensions", command=reduce_dimensions)
    reduce_dimensions_button.pack()
    display_scatter_plot_button = tk.Button(root, text="Display Scatter Plot", command=display_scatter_plot)
    display_scatter_plot_button.pack()
    find_centroids_button = tk.Button(root, text="Find Centroids", command=find_centroids)
    find_centroids_button.pack()
    find_closest_embeddings_button = tk.Button(root, text="Find Closest Embeddings", command=find_closest_embeddings)
    find_closest_embeddings_button.pack()
    get_task_name_summaries_button = tk.Button(root, text="Get Task Name Summaries", command=get_task_name_summaries)
    get_task_name_summaries_button.pack()
    display_cluster_table_button = tk.Button(root, text="Display Cluster Table", command=display_cluster_table)
    display_cluster_table_button.pack()
    display_centroid_scatter_plot_button = tk.Button(root, text="Display Centroid Scatter Plot", command=display_centroid_scatter_plot)
    display_centroid_scatter_plot_button.pack()
    root.mainloop()
if __name__ == "__main__":
    main()