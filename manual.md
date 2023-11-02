# ChatDev Software User Manual

## Introduction

Welcome to the user manual for ChatDev Software! This software is designed to help you complete various tasks related to data analysis and natural language processing. It provides a user-friendly interface and a set of functions to perform tasks such as reading Excel files, clustering embeddings, and generating task name summaries using OpenAI GPT-3.5-turbo.

## Installation

To use ChatDev Software, you need to install the required dependencies listed in the `requirements.txt` file. You can do this by running the following command:

```
pip install -r requirements.txt
```

Make sure you have Python installed on your machine before running this command.

## Usage

Once you have installed the dependencies, you can run the software by executing the `main.py` file. This will open a graphical user interface (GUI) where you can interact with the software.

### Step 1: Read Excel File

Click on the "Read Excel File" button to select an Excel file to read. The software will read the file and discard any columns except the first three columns (Task Name, Name, Link). The data will be saved to a Parquet file named "Data_pd.parquet".

### Step 2: Get Embeddings

Click on the "Get Embeddings" button to generate embeddings for the Task Name column using the Sentence Transformers package and the "BAAI/bge-small-en-v1.5" model. The embeddings will be added as a new column to the data and saved to the Parquet file.

### Step 3: Cluster Embeddings

Click on the "Cluster Embeddings" button to cluster the embeddings using K-means with 20 clusters. Each row in the data will be assigned a cluster label, which will be added as a new column. The updated data will be saved to the Parquet file.

### Step 4: Reduce Dimensions

Click on the "Reduce Dimensions" button to reduce the embedding dimensions to 2 using Linear Discriminant Analysis (LDA). The reduced embeddings will be added as a new column to the data and saved to the Parquet file.

### Step 5: Display Scatter Plot

Click on the "Display Scatter Plot" button to display a scatter plot of the reduced embeddings. Each point in the plot represents a data point, and the color of the point indicates its cluster label.

### Step 6: Find Centroids

Click on the "Find Centroids" button to find the centroid of each cluster. A new Pandas dataframe named "Cluster_pd" will be created, with one row per cluster label. The cluster centroids will be added as a new column to the dataframe and saved to the Parquet file.

### Step 7: Find Closest Embeddings

Click on the "Find Closest Embeddings" button to find the 10 embeddings closest to each cluster's centroid. A new Pandas dataframe named "Closest_pd" will be created, with one row per closest embedding. The dataframe will contain the Task Name and Embedding columns. The dataframe will be saved to the Parquet file.

### Step 8: Get Task Name Summaries

Click on the "Get Task Name Summaries" button to generate task name summaries using OpenAI GPT-3.5-turbo. For each cluster in "Closest_pd", the software will submit the list of 10 Task Names to the model and ask for a summary in 8 words. The summaries will be recorded in the "Cluster_pd" dataframe and saved to the Parquet file.

### Step 9: Display Cluster Table

Click on the "Display Cluster Table" button to display a table of the "Cluster_pd" dataframe. The table will show the Cluster label, Task Name summary, Reduced Embedding, and the count of records per cluster from the "Data_pd" dataframe.

### Step 10: Display Centroid Scatter Plot

Click on the "Display Centroid Scatter Plot" button to display a scatter plot of the cluster centroids. Each centroid will be labeled with its Task Name summary.

## Conclusion

Congratulations! You have successfully completed the tasks using ChatDev Software. Feel free to explore the different features and functionalities of the software to analyze your data and generate insights. If you have any questions or need further assistance, please refer to the documentation or contact our support team. Happy analyzing!