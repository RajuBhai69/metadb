def hierarchial_clusteringmetadb():
    print(""" 
         #Hierarchical Clustering
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.decomposition import PCA
import seaborn as sns
df = pd.read_csv('Wholesale customers data.csv')

df.drop(columns=['Channel','Region'],inplace=True,axis=1)
df.dropna(inplace=True)

scaled_feature = StandardScaler().fit_transform(df)

agg = AgglomerativeClustering(n_clusters=3, linkage='complete')
agg_labels = agg.fit_predict(scaled_feature)

# Add labels to original dataframe
df['Agglomerative_Cluster'] = agg_labels
# scaled_feature['Cluster'] = cluster_labels

dendrogram(
    linkage(scaled_feature,method='complete',metric='euclidean'),
    orientation='top',
)

plt.title('Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_feature)

df['PCA1'] = reduced_data[:, 0]
df['PCA2'] = reduced_data[:, 1]

sns.scatterplot(x='PCA1', y='PCA2', hue='Agglomerative_Cluster', data=df, palette='Set2')
plt.title('Agglomerative Clustering with PCA')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()


theory:

What is hierarchical clustering?
A clustering technique that builds a tree (dendrogram) of nested clusters.

What is the difference between agglomerative and divisive clustering?

Agglomerative: Bottom-up, merging clusters

Divisive: Top-down, splitting clusters

What is a dendrogram?
A tree diagram showing the hierarchy of clusters and their merging steps.

How do you interpret a dendrogram?
Cut it horizontally at a given height to form clusters; closer branches = more similar.

What is linkage? Name types of linkage methods.
Linkage determines how distance between clusters is calculated.
Types: Single, Complete, Average, Ward’s.
What is the difference between complete and single linkage?

Complete: Max distance between points in clusters
Single: Min distance between points

What distance metric did you use and why?
Euclidean distance—most common for numerical features and easy to interpret.

What happens if we cut the dendrogram at a higher height?
We get fewer clusters (more general groupings).

How does hierarchical clustering differ from K-Means?
No need to predefine number of clusters and better for hierarchical data; however, it's slower for large datasets.
Is hierarchical clustering suitable for large datasets?
Not ideal—computationally expensive (O(n²) or worse).
 
          
          
          
          """)