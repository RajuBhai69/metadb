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
    
def hierarchial_clusteringmetadb2():
    print("""
          #hierarchical----
# Required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score
import scipy.cluster.hierarchy as sch

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
data = pd.read_csv(url)

# Drop categorical columns for clustering
X = data.drop(columns=["Region", "Channel"])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Hierarchical Clustering - Agglomerative
agg_cluster = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean')
labels_agg = agg_cluster.fit_predict(X_scaled)

# Plot Dendrogram
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='complete', metric='euclidean'))
plt.axhline(y=15, color='red', linestyle='--')
plt.title('Dendrogram (Complete Linkage + Euclidean Distance)')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Dimensionality Reduction for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize Agglomerative Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_agg, palette='Set2', s=60)
plt.title('Hierarchical Clustering (Agglomerative) - 2D PCA Plot')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Compare Clustering Results
ari = adjusted_rand_score(labels_agg, labels_kmeans)
print(f"Adjusted Rand Index between Agglomerative and KMeans: {ari:.2f}")
""")    