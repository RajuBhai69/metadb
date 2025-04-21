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
 
          
          
          
          """)