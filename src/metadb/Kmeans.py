def Kmeansmetadb():
    print("""
          #K-Means Clustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

df.drop(columns=['CustomerID'],axis=1,inplace=True)
df['Gender'] = df['Gender'].map({'Male':0,'Female':1})
scale_factors = ['Gender','Annual Income (k$)','Spending Score (1-100)']

scaled_df = df.copy()
scaler = StandardScaler()

scaled_df[scale_factors] = scaler.fit_transform(scaled_df[scale_factors])

kmean = KMeans(n_clusters=3,random_state=42)
scaled_df['cluster'] = kmean.fit_predict(scaled_df)

sns.scatterplot(
    x=scaled_df['Annual Income (k$)'],
    y=scaled_df['Spending Score (1-100)'],
    hue=scaled_df['cluster']
)

plt.show()
df['cluster'] = scaled_df['cluster']
print(df.groupby('cluster').mean())






THeory:

What is K-Means Clustering?
An unsupervised algorithm that groups data into k clusters based on similarity.

How does the K-Means algorithm work?
Randomly initializes k centroids, assigns points to nearest cluster, recalculates centroids, and repeats until convergence.

What does the “k” in K-Means stand for?
The number of clusters we want to form.

How do you choose the best value of k?
Using the Elbow Method—find the point where adding more clusters doesn’t reduce within-cluster distance significantly.

What is the Elbow Method?
A plot of k vs. inertia (SSE); the ‘elbow’ point indicates optimal k.

Why do we normalize data before clustering?
To give all features equal weight; otherwise, features with larger scales dominate.

How does K-Means handle high-dimensional data?
Not very well—performance and visualization decline. Dimensionality reduction (e.g., PCA) is often used.

What happens if clusters are not spherical?
K-Means may fail because it assumes clusters are convex and spherical.

What are the limitations of K-Means?

Sensitive to initial centroids

Requires predefined k

Doesn’t handle outliers well

Assumes spherical clusters

Can K-Means be used for non-numeric data?
No, it works only with numeric data since it relies on Euclidean distance.


          """)
    
def Kmeansmetadb2():
    print("""
          #k means---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('mall_customers.csv')

# Select features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)', 
    y='Spending Score (1-100)', 
    hue='Cluster', 
    palette='viridis', 
    data=df, 
    s=100
)
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Cluster characteristics
print("\nCluster Analysis (Mean Values):")
print(df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())
          
          
          """)    
    