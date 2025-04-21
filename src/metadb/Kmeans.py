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

          """)