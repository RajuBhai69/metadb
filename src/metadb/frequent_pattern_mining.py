def frequent_pattern_miningmetadb():
    print("""
          #Association Rule Mining
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

df = pd.read_csv('Groceries_dataset.csv')
df['Transaction'] = df['Member_number'].astype(str)+'_'+df['Date']
transactions = df.groupby('Transaction')['itemDescription'].apply(list)
tr = TransactionEncoder()
te_arr = tr.fit(transactions).transform(transactions)
transformed = pd.DataFrame(te_arr,columns=tr.columns_)

frequent_itemsets =apriori(transformed,min_support=0.001,use_colnames=True)
rules = association_rules(frequent_itemsets,metric='confidence',min_threshold=0.05)
top_rules = rules.sort_values(by='lift',ascending=False).head(10)
print(top_rules[['antecedents','consequents','support','confidence','lift']])

frequent_items = df.groupby('itemDescription').count().head(10)
frequent_items.plot(kind='bar')
plt.title('Frequent Items')
plt.xlabel('Item Description')
plt.ylabel('Count')
plt.show()


theory:
          """)