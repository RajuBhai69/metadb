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
    
          
          theory:
          What is the Apriori algorithm? How does it work?
Apriori is an algorithm to find frequent itemsets and generate association rules. It uses a bottom-up approach, extending frequent itemsets one item at a time (candidate generation) and pruning those that don’t meet minimum support.

Define support, confidence, and lift.

Support: Frequency of an itemset in all transactions.

Confidence: Likelihood of occurrence of Y given X in a rule X ⇒ Y.

Lift: How much more likely Y is to occur with X than without.

What is the role of minimum support in association rule mining?
It filters out infrequent itemsets, reducing the number of rules and focusing on more relevant patterns.

Why do we sort rules by lift?
Lift indicates the strength and usefulness of a rule. A higher lift means a stronger association.

How do you interpret a rule like {butter} ⇒ {bread}?
It means customers who buy butter are likely to buy bread too, based on historical data.

What are frequent itemsets?
Itemsets that appear in the dataset at least as often as the minimum support threshold.

What does a lift value > 1 signify?
It means the occurrence of X increases the likelihood of Y; the rule is positively correlated.

How is Apriori different from FP-Growth?
Apriori generates candidates explicitly and scans data multiple times, while FP-Growth builds a tree structure to find frequent itemsets without candidate generation.

What type of datasets are suitable for association rule mining?
Market basket, transactional, or categorical datasets with discrete items.

Can we generate rules from 1-item frequent sets?
No, association rule
          """)