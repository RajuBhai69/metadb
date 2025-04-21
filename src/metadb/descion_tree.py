def descion_treemetadb():
    print("""
          #Decision Tree Classifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,precision_score,f1_score,ConfusionMatrixDisplay

df = pd.read_csv('iris.csv')
df.drop(columns=['Id'],axis=1,inplace=True)

X = df.drop('Species',axis=1)
Y = df['Species']

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.2,random_state=42)
clf = DecisionTreeClassifier(criterion='entropy',random_state=42)
clf.fit(x_train,y_train)

plot_tree(clf,feature_names=x_train.columns)
plt.show()
predict = clf.predict(x_test)

print(accuracy_score(y_test,predict))
print(precision_score(y_test,predict,average='weighted'))
print(f1_score(y_test,predict,average='weighted'))
ConfusionMatrixDisplay.from_predictions(y_test,predict)
plt.show()



theory:

What is a decision tree and how does it work?
A flowchart-like structure that splits data based on feature values to classify input.

What’s the difference between Gini index and entropy?
Both measure impurity. Entropy uses logarithmic values, while Gini uses squared probabilities.

How does a decision tree decide which feature to split?
It selects the feature that gives the highest information gain or lowest Gini impurity.

What is overfitting in decision trees?
When the model learns noise from training data, performing poorly on unseen data.

How can we avoid overfitting?
By pruning the tree, limiting depth, or setting minimum samples per leaf.

What is pruning? What are its types?
Reducing tree size by removing nodes. Types: Pre-pruning (limit depth early), Post-pruning (trim after full tree).

What is the depth of a tree?
Number of levels from root to the deepest leaf.

What does a leaf node represent?
A final decision or predicted class label.

How do you interpret a decision tree diagram?
Follow the branches based on feature values until reaching a leaf (class prediction).

What will be the predicted class for sepal=5.1, petal=1.5?
Likely Setosa, based on typical Iris measurements.


          """)