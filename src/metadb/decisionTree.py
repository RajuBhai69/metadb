def decision_treemetadb():
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
    
def decision_treemetab2():
   print("""
         import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

# Display summary statistics
print("Summary Statistics:\n", X.describe())
print("\nClass Distribution:\n", y.value_counts())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree using entropy
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)


# Plot the full decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Full Decision Tree")
plt.show()

# Evaluate accuracy and show confusion matrix
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy (full tree): {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix - Full Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Predict for a custom input
custom_input = np.array([[5.1, 3.5, 1.5, 0.2]])
prediction = clf.predict(custom_input)
print("\nPrediction for [5.1, 3.5, 1.5, 0.2]:", iris.target_names[prediction[0]])

# Apply pruning with max_depth
pruned_clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
pruned_clf.fit(X_train, y_train)

# Evaluate pruned model
pruned_pred = pruned_clf.predict(X_test)
pruned_acc = accuracy_score(y_test, pruned_pred)
print(f"\nAccuracy (pruned tree, max_depth=3): {pruned_acc:.2f}")

# Plot pruned decision tree
plt.figure(figsize=(10, 6))
plot_tree(pruned_clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Pruned Decision Tree (max_depth=3)")
plt.show()
         """)