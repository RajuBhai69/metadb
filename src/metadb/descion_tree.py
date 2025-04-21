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
          """)