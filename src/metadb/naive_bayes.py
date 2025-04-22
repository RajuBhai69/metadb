def naive_bayesmetadb():
    print("""
          #Naive Bayes Classifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,ConfusionMatrixDisplay
df = pd.read_csv('iris.csv')
df.drop(columns=['Id'],axis=1,inplace=True)

X = df.drop('Species',axis=1)
Y = df['Species']

x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.2,random_state=42)
gnb = GaussianNB()
gnb.fit(x_train,y_train)

predict = gnb.predict(x_test)
print(accuracy_score(y_test,predict))
print(precision_score(y_test,predict,average='weighted'))
print(recall_score(y_test,predict,average='weighted'))
print(f1_score(y_test,predict,average='weighted'))
ConfusionMatrixDisplay.from_predictions(y_test,predict)
plt.show()



theory:
What is the Naive Bayes assumption?
It assumes that all features are independent given the class label.

Why is it called “naive”?
Because it naively assumes feature independence, which rarely holds in real data.

Which Naive Bayes variant is used for text classification?
Multinomial Naive Bayes—it works well with discrete text data like word counts.

What is the formula for Bayes’ Theorem?

𝑃(𝐴∣𝐵) =𝑃(𝐵∣𝐴) ⋅ 𝑃(𝐴)/𝑃(𝐵)
P(A∣B)= P(B)
P(B∣A)⋅P(A)
 
What is Laplace smoothing and why is it used?
It adds 1 to each word count to avoid zero probability for unseen words.

Why is text preprocessing important in spam classification?
To clean and normalize data, improving accuracy (e.g., removing stopwords, punctuation).

How do we evaluate a spam classifier? What metrics do we use?
Accuracy, Precision, Recall, and F1-Score.


What does a confusion matrix show?
It shows the number of true positives, true negatives, false positives, and false negatives.

What is precision and recall?

Precision: TP / (TP + FP) – how many predicted spams were actual spam.

Recall: TP / (TP + FN) – how many actual spams were correctly predicted.

What will your model predict for the message: “Congratulations, you’ve won a free trip!”?
Likely SPAM, as it contains spam-like trigger words.

          
          
          """)
    
def naive_bayesmetadb2():
    print("""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
          
     # Load CSV
df = pd.read_csv('sentiment_analysis.csv')

# Extract text and labels
texts = df['text']
labels = df['sentiment']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Define a new sentence for prediction
new_sentence = ["I love this product! It's amazing."]

# Convert the sentence into TF-IDF format
X_new = vectorizer.transform(new_sentence)

# Predict sentiment
predicted_sentiment = model.predict(X_new)
print("Predicted Sentiment:", predicted_sentiment[0])

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

          
          
          """)