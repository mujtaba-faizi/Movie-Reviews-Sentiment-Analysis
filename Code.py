from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import re
import numpy as np

def review_to_words(raw_review,n):#cleaning the reviews data
    example1 = BeautifulSoup(raw_review["review"][n], "lxml")
    a = example1.get_text()
    example_ = re.sub('[^A-Za-z0-9]+', " ", a)
    words = example_.lower().split()
    stops = set(stopwords.words('english'))
    words = [w for w in words if not w in stops]
    # Now join the words back into one string separated by :
    sentence = " ".join(words)
    return sentence

reviews = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
train=[]
validation=[]
for a in range(0,20000):#20000 rows for training data
    b=review_to_words(reviews,a)
    train.append(b)
for a in range(20000,25000):#remaining 5000 rows for validation data
    b=review_to_words(reviews,a)
    validation.append(b)
Y=[]
for c in range(0,20000):     #adding sentiment values for the training data reviews
    label = reviews["sentiment"][c]
    Y.append(label)
 # review labels. 1 indicate spam, 0 non-spam
# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None, preprocessor = None, stop_words = None, max_features = 3000)
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
X = vectorizer.fit_transform(train)

# Numpy arrays are easy to work with, so convert the result to an
# array
X = X.toarray()
clf = MultinomialNB(alpha=0.00001) # alpha=0 means no laplace smoothing
clf.fit(X, np.array(Y))

tX = vectorizer.transform(validation).toarray()
# prediction
tX=clf.predict(tX)
Y2=[]
for a in tX:
    Y2.append(a)        #predicted values

#finding accuracy
Y=[]
for c in range(20000,25000):     #adding correct sentiment values for the validation data reviews
    label = reviews["sentiment"][c]
    Y.append(label)
sum=0.00


for d in range(0,5000):
    sum=sum+abs(Y[d]-Y2[d])    #error finding
print (sum)
acc=100-((sum/5000)*100)
print ("Accuracy=",acc)