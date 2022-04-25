Exploratory Data Analysis
# Import libraryimport pandas as pd
import numpy as np
import string
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
%matplotlib inline# Load data
data = pd.read_excel('data.xlsx')# Rename names columns 
data.columns = ['label', 'messages']
Let’s see a description of our data:
data.describe()
 
Data Description 
Note that our data contains a collection of 5574 SMS and also we have only 2 label: ham and spam. Now, we create a column called ‘length’ to know how long the text messages are and then plot it against the label:
data["length"] = data["messages"].apply(len)
data.sort_values(by='length', ascending=False).head(10)
 
data.hist(column = 'length', by ='label',figsize=(12,4), bins = 5)
 
Histogram between lenght and label (Image by Author)
Note that through the histogram, we have been able to discover that spam messages tend to have more characters.
Most likely, most of the data you have come across is numeric or categorical, but what happens when it is of type string (text format)?
As you may have noticed, our data is of type string. Therefore, we should transform it into a numeric vector to be able to perform the classification task. To do this, we use bag-of-words where each unique word in a text will be represented by a number. However, before doing this transformation, we should remove all punctuations and then common words like: [‘I’, ‘my’, ‘myself’, ‘we’, ‘our’, ‘our’ , ‘ourselves’, ‘you’, ‘are’ …]. This process is called tokenization. After this process, we convert our string sequence into number sequences.
1.	Remove all punctuation: Suppose we have the following sentence:
**********Hi everyone!!! it is a pleasure to meet you.**********
and we want to remove !!! and .
First, we load the import string library and do the following:
message = "Hi everyone!!! it is a pleasure to meet you."message_not_punc = []for punctuation in message:
    if punctuation not in string.punctuation:
           message_not_punc.append(punctuation)# Join the characters again to form the string.
message_not_punc = ''.join(message_not_punc)
print(message_not_punc)>>> Hi everyone it is a pleasure to meet you
2. Remove common words:
To do that, we use the library nltk, i.e, from nltk.corpus import stopwords
Is important to know that stopwords have 23 languages supported by it (this number must be up to date). In this case, we use English language:
from nltk.corpus import stopwords# Remove any stopwords for remove_punc, but first we should to transform this into the list.message_clean = list(message_not_punc.split(" "))# Remove any stopwordsi = 0
while i <= len(message_clean):
 for mess in message_clean:
     if mess.lower() in stopwords.words(‘english’):
                  message_clean.remove(mess)
 
 i =i +1
 
print(message_clean)>>> ['Hi', 'everyone', 'pleasure', 'meet']
Thus, with the steps 1 and 2, we can create the following function:
def transform_message(message):
    message_not_punc = [] # Message without punctuation
    i = 0
    for punctuation in message:
        if punctuation not in string.punctuation:
            message_not_punc.append(punctuation)
    # Join words again to form the string.
    message_not_punc = ''.join(message_not_punc) 

    # Remove any stopwords for message_not_punc, but first we should     
    # to transform this into the list.
    message_clean = list(message_not_punc.split(" "))
    while i <= len(message_clean):
        for mess in message_clean:
            if mess.lower()  in stopwords.words('english'):
                message_clean.remove(mess)
        i =i +1
    return  message_clean
Now, we can apply the above function to our data analysis in the following way:
data['messages'].head(5).apply(transform_message)>>>
0    [Go, jurong, point, crazy, Available, bugis, n...
1                       [Ok, lar, Joking, wif, u, oni]
2    [Free, entry, 2, wkly, comp, win, FA, Cup, fin...
3        [U, dun, say, early, hor, U, c, already, say]
4    [Nah, dont, think, goes, usf, lives, around, t...
Name: messages, dtype: object

from sklearn.feature_extraction.text import CountVectorizer
CountVectorizer has many parameters, but we only use “analyzer”, which is our own previously defined function:
vectorization = CountVectorizer(analyzer = transform_message )X = vectorization.fit(data['messages'])
Now, we should transform the entire DataFrame of the messages in a vector representation. For that, we use transform function:
X_transform = X.transform([data['messages']])print(X_transform)
>>> :	:
  (0, 11383)	9
  (0, 11384)	20
  (0, 11385)	14
  (0, 11386)	2
  (0, 11387)	4
  (0, 11391)	11
  (0, 11393)	5
  (0, 11396)	1
  (0, 11397)	1
  (0, 11398)	18
  (0, 11399)	18
  (0, 11405)	2
  (0, 11408)	1
  (0, 11410)	1
  (0, 11411)	8
  (0, 11412)	7
  (0, 11413)	1
  (0, 11414)	1
  (0, 11415)	27
  (0, 11417)	3
  (0, 11418)	104
  (0, 11420)	9
  (0, 11422)	1
  (0, 11423)	7
  (0, 11424)	1


import TfidfVectorizer from sklearn.feature_extraction.text and then:
tfidf_transformer = TfidfTransformer().fit(X_transform)
To transform the entire bag-of-words corpus into TF-IDF corpus at once:
X_tfidf = tfidf_transformer.transform(X_transform)
print(X_tfidf.shape)>>> (5572, 11425)
Classification Model
Having the features represented as vectors, we can finally train our spam/ham classifier. You can use any classification algorithms . Here we use Support Vector Classification (SVC) algortihm.
First, we split the data into train and test data. We take 80 % (0.80) training data and 30% (0.30) test data and the we fit the model using SVC:
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['messages'], test_size=0.30, random_state = 50)    
clf = SVC(kernel='linear').fit(X_train, y_train)
Test model
To test the model we use X_test previously calculated:
predictions = clf.predict(X_test)
print('predicted', predictions)>>> predicted ['spam' 'ham' 'ham' ... 'ham' 'ham' 'spam']
The question that arises is: our model reliable across the entire data set?
For that, we can use SciKit Learn’s built-in classification report, which returns Precision, Recall, F1-Score and also Confusion Matrix
from sklearn.metrics import classification_report
print (classification_report(y_test, predictions))
 
Classification Report 
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,predictions))
 
Confusion Matrix 
