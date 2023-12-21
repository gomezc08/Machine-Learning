#Importing the libraries 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')

#reading the dataset
dataset=pd.read_csv("Restaurant_Reviews.tsv", delimiter = "\t", quoting=3)
x = dataset.iloc[0:1000,0].values  #string review text in x
y = dataset.iloc[0:1000,1].values  #string review class in y
reviews = []   #creating a list to store a review
corpus = []     #creating a list to store corpus containing all the reviews 

for character in range(0, len(x)):    # processing of all reviews in for loop
    # removing any special characters or numbers from reviews
    #if word is not containing alphabets then it will be replaced by blank space otherwise passed         
    # as such 
    review = re.sub('[^a-zA-Z]', ' ', str(x[character])) 
    #[^a-zA-Z] means any character that IS NOT a-z OR A-Z
    # sub method return the string obtained by replacing the leftmost non-overlapping       
    #occurrences of the pattern in string by the replacement 
    #mentioned as second parameter
    # converting the text to lower case 
    review = review.lower()
    #split the sentences into words by split it over blank space 
    review = review.split()
    #extracting the root words and removing the stop words from the word in the review
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)  #processed words are joined to form a review text
    #all processed reviews are appended into a corpus 
    corpus.append(review)

# Creating the Bag of Words for converting strings to vector
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)

# CLASSIFIER 1: DECISION TREE
from sklearn.tree import DecisionTreeClassifier
treeClassifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
treeClassifier.fit(x_train, y_train)

y_pred_tree = treeClassifier.predict(x_test)

# CLASSIFIER 2: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
forest.fit(x_train, y_train)

y_pred_forest = forest.predict(x_test)

# CLASSIFIER 3: NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
naiveBayes = GaussianNB()
naiveBayes.fit(x_train, y_train)

y_pred_bayes = naiveBayes.predict(x_test)

# CLASSIFIER 4: KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 6, metric = "minkowski", p = 2)
KNN.fit(x_train, y_train)

y_pred_knn = KNN.predict(x_test)

# CLASSIFER 5: Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

# CLASSIFER 6: Support vector linear
from sklearn.svm import SVC
supportVectorLinear = SVC(kernel = "linear", random_state = 0)
supportVectorLinear.fit(x_train, y_train)

y_pred_svc_linear = supportVectorLinear.predict(x_test) 

# CLASSIFER 7: Support vector non-linear
from sklearn.svm import SVC
supportVector = SVC(kernel = "rbf", random_state = 0)
supportVector.fit(x_train, y_train)

y_pred_svc = supportVector.predict(x_test)

# PREDICTIONS
from sklearn.metrics import confusion_matrix
cm_tree = confusion_matrix(y_test, y_pred_tree)
cm_forest = confusion_matrix(y_test, y_pred_forest)
cm_naive = confusion_matrix(y_test, y_pred_bayes)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_svLinear = confusion_matrix(y_test, y_pred_svc_linear)
cm_svRbf = confusion_matrix(y_test, y_pred_svc)

from sklearn.metrics import accuracy_score
acc_tree = accuracy_score(y_test, y_pred_tree)
acc_forest = accuracy_score(y_test, y_pred_forest)
acc_naive = accuracy_score(y_test, y_pred_bayes)
acc_knn = accuracy_score(y_test, y_pred_knn)
acc_lr = accuracy_score(y_test, y_pred_lr)
acc_svl = accuracy_score(y_test, y_pred_svc_linear)
acc_svR =  accuracy_score(y_test, y_pred_svc)

print("Accuracy of Decision tree: " + str(acc_tree))
print("Accuracy of Random Forest: " + str(acc_forest))
print("Accuracy of Naive Bayes: " + str(acc_naive))
print("Accuracy of KNN: " + str(acc_knn))
print("Accuracy of Logistic Regression: " + str(acc_lr))
print("Accuracy of Support Vector Linear Classifier: " + str(acc_svl))
print("Accuracy of Support Vector Non-Linear Classifier: " + str(acc_svR))