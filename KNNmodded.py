from sklearn import datasets
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import random
#from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

# Euclidean Distance
def euc(a,b):
    return distance.euclidean(a,b)

class MyKNN():

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self,X_test):
        predictions = []
        for i in X_test:
            label = self.closest(i)
            predictions.append(label)


        return predictions

    def closest(self,row):
        best_dist = euc(row,self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = euc(row,self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        
        return self.y_train[best_index]

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)

dtree = tree.DecisionTreeClassifier()
dtree.fit(X_train, y_train)

knn = MyKNN()
knn.fit(X_train,y_train)


pred1 = dtree.predict(X_test)
pred2 = knn.predict(X_test)

'''
for i in pred1:
    print pred1[i], " ",y_test[i]
'''

print "accuracy of decision tree is :", accuracy_score(pred1,y_test)
print "accuracy of my modified KNN:",accuracy_score(pred2,y_test)