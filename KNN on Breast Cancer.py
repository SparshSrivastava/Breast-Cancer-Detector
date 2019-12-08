# Importing the dataset
from sklearn import datasets
breast_cancer=datasets.load_breast_cancer()
print(breast_cancer.target_names)

# Splitting the data into training and testing datasets
from sklearn import model_selection
X_train,X_test,Y_train,Y_test=model_selection.train_test_split(breast_cancer.data,breast_cancer.target,test_size=0.2,random_state=0)

# Writing the function to make the predictions 

from collections import Counter

# Defining the function predict_single to predict the class for the single row of the test data
def predict_single(X_train,Y_train,x_test,k):
    # distance to hold the Euclidean Distance from each point of x_test to the each i of X_train
    distance=[]
    for i in range(len(X_train)):
        distance_dummy=((X_train[i,:]-x_test)**2).sum()
        distance.append([distance_dummy,i])
    
    # Now we need to sort the distance array and take into consideration only the first k
    distance=sorted(distance)
    
    # target array to take the value of each class predicted
    target=[]
    
    for i in range(k):
        number_of_training_data=distance[i][1]
        target.append(Y_train[number_of_training_data])
    
    # Now using the inbuilt function to count the majority of the class present in target to predict the one for the test data
    return Counter(target).most_common(1)[0][0] # As it returns a tuple so taking the first value which will be the required class    
    

def predict(X_train,Y_train,X_test,k):
    predictions=[]
    for x_test in X_test:
        predictions.append(predict_single(X_train,Y_train,x_test,k))
    return predictions


# Using Cross Validation to find the Optimal value of K
from sklearn.metrics import accuracy_score # To find the accuracy of the model

x_axis=[] # To be used for plotting of values of K and score of the model
y_axis=[] # To be used for plotting of values of K and score of the model

for i in range(1,26,2):
    y_pred=predict(X_train,Y_train,X_test,i)
    score=accuracy_score(y_pred,Y_test)
    x_axis.append(i)
    y_axis.append(score)

import matplotlib.pyplot as plt

plt.plot(x_axis,y_axis)

# It becomes constant at around K=9 therefore taking the optimal value of K to be 9

# Making predictoins using our implemented model
Y_pred_scratch=predict(X_train,Y_train,X_test,9)

# Checking the accuracy of our model
print(accuracy_score(Y_test,Y_pred_scratch))


# Now using the inbuilt model to check our model against it
from sklearn.neighbors import KNeighborsClassifier
clg=KNeighborsClassifier()

# Applying Cross Validation on this to find the optimum value of k
 from sklearn.model_selection import cross_val_score
 
 x_axis=[]
 y_axis=[]
 
 for i in range(1,26,2):
     clg=KNeighborsClassifier(n_neighbors=i)
     score=cross_val_score(clg,X_train,Y_train)
     x_axis.append(i)
     y_axis.append(score.mean()) # As by default it is 3Fold so will return 3 values and we need to take the mean of them

plt.plot(x_axis,y_axis)

# This plot also gives the optimal value of K as 9

# Making the predictions against the inbuilt model
clg=KNeighborsClassifier(n_neighbors=9)

clg.fit(X_train,Y_train)

Y_pred_inBuilt=clg.predict(X_test)

print(clg.score(X_test,Y_test)) # Checking the accuracy

# Thus both our model are giving the same accuracy. 