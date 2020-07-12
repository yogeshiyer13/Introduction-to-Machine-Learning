# Introduction-to-Machine-Learning
**Part 1 - Data Preprocessing -** 
Data preprocessing has the following parts: 
- Importing the libraries - Basic libraries which would be useful while processing the data such as numpy, pandas, matplotlib
- Importing the dataset- How to get data from .csv file
- Taking care of missing data - How to deal when some filed in the column is missing
- Encoding categorical data - Most machine learning algorithms require numerical input and output variables 
hence one hot encoding is used to convert categorical data to integer data.
- Splitting the data into training and test set
- Feature Scaling: It is a step of Data Pre Processing which is applied to independent variables or features of data. 
It basically helps to normalise the data within a particular range

**Part 2 - Simple Linear Regression -**
Simple linear regression is a linear regression model with a single explanatory variable. That is, it concerns two-dimensional sample points with one independent variable and one dependent variable (conventionally, the x and y coordinates in a Cartesian coordinate system) and finds a linear function (a non-vertical straight line) that, as accurately as possible, predicts the dependent variable values as a function of the independent variables.  
The simple linear regression model is represented by:  
![SLR](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%202%20-%20%20Simple%20Linear%20Regression/SLR.JPG)
- Simple Linear Regression has the following parts:
- Importing the libraries - Basic libraries which would be useful while processing the data such as numpy, pandas, matplotlib
- Importing the dataset- How to get data from .csv file
- Splitting the dataset into the Training set and Test set
- Training the Simple Linear Regression model on the Training set
- Predicting the Test set results
- Visualising the Training set results
- Visualising the Test set results

**Part 3 - Multiple Linear Regression -**
Multiple linear regression (MLR), also known simply as multiple regression, is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. The goal of multiple linear regression (MLR) is to model the linear relationship between the explanatory (independent) variables & response (dependent) variable.   
![MLR](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%203%20-%20Multiple%20Linear%20Regression/MLR.JPG)

- Multiple Linear Regression has the following parts:
- Importing the libraries - Basic libraries which would be useful while processing the data such as numpy, pandas, matplotlib
- Importing the dataset- How to get data from .csv file
- Encoding categorical data - Most machine learning algorithms require numerical input and output variables 
- Splitting the dataset into the Training set and Test set
- Training the Multiple Linear Regression model on the Training set
- Predecting the Test set results  

**Part 4 - Polynomial Regression -**
In the last section, we saw two variables in our data set were correlated but what happens if we know that our data is correlated, but the relationship doesn’t look linear? So hence depending on what the data looks like, we can do a polynomial regression on the data to fit a polynomial equation to it.  
![PR](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%204%20-%20Polynomial%20Regression/PR.JPG)

- Polynomial Regression has the following parts:
- Importing the libraries - Basic libraries which would be useful while processing the data such as numpy, pandas, matplotlib
- Importing the dataset- How to get data from .csv file
- Training the Linear Regression model on the whole dataset 
- Training the Polynomial Regression model on the whole dataset
- Visualising the Linear Regression results
- Visualising the Polynomial Regression results
- Visualising the Polynomial Regression results (for higher resolution and smoother curve)
- Predicting a new result with Linear Regression
- Predicting a new result with Polynomial Regression

**Part 5 - Support Vector Regression -**
Support Vector Regression (SVR) uses the same principle as SVM, but for regression problems.what if we are only concerned about reducing error to a certain degree? What if we don’t care how large our errors are, as long as they fall within an acceptable range? Here the SVR is quite useful.  
![SVR](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%205%20-%20Support%20Vector%20Regresssion/SVR.JPG)

- SVR has the following parts: 
- Importing the libraries
- Importing the dataset
- Feature Scaling
- Training the SVR model on the whole dataset
- Predicting a new result
- Visualising the SVR results
- Visualising the SVR results (for higher resolution and smoother curve)

**Part 6 - Decision Tree Regression -**
Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node (e.g., Outlook) has two or more branches (e.g., Sunny, Overcast and Rainy), each representing values for the attribute tested. Leaf node (e.g., Hours Played) represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called root node. Decision trees can handle both categorical and numerical data.   
![DTR](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%206%20-%20Decision%20Tree%20Regression/DTR.JPG)

- DTR has the following parts:
- Importing the libraries
- Importing the dataset
- Training the Decision Tree Regression model on the whole dataset
- Predicting a new result
- Visualising the Decision Tree Regression results (higher resolution)

**Part 7 - Random Forest Regression -**
Random forest is a Supervised Learning algorithm which uses ensemble learning method for classification and regression.
Random forest is a bagging technique and not a boosting technique. The trees in random forests are run in parallel. There is no interaction between these trees while building the trees.
It operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.  
![RFR](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%207%20-%20Random%20Forest%20Regression/RFR.JPG)

- RFR has the following parts:
- Importing the libraries
- Importing the dataset
- Training the Random Forest Regression model on the whole dataset
- Predicting a new result
- Visualising the Random Forest Regression results (higher resolution)

**Regression folder-**
Assuming all the features on the forst columns and all the dependent var in the last column.  

The Dataset which we we are using is UCI Machine Learning Repository.  
Dataset is called as: Combined Cycle Power Plant  

We are predicting the dependent var Energy output(PE) with the features given.  
We have several features in the dataset such as: Engine Temp(AT), Vaccuum(V), Ambient pressure(AP), Relative Humidity(RH). 

The following parts are being executed in this folder - 
Multiple Linear Regression(MLR), Polynomial Linear Regression(PLR), Support Vector Regression(SVR), Decision Tree Regression(DTR), Random Forest Regression(RFR).  

- This figure given below helps us in understanding which regression to use when in brief:  
![Regression](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Regression/Regression.JPG)

**Part 8 - Logistic Regression -**
Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression).  
   
![LR](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%208%20-%20Logistic%20Regression/LR.JPG)  
 - The graph looks like this:  
![LogReg](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%208%20-%20Logistic%20Regression/logreg.JPG)  

**Part 9 - K Nearest Neighbour -**
k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation.  

Both for classification and regression, a useful technique can be to assign weights to the contributions of the neighbors, so that the nearer neighbors contribute more to the average than the more distant ones. For example, a common weighting scheme consists in giving each neighbor a weight of 1/d, where d is the distance to the neighbor.  

The neighbors are taken from a set of objects for which the class (for k-NN classification) or the object property value (for k-NN regression) is known. This can be thought of as the training set for the algorithm, though no explicit training step is required.  

A peculiarity of the k-NN algorithm is that it is sensitive to the local structure of the data.  

![KNN](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%209%20-%20KNN/KNN.png)  

It has the following:  
- Importing the libraries
- Importing the dataset
- Splitting the dataset into the Training set and Test set
- Feature Scaling
- Training the K-NN model on the Training set
- Predicting a new result
- Predicting the Test set results
- Making the Confusion Matrix
- Visualising the Training set results
- Visualising the Test set results

**Part 10 - Support Vector Machines -**  
Let’s imagine we have two tags: red and blue, and our data has two features: x and y. We want a classifier that, given a pair of (x,y) coordinates, outputs if it’s either red or blue. We plot our already labeled training data on a plane:  
![LBD](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%2010%20-%20Support%20Vector%20Machine/Labelled%20data.JPG)  

A support vector machine takes these data points and outputs the hyperplane (which in two dimensions it’s simply a line) that best separates the tags. This line is the decision boundary: anything that falls to one side of it we will classify as blue, and anything that falls to the other as red.  
![SVM](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%2010%20-%20Support%20Vector%20Machine/SVM.JPG)  

It has the following: 
- Importing the libraries
- Importing the dataset
- Splitting the dataset into the Training set and Test set
- Feature Scaling
- Training the SVM model on the Training set
- Predicting a new result
- Predicting the Test set results
- Making the Confusion Matrix
- Visualising the Training set results
- Visualising the Test set results

**Part 11 - Kernel SVM -**  
Lets take an example of non-linear data is:  
![KSVM](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%2011%20-%20Kernel%20SVM/KSVM.JPG)  

In this case we cannot find a straight line to separate apples from lemons. So how can we solve this problem. We will use the Kernel Trick!
The basic idea is that when a data set is inseparable in the current dimensions, add another dimension, maybe that way the data will be separable. 
Just think about it, the example above is in 2D and it is inseparable, but maybe in 3D there is a gap between the apples and the lemons, maybe there is a level difference, so lemons are on level one and lemons are on level two. 
In this case we can easily draw a separating hyperplane (in 3D a hyperplane is a plane) between level 1 and 2.

It has the following: 
- Importing the libraries
- Importing the dataset
- Splitting the dataset into the Training set and Test set
- Feature Scaling
- Training the Kernel SVM model on the Training set
- Predicting a new result
- Predicting the Test set results
- Making the Confusion Matrix
- Visualising the Training set results
- Visualising the Test set results


**Part 12 - Naive Bayes -**  
Naive Bayes is a simple technique for constructing classifiers: models that assign class labels to problem instances, represented as vectors of feature values, where the class labels are drawn from some finite set. There is not a single algorithm for training such classifiers, but a family of algorithms based on a common principle: all naive Bayes classifiers assume that the value of a particular feature is independent of the value of any other feature, given the class variable. For example, a fruit may be considered to be an apple if it is red, round, and about 10 cm in diameter. A naive Bayes classifier considers each of these features to contribute independently to the probability that this fruit is an apple, regardless of any possible correlations between the color, roundness, and diameter features.

For some types of probability models, naive Bayes classifiers can be trained very efficiently in a supervised learning setting. In many practical applications, parameter estimation for naive Bayes models uses the method of maximum likelihood; in other words, one can work with the naive Bayes model without accepting Bayesian probability or using any Bayesian methods.

Despite their naive design and apparently oversimplified assumptions, naive Bayes classifiers have worked quite well in many complex real-world situations. In 2004, an analysis of the Bayesian classification problem showed that there are sound theoretical reasons for the apparently implausible efficacy of naive Bayes classifiers.[6] Still, a comprehensive comparison with other classification algorithms in 2006 showed that Bayes classification is outperformed by other approaches, such as boosted trees or random forests.

![BT](https://github.com/yogeshiyer13/Introduction-to-Machine-Learning/blob/master/Part%2012%20-%20Naive%20Bayes/Bayes%20Theorem.JPG)  

It has the following: 
- Importing the libraries
- Importing the dataset
- Splitting the dataset into the Training set and Test set
- Feature Scaling
- Training the Kernel SVM model on the Training set
- Predicting a new result
- Predicting the Test set results
- Making the Confusion Matrix
- Visualising the Training set results
- Visualising the Test set results



