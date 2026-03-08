import pandas as pd  
import numpy as np
import seaborn as sns
from sklearn import linear_model 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt  

## Data pre  preocessing 

##loading the data from csv file to a pandas dataframe 

salary_data = pd.read_csv("salary_data.csv")

print(salary_data.head()) 

print(salary_data.shape) 

print(salary_data.isnull().sum())

# splitting the feature and target 

X = salary_data.iloc[:,:-1].values
Y = salary_data.iloc[:,1].values  

print(X)
print(Y) 

## split DATA 
x_train,x_test,y_train,y_test =train_test_split(X,Y, test_size= 0.33,random_state=2)  

# training the linear regression model 


class Linear_Regression():

   def __init__( self, learning_rate, no_of_iterations ) :
          
        self.learning_rate = learning_rate
          
        self.no_of_iterations = no_of_iterations

    # fit function to train the model

   def fit( self, X, Y ) :
          
        # no_of_training_examples, no_of_features
          
        self.m, self.n = X.shape
          
        # initiating the weight and bias
          
        self.w = np.zeros( self.n )  ##?
          
        self.b = 0
          
        self.X = X
          
        self.Y = Y


        # implementing Gradient Descent for Optimization
                  
        for i in range( self.no_of_iterations ) :
              
            self.update_weights()
              
        
      
    # function to update weights in gradient descent
      
   def update_weights( self ) :
             
        Y_prediction = self.predict( self.X )
          
        # calculate gradients  
      
        dw = - ( 2 * ( self.X.T ).dot( self.Y - Y_prediction )  ) / self.m
       
        db = - 2 * np.sum( self.Y - Y_prediction ) / self.m 
          
        # updating the weights
      
        self.w = self.w - self.learning_rate * dw
      
        self.b = self.b - self.learning_rate * db
          
      
    # Line function for prediction:
      
   def predict( self, X ) :
      
        return X.dot( self.w ) + self.b

model = Linear_Regression(learning_rate= 0.01,no_of_iterations= 100) 


model.fit(x_train,y_train) 

## printing the parameter values(weights and bias)

print('weight =',model.w[0])
print('bias = ',model.b) 


## y = 9514(x)+23697 
## salary = 9514(experience)+23697 
## best fit line equation 

## predict the salary value for test data 

test_data_prediction =model.predict(x_test) 
print(test_data_prediction)  


## visualizing 

plt.scatter(x_test,y_test,color= 'red') 
plt.plot(x_test,test_data_prediction,color = 'blue') 
plt.xlabel('Work Experience')
plt.ylabel('salary')
plt.title('salary  vs experience ')
plt.show()
