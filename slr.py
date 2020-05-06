import numpy as np
import pandas as pd
import pickle


calories = pd.read_table(r'D:\DATA SCIENCE COURSE\Assignments\Simple Linear Regression\calories_consumed.csv',sep=',')

calories.rename(columns={'Weight gained (grams)':'weight','Calories Consumed':'calories'},inplace=True)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()

calories
x=calories.iloc[:,1:2].values
y=calories.iloc[:,0].values

regression.fit(x,y)
regression.predict([[1500]])


pickle.dump(regression, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1500]]))




