import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import requests
import json

wine_data = pd.read_csv('WineQT.csv')
X = wine_data.drop(columns=['quality'])
y = wine_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .33, random_state = 0)

model = RandomForestClassifier()

model.fit(X, y)

predictions = model.predict([ [7, .3, 0, 1.5, .05, 15, 30, .99, 3.5, .75, 14] ])

y_pred = model.predict(X_test)

pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
#print(model.predict([ [7,.3,0,1.5,.05,15,30,.99,3.5,.75,14, 0] ]))
#Linear Regression
''' 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json


wine_data = pd.read_csv('WineQT.csv')

correct_features = wine_data.drop(columns=['Id',])

X = correct_features.iloc[:,:-1].values
#X = wine_data.iloc[:, :-1].values
y = wine_data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state = 0 )

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([ [7,.3,0,1.5,.05,15,30,.99,3.5,.75,14, 0] ]))
'''