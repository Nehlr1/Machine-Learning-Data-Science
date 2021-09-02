"""
Pandas Coding

"""

import pandas as pd

"""
Series

"""

Age = pd.Series([10,20,30,40], index=['age1', 'age2', 'age3', 'age4'])

Filter_Age = Age[Age>10]

#Calling vakues of Age
Age.values

#Calling index of Age
Age.index

Age.index=['A1', 'A2', 'A3', 'A4']


"""
DataFrame

"""
import numpy as np

DF = np.array([[25,57,98], [25,7,78], [24,27,45], [45,7,7]])

Data_Set = pd.DataFrame(DF, index=['S1','S2','S3','S4'], columns = ['Age', 'Grade1', 'Grade2'])

Data_Set['Grade3'] = [45,34,34,23]

Data_Set.loc['S2']

Data_Set.iloc[1, 3]

Data_Set.iloc[:, 0]

Filtered_Data = Data_Set.iloc[:, 1:3]

Data_Set.drop('Grade1', axis = 1)

Data_Set = Data_Set.replace(7, 15)

Data_Set = Data_Set.replace({25:15, 34:50})


Data_Set.head(2)

Data_Set.tail(2)

Data_Set.sort_values('Grade2', ascending = True)

Data_Set.sort_index(axis=0, ascending = False)

Data = pd.read_csv('Data_Set.csv')



















