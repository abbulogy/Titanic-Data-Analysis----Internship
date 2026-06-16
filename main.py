#Task 1

#importing pandas, matplotlib and seaborn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#titanic.csv is added in folder via GitHub
#------------------DATA LOADING-------------------------------------------

df = pd.read_csv('titanic.csv')

#-----------Exploratory Data Analysis (EDA)-------------------------------

#data type inspection
print('The following are the data types: ')
print(df.dtypes)

#summary statistics
print('         ')
print('         ')
print('The following is the summary statistics of the dataset:')
print(df.describe())

#Missing Value Assessment
print('         ')
print('         ')
print('The following is the sum of missing values per column:')
print(df.isnull().sum())

#Unique Value Count
print('         ')
print('         ')
print('The following is the display of unique enteries')
print(df.nunique())

#----------------Data  Visualization---------
print('         ')
print('         ')

for col in df.columns:

    df[col].value_counts().plot(kind='bar')
    plt.title(col, size = 18)
    plt.ylabel(col)
    plt.xlabel('count')
    plt.xticks(rotation = 0)
    plt.tight_layout
    plt.show()

#creating scatter plot for numerical values
survived = df['Survived'] == 1
not_survived = df['Survived'] == 0
survived_passengers = df[survived]
dead_passengers = df[not_survived]
plt.scatter(dead_passengers['Age'], dead_passengers['Fare'], c = 'red')
plt.scatter(survived_passengers['Age'], survived_passengers['Fare'], c = 'blue')
plt.title ('Age vs Ticket Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

#---------Correlation Heatmap----------------

corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()
