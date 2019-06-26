print("Data Analysis with Python - DA0101EN")
import sys

print(sys.version)

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 1. Introduction
# Understanding Domain, Dataset
# Python package for data science
# Importing and Exporting Data
# Basic Insights from Datasets

# Q: Can we estimate the price of used car based on characteristics?
# Using Data Science to answer them
#
# Tom wants to sell a car, but does not know the price
# Data Analyst - find relevant Data
# Import it to Python
# Look at basic insights
# Data does not mean information

# Libraries used for Data science:::


# SCIENCTIFIC COMPUTING LIBRARIES
# 1. PANDAS Data Structures and tools
# Data Frame, 2D Table consisting of a row and collumn
# 2. NumPy Libraries =
# Array and Matrices, used for inouts and outputs
# Extended to objects for matrices and minor coding changes
# Able to perform fast array processing
# 3. SciPy
# Includes functions for advanced math problems
# As well as Data visualisation


# VISUALISATION LIBRARIES:
# 1. Matplotlib - Graphs and plots
# 2. Seaborn - plots: heat maps, time series, violin plots


# ALGORITHMIC Libraries
# Machine learning LIBRARIES
# 1. Scikit-learn
# Machine learning: regression, classiciations
# 2. Statsmodels: Explore data, estimate statistical models and perform statistical testsself.
#





# DATA WRANGLING
# Data Preprocessing how to Identify and handle missing values
# Data Normalisation
# Normalisation - centering and scaling
# Data Binning is useful for comparison between DATA
# 2 Methods to deal with missing data:
# 1 - Perform a replacement by replacing with the average of that collumn
# 2 - By Dropping the problematic rows or collumns via dropna

# Data Formating - means bringing data to make meaningful comparisonsself.
# Ensure data is cleaned: From unclean to clean
# Data Normalisation
# Is used to ensure data is fair between variables, ensuring fairer comparison.
# Attribute Income will intrisically influence the result more
#
#
# 1. Simple Feature Scaling = xnew = xold / xmax
# 2. Min Max =  xnew = (xold - xmin) / (xmax - xmin)
# 3. Z-score = xxnew = (xold - Average) / Standard Deviation
# No 3 hover between -3 and 3


# Binnings is grouping of valies into bins. We can use this group categorical and numerical valueself.
# To ensure that we have a good data diistribution.
# binwidth = int(( max(df["price"]) - min(df["price"]) ) / 4)
# bins = range( min(df["price"], max(df["price"]), binwidth )
# group_names = ['low','Medium','High']

# df['price-binned'] = pd.cut(df['price'],  bins, labels=group_names)
# Converting Categorical into numerical
# This can be done by setting new unique category
# Example Fuel: Gas 1 Diesel 0
# This technique is called one-hot enconding
# In pandas use pd.get_dummieis(df['fuel'])

# list.isnull() Detect missing values and generate True for the file with NaN
#
# Counting the list:
# for column in missing_data.columns.values.tolist():
#   print(column)
#   print( missing_data[column].value_counts())
#   print("")
#
# After finding which data are missing
# We need to find how to normalised the data
# In here there are a few methods:
# 1. Drop Data
# 1 A. Drop the row
# 1 B. Drop the column
# 2. Replace Data
# 2 A. Replace it by Mean
# 2 B. Replace it by Frequency
# 2 C. Replace it based on functions

# How to replace the NAN df (data frame) with the mean value
# avg_5=df['peak-rpm'].astype('float').mean(axis=0)
# df['peak-rpm'].replace(np.nan, avg_5, inplace= True)

# To see which values are there we use value counts:
# >>> df['num-of-doors'].value_counts()
# four    115
# two      86
# Name: num-of-doors, dtype: int64

# To check which frequency has the most count
# >>> df['num-of-doors'].value_counts().idxmax()
# 'four'

# #replace the missing 'num-of-doors' values by the most frequent
# df["num-of-doors"].replace(np.nan, "four", inplace = True)

# # simply drop whole row with NaN in "price" column
# df.dropna(subset=["price"], axis=0, inplace = True)
#
# # reset index, because we droped two rows
# df.reset_index(drop = True, inplace = True)


# ABOVE we have cleaned the data sets, now we need to check the DATA
# >>> df.dtypes
# symboling              int64
# normalized-losses     object
# make                  object
# fuel-type             object
# aspiration            object
# num-of-doors          object
# body-style            object
# drive-wheels          object
# engine-location       object
# wheel-base           float64
# length               float64
# width                float64
# height               float64
# curb-weight            int64
# engine-type           object
# num-of-cylinders      object
# engine-size            int64
# fuel-system           object
# bore                  object
# stroke                object
# compression-ratio    float64
# horsepower            object
# peak-rpm              object
# city-mpg               int64
# highway-mpg            int64
# price                 object
# dtype: object


# Convert Data to correct them

# df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
# df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
# df[["price"]] = df[["price"]].astype("float")
# df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
# print("Done")

# Afterwards

# >>> df.dtypes
# symboling              int64
# normalized-losses      int64
# make                  object
# fuel-type             object
# aspiration            object
# num-of-doors          object
# body-style            object
# drive-wheels          object
# engine-location       object
# wheel-base           float64
# length               float64
# width                float64
# height               float64
# curb-weight            int64
# engine-type           object
# num-of-cylinders      object
# engine-size            int64
# fuel-system           object
# bore                 float64
# stroke               float64
# compression-ratio    float64
# horsepower            object
# peak-rpm             float64
# city-mpg               int64
# highway-mpg            int64
# price                float64
# dtype: object
# >>>

# # DATA STANDARDISATION
# <!-- Your answer is below:
# # transform mpg to L/100km by mathematical operation (235 divided by mpg)
# df["highway-mpg"] = 235/df["highway-mpg"]
#
# # rename column name from "highway-mpg" to "highway-L/100km"
# df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
#
# # check your transformed data
# df.head()
# -->

# # DATA Normalisation
# df['length'] = df['length']/df['length'].max()
# df['width'] = df['width']/df['width'].max()
# Data Normalisation is the process of transforming variables into a similar rangeself.
# Using Data Normalisation, you are able to scale the variable so that the values range

# DATA Binning
# Is the process of transforming numerical variables into categorical bins for grouped analysisself.

# Convert Data tot the correct format
# df["horsepower"]=df["horsepower"].astype(float, copy=True)
# Make 4 bins of equal size bandwidth
# binwidth = (max(df["horsepower"])-min(df["horsepower"]))/4

# Build a bin Array
# >>> bins = np.arange(min(df["horsepower"]), max(df["horsepower"]),binwidth)
# >>> bins
# array([ 48. , 101.5, 155. , 208.5])

# Set the 3 Group group_names
# group_names = ['Low', 'Medium', 'High']
#
# df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names,include_lowest=True )
# df[['horsepower','horsepower-binned']].head(20)
#
# >>> df[['horsepower','horsepower-binned']].head(20)
#     horsepower horsepower-binned
# 0        111.0            Medium
# 1        111.0            Medium
# 2        154.0            Medium
# 3        102.0            Medium
# 4        115.0            Medium
# 5        110.0            Medium
# 6        110.0            Medium
# 7        110.0            Medium
# 8        140.0            Medium
# 9        101.0               Low
# 10       101.0               Low
# 11       121.0            Medium
# 12       121.0            Medium
# 13       121.0            Medium
# 14       182.0              High
# 15       182.0              High
# 16       182.0              High
# 17        48.0               Low
# 18        70.0               Low
# 19        70.0               Low
# >>>

# Creating Dummy Varibale for Regression analysis
# REgression Analysis only understand numbers,
# TO use this attribute we can convert fuel-type into indicator variables
# Where there is only 2 fuel types: Diesel and Gas


# >>> dummy_variable_1 = pd.get_dummies(df["fuel-type"])
# >>> dummy_variable_1.head()
#    diesel  gas
# 0       0    1
# 1       0    1
# 2       0    1
# 3       0    1
# 4       0    1
# >>> dummy_variable_1.rename(columns={ 'fuel-type-diesel':'gas', 'fuel-type-diesel':'diesel'}, inplace=True)
# >>> dummy_variable_1.head()
#    diesel  gas
# 0       0    1
# 1       0    1
# 2       0    1
# 3       0    1
# 4       0    1
# >>>


# # merge data frame "df" and "dummy_variable_1"
# df = pd.concat([df, dummy_variable_1], axis=1)
#
# # drop original column "fuel-type" from "df"
# df.drop("fuel-type", axis = 1, inplace=True)


# Exploratory Data Analysis (EDA)
# A step to analyze data to ensure that we gain a better understandiing of the dataself.
# values_count() is a method that summarise categorical data.
# Box Plot shows visual distribution of the data
# Lower and Upper extremene, where there is the 75th and 25th percentiles.
# Scatterplot is used to predict relationship between 2 variables.
# After depicting the 2 relationship, then you can make a more generalised assumptions
# As they have a linear relationship - if the price of a goes up, the price of b also goes up.


# .describe() Method will yield all statistical value of the data frame: value such as mean, average, standard deviation
# .groupby() method is used to create a pivot table, which can then be translated into a heat map.

# ANOVA Analysis of Variance
# is used to find different correlation between different groups of categorical variiable.
# ANOVA test returns two values: F test and P value
# F >  Large and SMALL P Value suggest that there is a strong corellationself.
# F < 1 and P Value larger than 0.05 -> Prices of Honda and Subaru are not significantly different
# THIS CAN be calculated using the f_oneway method.

# Correlation -
# Is a statistical metric for measuring to what extent different variables are interdependent.
# RAIN -> UMBRELLA
# Lung Cancer -> Smoking.

# Correlation does not impose causation.
# A rainy weather does not mean everyone will buy an umbrella.

# Steep correlation of a linear correlation, shows that there is a positive relationship.
# Negative linear relationship -
# This mean that the value of a variable still have an impact on the value of prices.
# There is also an example where there is a weak correlation, where data is spread throughoutself.

# Pearson Correlation method
# Is a method to measure the strength of the correlation between continuous variables.
#
# Pearson gives u 2 values:
# P Value
# - PValue < 0.001 Strong centrainty in result
# - PValue < 0.05 Moderate certainty in result
# - Pvalue < 0.1 Weak certainty in result
# - Pvalue > 0.1 No Certainty in result
# Note; Result is the correlation

# Correlation coefficient
# - Close to +1 mean large positive relationship
# - Close to -1 mean Large Negative relationship
# - Close to 0 mean no relationship

# After establishing a strong correlation, then u can create a heatmap.
# It makes sense, because the value on the diagonal are the correlation
# How these variables are related to price.

# Finding Correlation
# >>> df[['bore','stroke','compression-ratio','horsepower']].corr()
#                        bore    stroke  compression-ratio  horsepower
# bore               1.000000 -0.055390           0.001263    0.566936
# stroke            -0.055390  1.000000           0.187923    0.098462
# compression-ratio  0.001263  0.187923           1.000000   -0.214514
# horsepower         0.566936  0.098462          -0.214514    1.000000
# >>>


# We can analyse the positive linear relationship of 2 numerical variables via a scatterplot
# Scatterplot Diagram :

# # Engine size as potential predictor variable of price
# sns.regplot(x="engine-size", y="price", data=df)
# plt.ylim(0,)

# Using scatterplot to determin Engine size as potential predictor of variable price.
# as Engine size goes up, Price goes up: it indicates a positive direct correlation between 2 variables.

# Negative Correlation of two variables, shows that as highway-mpg goes up, price goes down. It inidiciates a negative relationship.
# This shows that it can potentially be a prediictor of a price.
# - 0.704

# Weak Correlation shows that there exist no relationship between A and B.
# usually weak correlation is defined when the .corr() value is close to 0

# Categorical Variiables
# Variables that describe a characteristic of a data unit
# Selected from Small Group of Categories
# Categoriical variables are visualised using boxplots
# >>> sns.boxplot(x='body-style',y='price',data=df)
# <matplotlib.axes._subplots.AxesSubplot object at 0x12b9fcef0>
# >>> plt.show()

# When you see the enginie location and priice graph. When you see it to be different enough and distinct enough for it to be taken as a good predictor of price.

# Examining drive-wheels and price, you'd see that the distribution of price and Wheel Drive can potentially be a predictor of price.

# 3 DESCRIPTIVE STATISTIICAL ANALYSIS
# Describe always skip object.
# If you want to include object, you need to place include=['object']

# Value Counts
# Is a method to understand how many units of each variiable there is.
# Can convert it to Data Frame
# >>> df['drive-wheels'].value_counts()
# fwd    118
# rwd     75
# 4wd      8
# Name: drive-wheels, dtype: int64
# >>> df['drive-wheels'].value_counts().to_frame()
#      drive-wheels
# fwd           118
# rwd            75
# 4wd             8
# >>> drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
# >>> drive_wheels_counts
#      drive-wheels
# fwd           118
# rwd            75
# 4wd             8
# >>> drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'},inplace=True)
# >>> drive_wheels_counts
#      value_counts
# fwd           118
# rwd            75
# 4wd             8
# >>>
# Note: To be able to determine the success of such statistics, we need to ensure that we have enough  data sets

# GroupBY
# grouping results
# df_gptest=df[['drive-wheels','body-style','price']]
# grouped_test1=df_gptest.groupby(['drive-wheels','body-style'],as_index= False).mean()
# grouped_test1

# Grouping of variables to ensure that the data matches.
# From grouping by we make sure that we eliminate redundant data as we take the average of the value.



# 5. CORRELATION AND CAUSATION
# Correlation measure the extent of interdependence between variables
# Causation measure the relationship between cause and effect between 2 variables

# Pearson Correlation
# It measures the linear dependence between two variables: X and Y.
# Resulting coefficient is a value between -1 and 1.

# P value is the probability that the correlation between 2 variables is significant.
# Normally, we choose a signifcance level of 0.05, which means that we are 95% confidence.
# By Convention:
# P Value < 0.001 = Strong Evidence that correlation is significant
# P Value < 0.05 =  Moderate Evidence that corellation is significant
# P Value < 0.1 = Weak Evidence that correlation is significant.
# P Value > 0.1 = No evidence taht correlation is signicant.
#
# pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)
#
#

# ANOVA
# ANOVA: Analysis of Variance
# Its a statistical method used to test whether there are significant differences between means of two or more groups.
# ANOVA returns two parameter.

# F-test score - Assumes means of group are the same, calculates how much means deviate from assumption
#
# P Value Score - tells how statistically significant is our calculated score
#
# If our price is strongly correlated with variable, ANOVA will return size-able F-test and small p-value.

# # ANOVA
# grouped_test2.get_group('4wd')['price']
# # ANOVA
# f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
#
# print( "ANOVA results: F=", f_val, ", P =", p_val)
# grouped_test2=df_gptest[['drive-wheels','price']].groupby(['drive-wheels'])
# grouped_test2.head(2)

# f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
#
# print( "ANOVA results: F=", f_val, ", P =", p_val )

# f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
#
# print( "ANOVA results: F=", f_val, ", P =", p_val)




# MODEL DEVELOPMENT - Machine learning
# 1. SIMPLE LINEAR REGRESSION
# 2. MULTIPLE LINEAR REGRESSION
# 4. POLYNOMIAL REGRESSION

# %. MODEL EVALUATION USING VISUALISATION - Polynomiial Regression and Pipelines
# %. R-squared and MSE for In-Sample Evaluation - Prediction and Decision Making
#
# Polynomial Regression
# Is a special method useful for describing a curveliniar relationship.
# Model can also be cubic, where the predictor variable is cubed.
# Higher order polynomial regressions
# Degree of regression makes big difference if u can result in a better fit.
# polyfit()
#

# Pipelines Library
# Normalisation -> Polynomial Transform -> Linear Regression
# Two important Measures to determine the fit
# To determine or measure MSE


# Measures for in-sample evaluation
# 2 Measures to determine the fit of a model are:
# 1. Mean squared Error (MSE)
# MSE works by finding the difference between actual value and predicted value, then square it.
# Then take the mean or average of all errors and divide by number of samples
# 2. R-Squared
# R Square is called coeffcient determination. Its a measure to determine how close the data
# Is compared to
# 1- (MSE)/ MSE of data points
# In this case because line is good fit.
# Mean squared error of the line is relatively large.
# R-squared value in python.
#
# Determine how accurate the prediction is and the decision making.
#
# 1. Linear Regression and Multiple Linear Regression
# Establish an understanding of the data and relationship between 2 variables.
# Independent vs dependent variables - (Dependent variable that we want to predict)
#
# Module for linear regression:
# from sklearn.linear_model import LinearRegression
#
# >>> Yhat = lm.predict(X)
# >>> Yhat[0:5]
# array([[16236.50464347],
#        [16236.50464347],
#        [17058.23802179],
#        [13771.3045085 ],
#        [20345.17153508]])
# >>>


#>>> Z = df[['horsepower','curb-weight','engine-size','highway-mpg']]
# >>>
# >>>
# >>> lm.fit(Z, df['price'])
# LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
#          normalize=False)
# >>> lm.initercept_
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# AttributeError: 'LinearRegression' object has no attribute 'initercept_'
# >>> lm.intercept_
# -15806.62462632922
# >>>
# >>>
# >>> lm.coef_
# array([53.49574423,  4.70770099, 81.53026382, 36.05748882])
# >>>

#
# Linear Regression, you can visualise it with a Regresion plot / Distribution Plot.
# Steps are as follows:
#
# Regression Plot
# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="highway-mpg", y="price", data=df)

# plt.ylim(0,)
#

# Multiple Linear Regression however, you need to visualise it via distribution plot:
#
# # VIA Distribution PLOT
# >>> Y_hat = lm.predict(Z)
# >>> plt.figure(figsize=(width,height))
# <Figure size 1200x1000 with 0 Axes>
# >>>
# >>> ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
# >>>
# >>> sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)
# <matplotlib.axes._subplots.AxesSubplot object at 0x12a60e7f0>
# >>>
# >>> plt.title("Actual vs Fitted Values for Price")
# Text(0.5, 1.0, 'Actual vs Fitted Values for Price')
# >>>
# >>> plt.xlabel("Price in dollars")
# Text(0.5, 0, 'Price in dollars')
# >>> plt.ylabel("Propertion of Cars")
# Text(0, 0.5, 'Propertion of Cars')
# >>> plt.show()
# >>>
#


# Polynomial Regression and Pipelines
#
# Polynomial regression is a case of the linear regression model or multiple linear regression models.
# The non-linear relationship is obtained by squaring or setting higher-order
#

path = 'https://ibm.box.com/shared/static/q6iiqb1pd7wo8r3q28jvgsrprzezjqk3.csv'
df = pd.read_csv(path)
df.head()


def PlotPolly(model,independent_variable,dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable,dependent_variabble,'.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
#    ax.set_axis_bgcolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

print("done")

#
# x = df['highway-mpg']
# y = df['price']
# print("done")
#
# f = np.polyfit(x,y,3)
# p = np.poly1d(f)
# # print(p)
# #
# # PlotPolly(p,x,y, 'highway-mpg')
#
#
#
# f1 = np.polyfit(x,y,11)
# p1 = np.poly1d(f1)
# print(p)
# # PlotPolly(p1,x,y,'Length')

#
# from sklearn.preprocessing import PolynomialFeatures
#
# pr = PolynomialFeatures(degree=2)
# print(pr)
# Z_pr = pr.fit_transform(Z)


def DistributionPlot(RedFunction,BlueFunction,RedName,BlueName,Title ):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()



def PollyPlot(xtrain,xtest,y_train,y_test,lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))


    #training data
    #testing data
    # lr:  linear regression object
    #poly_transform:  polynomial transformation object

    xmax=max([xtrain.values.max(),xtest.values.max()])

    xmin=min([xtrain.values.min(),xtest.values.min()])

    x=np.arange(xmin,xmax,0.1)


    plt.plot(xtrain,y_train,'ro',label='Training Data')
    plt.plot(xtest,y_test,'go',label='Test Data')
    plt.plot(x,lr.predict(poly_transform.fit_transform(x.reshape(-1,1))),label='Predicted Function')
    plt.ylim([-10000,60000])
    plt.ylabel('Price')
    plt.legend()


y_data=df['price']

x_data=df.drop('price',axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)


from sklearn.linear_model import LinearRegression

lre = LinearRegression()
lre.fit(x_train[['horsepower']],y_train)
lre.score(x_test[['horsepower']], y_test)

# Question 2 find the R squared

x_train2, x_test2, y_train2, y_test2 = train_test_split(x_data, y_data, test_size=0.9, random_state =0)
print(lre.fit(x_train2[['horsepower']], y_train2))
print(lre.score(x_test2[['horsepower']], y_test2))



# Cross Validation score
from sklearn.model_selection import cross_val_score
print("done")

Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print(Rcross)
print("Mean of product is", Rcross.mean(), "And the SD is:", Rcross.std())


Rc = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)
print(Rc[1])



from sklearn.preprocessing import PolynomialFeatures
pr1 = PolynomialFeatures(degree=2)
# Polynomial Transformation of degree 2

x_train_pr1 = pr1.fit_transform(x_train[['horsepower','curb-weight','engine-size','highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test[['horsepower','curb-weight','engine-size','highway-mpg']])


x_train_pr1.shape


poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train)
