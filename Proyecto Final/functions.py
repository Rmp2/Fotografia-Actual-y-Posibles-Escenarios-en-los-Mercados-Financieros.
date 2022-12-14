import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import sklearn.metrics
import statsmodels.api as sm
from statsmodels.formula.api import ols
sns.set()
from pandas_profiling import ProfileReport
from IPython.core.display import display
from pandas_profiling.report.presentation.flavours.widget.notebook import (get_notebook_iframe,)
import csv
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
import regex as re

import requests 
import pandas_datareader.data as web
from datetime import datetime
import matplotlib.pyplot as plt
#plt.style.use('Solarize_Light2')


def check_nan(df: pd.DataFrame) -> None:
    '''
    This function allows to check the nan in a dafa fram in pandas
    '''     
    nan_cols  = df.isna().mean()*100
    display(f'N nan cols: {len(nan_cols[nan_cols>0])}')
    display(nan_cols[nan_cols>0])
    plt.figure(figsize=(10,6))
    sns.heatmap(df.isna(),
               yticklabels=False,
               cmap='viridis',
               cbar = False)
    plt.show();


def summary_regression_model(x,y):
    '''
    This functions creates an accurate report for linear regression models

     x_const = sm.add_constant(x) # add a constant to the model
    
    modelo = sm.OLS(y, x_const).fit() # fit the model
    
    pred = modelo.predict(x_const) # make predictions
    '''
    
    x_const = sm.add_constant(x) # add a constant to the model
    
    modelo = sm.OLS(y, x_const).fit() # fit the model
    
    pred = modelo.predict(x_const) # make predictions
    
    print(modelo.summary())   

def print_corr(df):
    
    '''
    this functions plosts a correlation head map with pearson method
    '''
   
    correlation = df.corr(method='pearson')

    mask = np.zeros_like(correlation, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    f, ax = plt.subplots(figsize=(10, 12))

    cmap = sns.diverging_palette(180, 20, as_cmap=True)
    sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    return plt.show()


def writelstcsv(guest_list, filename):
    """Write the list to csv file."""

    with open(filename, "w", encoding='utf-8') as outfile:
        for entries in guest_list:
            outfile.write(entries)
            outfile.write("\n")


def report_pd(df, to_html:bool, name=str):

    '''
    This function implements the pandas data frame report
    If to_html is True, it will return an html report. If False, it will show the report 
    on your jupyter notebook
    '''
    x= ProfileReport(df)
    y= x.to_file(name)
    if to_html == True:
        return y
    else:
        return x
   
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    PLEASE COPY PASTE THE BELOW METHOD:
    cnf_matrix = confusion_matrix(y_test, neigh5_pre, labels=[0,1, 2])
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['0=0','50=1', '100=2'],normalize= False,  title='Confusion matrix')
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def k_perfect(X_train, y_train,X_test, y_test ):
    '''
    This function returns the optimal K for your K-nearest 
    classification. you have to introduce x and y for training and x-y test
    '''
    Ks = 10
    mean_acc = np.zeros((Ks-1))
    std_acc = np.zeros((Ks-1))

    for n in range(1,Ks):
        K = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
        yhat=K.predict(X_test)
        mean_acc[n-1] =accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

    mean_acc

    plt.plot(range(1,Ks),mean_acc,'g')
    plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
    plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
    plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
    plt.ylabel('Accuracy ')
    plt.xlabel('Number of Neighbors (K)')
    plt.tight_layout()
    plt.show()

def perfect_epsilon(df):
    '''
    This function finds the perfect epsilon for
    your BDSCAN clustering.
    Please note that the df needs to be normatized
    to make it work.
    '''
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(df)
    distances, indices = nbrs.kneighbors(df)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances);

def log_r_summary(x1, y):
    '''
    This function displays a summary for the logistic regression
    '''
    x = sm.add_constant(x1)
    reg_log = sm.Logit(y,x)
    results_log = reg_log.fit()

# Get the regression summary
    return results_log.summary()

def log_r_plot(x1,y, x_label=str, y_label=str):
    '''
    This function creates a plot for the logistic regression
    '''
    plt.scatter(x1,y,color = 'C0')

    # Don't forget to label your axes!
    plt.xlabel('Duration', fontsize = 20)
    plt.ylabel('Subscription', fontsize = 20)
    return plt.show()

def perfect_wcss_k_means(x, y, data):
    '''
    This function creates a loop to check the perfect
    WCSS value. The parameters are the range of clusters
    (x, y) and the data to be fitted.
    '''
    wcss=[]
    for i in range(x,y):
    
        kmeans = KMeans(i)
        kmeans.fit(x)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    return wcss

def elbow_method(x,y, wcss=list):
    '''
    This function return a plot for the 
    elbow method, another way to see the 
    most accurate K for the K-means
    method.

    The paramenter are the range, (x,y)
    and the wcss which needs to be calculated 
    with the prefect_wcss_k_means function.
    '''
    number_clusters = range(x,y) 
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    return plt.plot(number_clusters,wcss)

def api_to_json(name, file_name=str):
    json2 = json.dumps(name)

    # open file for writing, "w" 
    f = open(file_name,"w")

    # write json object to file
    f.write(json2)

    # close file
    return f.close()




def fib_retrace(df):
  
  # Fetch data
  df=df.set_index(df['Date'])

  # Fibonacci constants
  max_value = df['Close'].max()
  min_value = df['Close'].min()
  difference = max_value - min_value

  # Set Fibonacci levels
  first_level = max_value - difference * 0.236
  second_level = max_value - difference * 0.382
  third_level = max_value - difference * 0.5
  fourth_level = max_value - difference * 0.618

  # Print levels
  print('Percentage level\t Price')
  print('0.00%\t\t', round(max_value, 3))
  print('23.6\t\t', round(first_level, 3))
  print('38.2%\t\t', round(second_level, 3))
  print('50%\t\t', round(third_level, 3))
  print('61.8%\t\t', round(fourth_level, 3))
  print('100.00%\t\t', round(min_value, 3))

  # Plot Fibonacci graph
  plot_title = 'Fibonacci Retracement for '
  fig = plt.figure(figsize=(22.5, 12.5))
  plt.title(plot_title, fontsize=30)
  ax = fig.add_subplot(111)
  plt.plot(df.index, df['Close'])
  plt.axhline(max_value, linestyle='--', alpha=0.5, color='purple')
  ax.fill_between(df.index, max_value, first_level, color='purple', alpha=0.2)

  # Fill sections
  plt.axhline(first_level, linestyle='--', alpha=0.5, color='blue')
  ax.fill_between(df.index, first_level, second_level, color='blue', alpha=0.2)

  plt.axhline(second_level, linestyle='--', alpha=0.5, color='green')
  ax.fill_between(df.index, second_level, third_level, color='green', alpha=0.2)

  plt.axhline(third_level, linestyle='--', alpha=0.5, color='red')
  ax.fill_between(df.index, third_level, fourth_level, color='red', alpha=0.2)

  plt.axhline(fourth_level, linestyle='--', alpha=0.5, color='orange')
  ax.fill_between(df.index, fourth_level, min_value, color='orange', alpha=0.2)

  plt.axhline(min_value, linestyle='--', alpha=0.5, color='yellow')
  plt.xlabel('Date', fontsize=20)
  plt.ylabel('Close Price (USD)', fontsize=20)
    
  plt.xticks(['2017-12-11', '2018-12-31', '2019-12-31', '2020-12-31','2021-12-31', '2022-12-09'], ['2018', '2019', '2020', '2021','2022', 'Actual'],rotation=20)
  

def plot_candlestick(prices, figsize = (13, 10), ax = None):
    """
    Plots the candlestick of a pricing data.
    Credits: https://www.statology.org/matplotlib-python-candlestick-chart/ 
    """
    
    if ax == None:
        ax = plt.figure(figsize = figsize)

    #define width of candlestick elements
    width = .4
    width2 = .05

    #define up and down prices
    up = prices[prices.Close>=prices.Open]
    down = prices[prices.Close<prices.Open]

    #plot up prices
    plt.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color='green')
    plt.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color='green')
    plt.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color='green')

    #plot down prices
    plt.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color='red')
    plt.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color='red')
    plt.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color='red')

    plt.xlim([prices.index.min(), prices.index.max()])
    
    return plt.gcf()

def plot_fibonacci_retracement(df):
    """
    Plots the Fibonacci retracement of a pricing data.
    :param df: The pricing data.
    :return: The plot.
    """
    highest_swing = df['High'].idxmax()
    lowest_swing = df['Low'].idxmin()

    ratios = [0, 0.236, 0.382, 0.5 , 0.618, 0.786, 1]
    colors = ["black","r","g","b","cyan","magenta","yellow"]
    levels = []
    max_level = df['High'][highest_swing]
    min_level = df['Low'][lowest_swing]

    for ratio in ratios:
        if highest_swing > lowest_swing:
            levels.append(max_level - (max_level-min_level)*ratio)
        else:
            levels.append(min_level + (max_level-min_level)*ratio)

    for i in range(len(levels)):
        plt.hlines(levels[i], df.index[0], df.index[-1], label="{:.1f}%".format(ratios[i]*100),colors=colors[i], linestyles="dashed")