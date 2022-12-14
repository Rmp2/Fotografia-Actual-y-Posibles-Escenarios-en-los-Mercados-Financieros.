![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

## INTRO

In the following project it will be processed and analyzed the relationship between six key elements of the global economic environment: 
-	The stocks represented in the S&P 500
-	The world growth represented by the MSCI World ETF
-	The gold prices
-	The Brent oil prices
-	The Inflation Index
-	The Interest Rate (set by the FED in the United States)

## DATA CLEANING
Therefore, the first part of the project is focused in the extraction, treatment and loading of the data. In order to build the final database, we used several sources such as: Web Scrapping to obtain the Inflation Index from an economical specialized portal, CSV import through Pandas library from specialized sources…etc. 
Following this, we proceeded to clean up, harmonize, normalize and summarize the data in a single database in order to achieve one standardized Dataframe to work with.
This first segment (and almost the entire project) has been made through Python language, using a wide range of libraries: NumPy, Pandas, Seaborn, Matplotlib…etc.
STATS AND VISUALIZATION
After that, once we have the final Dataframes, we use kit to build a statistic model and visualization through Matplotlib, Seaborn, Plotly and the software Microsoft PowerBi.
Hence, in this part we start to elaborate our analysis:
So, we start making a correlation matrix towards achieve a proper general view on our variables. Later we start making some relations through different kinds of visualization, such as bar plots, line plots, scatterplots…etc.
On this part, we also construct a statistics summary that give us the tools for the next part. In particular, we elaborate different Fibonacci Retracement models of the Gold, Brent and S&P 500. Which shows the resistance and the trends of the index. It also helps us to take an idea in case we want to build a machine learning model.

## MACHINE LEARNING
In this final part of our project, we continue using our constructed database to try to elaborate machine learning models. Concretely, we make a linear test that shows us the possibilities of this path. However, we decide to try a XGBRegressor model, which is more accurate for this purpose. Thus, we construct two models to try to predict the price of gold and Brent Oil.
NEXT STEPS
Due to the nature of this final project, we did not have the time to improve those models, so, the next stages of the project would be to adapt properly those machine learning. And finally, to be able to choose between them and take the more powerful. 
