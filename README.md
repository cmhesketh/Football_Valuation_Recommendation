![Alt text](/Images/Mbappe_Haaland_.png)

## Machine Learning Problem: <br>
Identify undervalued football players who are statistically similar to household names. <br>

### Data <br>
All data can be found here: https://drive.google.com/drive/folders/1ePqWsQ24m9TTHsZbgTmQ0nIz-uSTCAWR?usp=sharing

This comprises of:  <br>
•	Player Data: Top 7 Leagues - 2011/2017 - 2022/2023 <br>
•	Team Data: Top 7 leagues - 2011/2017 - 2022/2023 <br>
•	Transfer Data: <br>
•	TransferMarkt scrape: 2009-2021 (2015) <br>
•	Kaggle: 2022 (Leading up to 2022/2023 season) <br>
•	Market Value Data & Player Data: TransferMarkt <br>

### The problem area: <br>
•	Football transfers are becoming increasingly expensive, and budgets are more restricted due to Financial Fair Play rules (Only spend based on your earnings). <br>
•	Uncertainty on new player performance is a significant risk for clubs both on the pitch but also financially on the balance-sheet.<br>
•	Young players provide a long-term investment but are unpredictable and risky in terms of predicting their future skills / value.<br>
•	Scouting players is time intensive and subjective, statistics provide an opportunity to narrow this down.<br>
•	Data volumes are increasing significantly through wearables, AI, pattern recognition enabling more accuracy / data points in predictive models.<br>

### Purpose: <br>
The end goal is for football teams to be able to identify a player of interest, and shortlist suitable alternatives who are statistically similar, but undervalued in the market.<br>
The approach followed is to: <br>
•	Cluster Football players into groups and positions based on their individual statistics<br>
 •	Predict football player valuations based on their individual and team statistics<br>
 •	Generate player recommendations through finding statistically similar players<br>

### The user: <br>
•	All football teams and scouting departments<br>
•	Players of football games<br>

### Project Organisation: <br>
Notebooks: 
1. Data Preprocessing & Cleaning<br>
2. Basic EDA: Univariate and Bivariate analysis, question driven EDA<br>
3. Further EDA: Feature Engineering & Pre-Processing:  Feature Importance, Engineering, and baseline linear models<br>
4. Modelling: Player Clustering based on individual statistics<br>
5. Modelling: Model comparison for Market Value Predictions<br>
6. Modelling: Modelling with Log of Y variable<br>
7. Modelling: Player Recommendation Engine -  Cosine and Euclidean distance similarity and recommendations<br>

Streamlit Application: 
1. new_app.py - This code generates a streamlit application talking through the project phases and enabling the recommendation engine (See screenshots below)

### Project Approach: <br>
The project followed an iterative approach to modelling, with outputs from clustering and detailed models, feeding the final recommendation engine. This enables clubs to search for players similar to a player of interest, and also identify if they are undervalued in the market and by how much. 

![Alt text](/Images/Process_Capstone.png)


### Findings: <br>
The following summarises the outcomes from the analysis. I have highlighted the outputs of the three areas of Clustering, Modelling, and Recommendations. 

#### Clustering:  <br>
Multiple approaches to clustering were explored including K-means and Agglomerative, both t-SNE and UMAP algorithms were also considered visualise the data points in 2D. <br>
![Alt text](/Images/Agglomerative_Clustering.png)

#### Modelling: 
The best regression model was XGBoost, other models were compared including Linear Regression, Ridge, Lasso, Decision Trees, KNN and Random Forests. 

![Alt text](/Images/Modelling_1.png)

![Alt text](/Images/Modelling_2.png)

![Alt text](/Images/Modelling_3.png)

![Alt text](/Images/Modelling_4.png)

![Alt text](/Images/Modelling_5.png)

#### Recommendations: 
The recommendation engine takes inputs from the clustering and machine learning models to generate a shortlist of players who are statistically similar to a player of interest. 
Utilising the streamlit application, teams can search for their player of interest and specify parameters on age, budget, league, and position to find other statistically similar players who are undervalued in the market place. 

![Alt text](/Images/Recommendation_1.png)

![Alt text](/Images/Recommender_1.png)

![Alt text](/Images/Recommendation_2.png)

