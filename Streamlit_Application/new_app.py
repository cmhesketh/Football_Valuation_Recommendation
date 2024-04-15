import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from scipy.stats import norm
from mplsoccer import PyPizza, FontManager
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

st.title("GOAThunter" )
st.set_option('deprecation.showPyplotGlobalUse', False)

player_df_reduced = pd.read_csv('../Notebook_Data_Exports/player_df_EDA.csv')
graph_for_streamlit = pd.read_csv('graph_for_streamlit.csv')
agg_clust_df = pd.read_csv('streamlit_clustering.csv')
market_value_position = pd.read_csv('../Notebook_Data_Exports/player_value_final_pred_all_seasons.csv')
market_value_predictions = pd.read_csv('../Notebook_Data_Exports/market_value_predictions.csv')
xgb_overfitting = pd.read_csv('xgb_overfitting.csv')
market_value_predictions['under/over'] = np.where(market_value_predictions['Y_Pred']>market_value_predictions['Y_Actual'],
                                                  'Undervalued', 'Overvalued')

def home_page():    
    st.write('##### We shortlist undervalued football players who are statistically similar to your player of interest.')
    st.image('../Images/Mbappe_Haaland_.png', width=800)
    st.write('##### Project machine learning approach:')
    st.write('''
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1. Cluster Football players into groups and positions based on their individual statistics\n
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. Predict football player valuations based on their individual and team statistics\n
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3. Generate player recommendations through finding statistically similar players\n\n
    The following top-level process was followed, to build up the final product:
''', unsafe_allow_html=True)

    st.image('../Images/Process_Capstone.png', width=800)
    

def analysis_insights():
    st.write("### Analysis and Insights: Player Valuations")
    st.write('##### 1. How does player valuation differ across major European Leagues?')
    st.write('##### The average player in the Premier League is worth more than double any other major league')
    player_df_reduced1 = player_df_reduced.copy()
    player_df_reduced1['market_value_in_eur'] = round(player_df_reduced1['market_value_in_eur']/1000000,0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=player_df_reduced1, x='market_value_in_eur', y ='league', 
                showfliers=False, color = 'green'),
    ax.set_xlabel('Player Market Value (M)', fontsize = 12)
    ax.set_ylabel('League', fontsize = 12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
    st.pyplot(plt)

    st.write('##### 2. How does player valuation differ by team?')
    st.write('##### Although the distinction is clear by league, individual teams can be significant outliers') 

    leagues = ['Select league (optional)', 'Premier League', 'Eredivisie', 'Ligue 1', 'Bundesliga', 'La Liga',
           'Serie A', 'Liga NOS']
    selected_league = st.multiselect('Select league:', leagues, default='Premier League')

    filtered_df = player_df_reduced[(player_df_reduced['season'] == '2022/2023') & 
                                    (player_df_reduced['league'].isin(selected_league))]

    most_valuable_team = filtered_df.groupby(['Current Club', 'league'])['market_value_in_eur'].sum()
    most_valuable_team_df = most_valuable_team.reset_index()
    most_valuable_team_sorted = most_valuable_team_df.sort_values('market_value_in_eur', ascending=False).head(10)
    most_valuable_team_sorted['market_value_in_eur'] = round(most_valuable_team_sorted['market_value_in_eur']/1000000,0)
    most_valuable_team_sorted.rename(columns={'market_value_in_eur': 'Market Value (M)'}, inplace=True)

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Market Value (M)', y='Current Club', hue='league', data=most_valuable_team_sorted)
    plt.xlabel('Market Value (M)')
    plt.ylabel('Team')
    plt.title('Most Valuable Teams by Market Value and League')
    plt.legend(title='League')
    st.pyplot(plt)

    st.write('#### 3. How does age and position impact market value?')
    st.write('##### Market Value is significantly influenced by differing combinations of Age and Position')

    top_5_value = list(player_df_reduced.sort_values('market_value_in_eur', ascending = False).iloc[:1,].index)
    old_but_valuable1 = list(player_df_reduced[player_df_reduced['age']==36].sort_values('market_value_in_eur', ascending = False).head(1).index)
    old_but_valuable2 = list(player_df_reduced[player_df_reduced['age']==38].sort_values('market_value_in_eur', ascending = False).head(1).index)
    young_but_valuable1 = list(player_df_reduced[player_df_reduced['age']==19].sort_values('market_value_in_eur', ascending = False).head(1).index)
    young_but_valuable2 = list(player_df_reduced[player_df_reduced['age']==20].sort_values('market_value_in_eur', ascending = False).head(1).index)

    plt.figure(figsize = (30, 20))
    sns.scatterplot(data = player_df_reduced, x= 'age', y = 'market_value_in_eur', hue = 'position', palette='colorblind', s=100)

    indices_to_circle = top_5_value + old_but_valuable1 + old_but_valuable2 + young_but_valuable1 + young_but_valuable2

    # Extract the coordinates of the points to circle
    x_points = player_df_reduced['age'][indices_to_circle]
    y_points = player_df_reduced['market_value_in_eur'][indices_to_circle]
    player_names = player_df_reduced.loc[indices_to_circle]['full_name']

    for i, (x, y, txt) in enumerate(zip(x_points, y_points, player_names)):
        y_offset = -35 
        plt.annotate(txt, (x, y), textcoords="offset points", 
                     xytext=(0, y_offset), ha='center', fontsize = 30)

    # Plot circles around the specified data points
    plt.scatter(x_points, y_points, color='none', s=600, alpha=1, edgecolor='black', linewidth=2)
    plt.ylabel('Market Value (100m)', fontsize = 30)
    plt.yticks(fontsize = 30)
    plt.xlabel('Age', fontsize = 30)
    plt.xticks(fontsize = 30)
    plt.legend(fontsize = 30, scatterpoints=3)
    st.pyplot(plt)

    st.write('#### 4. Which features increase or decrease Market Value the most?')
    st.write('##### Features associated with forward players are most associated with increasing player value')

    plt.figure(figsize=(10,7))
    sns.set(style='whitegrid')  
    sns.barplot(x='Correlation Coefficient', y='Features', data=graph_for_streamlit, hue='Positive / Negative',
                dodge=False, palette={'Positive': 'Green', 'Negative': 'red'})
    plt.xlabel('Correlation Coefficient with Market Value')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.legend()
    st.pyplot(plt)

def clustering_section():
    st.write('### Clustering: Agglomerative Clustering')
    st.write('###### Statistically similar groupings enabled player position labelling & identified two interesting sub groups')
    plt.figure(figsize = (15,10))
    sns.scatterplot(data = agg_clust_df, x = 'U1', y = 'U2', hue = 'Position', palette='bright')
    plt.xlabel('UMAP - Feature 1', fontsize =15)
    plt.ylabel('UMAP - Feature 2', fontsize =15)
    st.pyplot(plt)

def modelling_section():
    st.write('### Final Model Evaluation & Insights: XGBoost Market Value Predictions')
   
    st.write('##### 1. Model Development: xgboost is notorious for overfitting on training data')
    st.write('###### The production of a model able to generalise to unseen data through increasing levels of regularisation')
    
    xgb_overfitting.drop(columns = 'Unnamed: 0', inplace = True)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=xgb_overfitting.set_index('Model'), marker='o', 
                palette=sns.color_palette("Greens", n_colors=3), linewidth = 2, dashes = False)
    plt.xlabel('Model Iterations', fontsize=12)
    plt.ylabel('RMSE (M)', fontsize=12)
    plt.legend(title='Score Type', fontsize=10)
    plt.grid(True)
    st.pyplot(plt)
    
    model1_params = {'xgb__learning_rate': 0.1, 'xgb__max_depth': 5, 'xgb__n_estimators': 500}
    model5_params = {'xgb__alpha': 1, 'xgb__gamma': 0.1, 'xgb__learning_rate': 0.1, 'xgb__max_depth': 5, 'xgb__min_child_weight': 10, 'xgb__n_estimators': 80}

    # Create expanders for each model
    with st.expander("Model 1: Hyperparameters"):
        st.write(model1_params)

    with st.expander("Model 5: Hyperparameters"):
        st.write(model5_params)

    st.write('##### 2. Model Accuracy: Player market value predictions from the final model')
    st.write('###### The model is relatively well balanced, with a tendency towards predicting below the actual value')
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='Y_Actual', 
        y='Y_Pred', 
        data=market_value_predictions, 
        hue='under/over',
        palette={'Undervalued': 'Green', 'Overvalued': 'red'}, 
        alpha=0.5
    )
    plt.plot(
        [market_value_predictions['Y_Actual'].min(), market_value_predictions['Y_Actual'].max()],
        [market_value_predictions['Y_Actual'].min(), market_value_predictions['Y_Actual'].max()], 
        'k--', lw=2
    ) 
    plt.xlabel('Actual Values (Million)')
    plt.ylabel('XGB: Predicted Values (Million)')
    plt.grid(True)
    plt.legend(title='Under / Over', loc='upper left')
    st.pyplot(plt)

    
    
    st.write('##### 3. Which features have the largest importance to the model?')
    st.write('###### Feature importance in XGBoost models is the contribution to the reduction of RMSE (Gain)')

    st.image('../Images/xgb_feature_importance.png', width=900)

    st.write('##### 4. Which players are the most under or overvalued based on the model?')
    st.write('###### Identification of which data points predicted the least accurately by the final model')


    leagues = ['Select league (optional)', 'Premier League', 'Eredivisie', 'Ligue 1', 'Bundesliga', 'La Liga',
       'Serie A', 'Liga NOS']
    selected_league = st.multiselect('Select league:', leagues, default='Premier League')
    
    df = market_value_position[(market_value_position['season']=='2022/2023') & (market_value_position['league'].isin(selected_league))]
    
    # Sort players by Prediction Difference to get top undervalued and overvalued players
    top_10_undervalued = df.sort_values('Prediction Difference', ascending = True).head(10)
    top_10_overvalued = df.sort_values('Prediction Difference', ascending=True).tail(10)

    # Concatenate top undervalued and overvalued players
    all_players_top10 = pd.concat([top_10_undervalued, top_10_overvalued])
    
    # Bar plot of under and overvalued players based on the model
    sns.set_style("whitegrid")
    # Define the colors for undervalued and overvalued players
    colors = {"Undervalued": "green", "Overvalued": "red"}
    # Create plot
    plt.figure(figsize=(11, 8))
    sns.barplot(x='Prediction Difference', y='full_name', data=all_players_top10, hue = 'Under / Over', palette=colors)
    plt.xlabel('Difference (in €Millions)')
    plt.ylabel('Player')
    plt.title('Market Value Difference of Players')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    st.pyplot(plt)


def recommendation_section():
    st.write("#### Similar player recommendation engine:")
    # Round Predicted Market Value and Prediction Difference
    market_value_position['Predicted Market Value'] = round(market_value_position['Predicted Market Value'],2)
    market_value_position['Prediction Difference'] = round(market_value_position['Prediction Difference'],2)
    market_value_position.rename(columns = {'market_value_in_eur (m)':'Actual Market Value (M)', 'Predicted Market Value': 'Predicted Market Value (M)'}, inplace = True)
    market_value_position_final = market_value_position[market_value_position['season']=='2022/2023']

    # Take a copy of player_df 
    player_df_reduced_to_scale = market_value_position_final.copy()

    # Separate columns - to join back - needed in later analysis
    columns_to_separate = ['full_name', 'age', 'league', 'season', 'Position_NEW', 'Current Club', 
                        'nationality', 'Actual Market Value (M)', 'annual_salary_eur',
                        'player_year_key', 'Predicted Market Value (M)', 'Prediction Difference', 
                        'Under / Over']

    # Df to keep (not to scale / use in similarity)
    player_df_reduced_keep = player_df_reduced_to_scale[columns_to_separate]

    # Scale other numerical columns
    player_df_reduced_to_scale1 = player_df_reduced_to_scale.drop(columns = columns_to_separate)

    # Standard Scalar
    rec_scaler = StandardScaler()
    x = rec_scaler.fit_transform(player_df_reduced_to_scale1)
    scaled_df = pd.DataFrame(x, columns = player_df_reduced_to_scale1.columns)

    # DF for recommendations 
    df_recommender = pd.concat([player_df_reduced_keep.reset_index(), scaled_df], axis=1)
    df_recommender = df_recommender.drop(columns = 'index')

    def getplayerstats(name):
        stats = df_recommender[df_recommender['full_name']==name].iloc[:1,15:]
        #club = player_df_reduced[player_df_reduced['full_name']==name]['Current Club'].iloc[0]
        return stats

    def comparetwoplayers(name1, name2):
        cosine_sim = cosine_similarity(getplayerstats(name1), getplayerstats(name2))
        return cosine_sim[0][0]

    def get_player_stats_plots(player_names):
        plots_list = []  # List to store the generated plots

        slice_colors = ["blue"] * 3 + ["green"] * 3 + ["red"] * 2
        text_colors = ["white"]*8
        font_normal = FontManager()
        font_bold = FontManager()
        
        for name in player_names:
            player_stats = market_value_position_final.loc[market_value_position_final["full_name"] == name]
            player_stats = player_stats.loc[player_stats['season'] == '2022/2023']
            team = list(player_stats['Current Club'])[0]
            league = list(player_stats['league'])[0]
            player_stats = player_stats[['npxg_per_90_overall', "goals_per_90_overall", "shots_on_target_per_90_overall",
                                        "assists_per_90_overall", "key_passes_per_90_overall", "dribbles_per_90_overall", 
                                        "duels_won_per_90_overall", "aerial_duels_won_per_90_overall"]]   
            
            per_90_columns = player_stats.columns[:]
            values = [round(player_stats[column].iloc[0], 2) for column in per_90_columns]
            percentiles = [int(stats.percentileofscore(df_recommender[column], player_stats[column].iloc[0])) for column in per_90_columns]
            

            # Create PyPizza plot for each player
            baker = PyPizza(
                params=per_90_columns,
                min_range=None,
                max_range=None,
                straight_line_color="#000000",
                straight_line_lw=1,
                last_circle_lw=1,
                other_circle_lw=1,
                other_circle_ls="-."
            )


            # Plot pizza for the player
            fig, ax = baker.make_pizza(
                percentiles,
                figsize=(4, 4),
                param_location=110,
                slice_colors=slice_colors,
                value_colors=text_colors,
                value_bck_colors=slice_colors,
                kwargs_slices=dict(
                    facecolor="cornflowerblue", edgecolor="#000000",
                    zorder=2, linewidth=1
                ),
                kwargs_params=dict(
                    color="#000000", fontsize=5,
                    fontproperties=font_normal.prop, va="center"
                ),
                kwargs_values=dict(
                    color="#000000", fontsize=5,
                    fontproperties=font_normal.prop, zorder=3,
                    bbox=dict(
                        edgecolor="#000000", facecolor="cornflowerblue",
                        boxstyle="round,pad=0.2", lw=1
                    )
                )
            )

            # Set values as text on the plot
            texts = baker.get_value_texts()
            for i, text in enumerate(texts):
                text.set_text(str(values[i]))

            # Add title
            fig.text(
                0.515, 0.97, f"{name} per 90 - {team} - {league}", size=10,
                ha="center", fontproperties=font_bold.prop, color="#000000"
            )

            # Add subtitle
            fig.text(
                0.515, 0.942,
                f"{league} | Season 2022-2023",
                size=6,
                ha="center", fontproperties=font_bold.prop, color="#000000"
            )

            plots_list.append(fig)
        
        return plots_list


    def most_similar_players_cos(name1, age, marketvalue, position = None, league = None):
        similarity_scores = []
        
    # Filter players based on criteria
        filtered_players = df_recommender[
            (df_recommender['age'] <= age) &
            (df_recommender['Actual Market Value (M)'] < marketvalue)
        ]
        
        if position is not None:
            filtered_players = filtered_players[filtered_players['Position_NEW'].isin(position)]
        
        if league is not None:
            filtered_players = filtered_players[filtered_players['league'].isin(league)]
        
        # If no players match - None
        if filtered_players.empty:
            return None
        
        all_players = filtered_players['full_name'].unique()
        
        # Calculate similarity score for player vs all other players
        for name in all_players:
            similarity_scores.append((
            name1, 
            name, 
            round(comparetwoplayers(name1, name)*100), 
            df_recommender[df_recommender['full_name'] == name]['age'].iloc[0], 
            df_recommender[df_recommender['full_name'] == name]['Current Club'].iloc[0], 
            df_recommender[df_recommender['full_name'] == name]['Actual Market Value (M)'].iloc[0], 
            df_recommender[df_recommender['full_name'] == name]['Predicted Market Value (M)'].iloc[0],
            df_recommender[df_recommender['full_name'] == name]['Prediction Difference'].iloc[0],
            df_recommender[df_recommender['full_name'] == name]['Under / Over'].iloc[0] 
        ))

        # Create DataFrame of similarity scores
        df_similarity = pd.DataFrame(similarity_scores, columns=['Player',
                                                                'Similar Player', 
                                                                'Cosine Similarity', 
                                                                'Age', 
                                                                'Current Club', 
                                                                'Actual Market Value (M)', 
                                                                'Predicted Market Value (M)',
                                                                'Prediction Difference',
                                                                'Under / Overvalued'])
        df_similarity['Similarity & Value'] = df_similarity['Cosine Similarity'] -df_similarity['Prediction Difference']
        
        df_similarity = df_similarity.sort_values('Cosine Similarity', ascending=False).head(5).reset_index(drop=True)
        with st.expander("Similar Player Details: Dropdown"):
            pd.set_option('display.max_columns', None) 
            st.dataframe(df_similarity)
        
        # Get player statistics plots
        player_stats_plots = get_player_stats_plots([name1] + list(df_similarity['Similar Player']))
    
        if player_stats_plots:
            st.title('Similar Player Plots')
        
        # Display each plot in the Streamlit app
        for idx, plot in enumerate(player_stats_plots):
            st.pyplot(plot)
        else:
            st.write('')
        return  


    # Usage in Streamlit
    player_name = st.selectbox('Enter player name:',
    (player_df_reduced['full_name'].unique()),
    index = None)   
    age = st.slider('Enter age:', min_value=16, max_value=40, value=30)
    market_value = st.slider('Enter market value (Millions):', min_value=0, value=200)
    leagues = ['Select league (optional)', 'Premier League', 'Eredivisie', 'Ligue 1', 'Bundesliga', 'La Liga',
       'Serie A', 'Liga NOS']
    selected_league = st.multiselect('Select league:', leagues)

    # Set league variable based on user input
    league = None if selected_league == "Select league (optional)" else selected_league
    
    positions = ['Select position (optional)', 'Central Midfielders', 'Young High Performers', 'Goalkeepers',
       'Defensive Midfielders / Wing Backs', 'Wingers',
       'Attacking Midfielders', 'Central Defenders', 'Strikers',
       'Strikers - Goalscorers']
    selected_position = st.multiselect('Select position:', positions)

    # Set league variable based on user input
    position = None if selected_position == "Select position (optional)" else selected_position

    if st.button('Find Similar Players'):
        most_similar_players_cos(player_name, age, market_value, position or None, league or None)

def contact_section():
    st.write("You are now in the Contact section.")
    st.write('### Linkedin:')
    st.write('https://www.linkedin.com/in/cris-hesketh/')
    st.write('### Email:')
    st.write('cmhesketh@hotmail.co.uk')
    st.write('### github:')
    st.write('https://github.com/cmhesketh/Football_Valuation_Recommendation')

# Define the main function to control the app
def main():
    st.sidebar.title("Navigation")
    selected_section = st.sidebar.radio("Go to:", ("Home Page", "Analysis & Insights", "Clustering: Player Positions",  
                                                   "Modelling: Player Valuations", "Recommender: Find Similar Players", 
                                                     "Contact: Cris Hesketh"))

    if selected_section == "Home Page":
        home_page()
    elif selected_section == "Analysis & Insights":
        analysis_insights()
    elif selected_section == "Clustering: Player Positions":
        clustering_section()
    elif selected_section == "Modelling: Player Valuations":
        modelling_section   ()
    elif selected_section == "Recommender: Find Similar Players":
        recommendation_section()
    elif selected_section == "Contact: Cris Hesketh":
        contact_section()

if __name__ == "__main__":
    main()




