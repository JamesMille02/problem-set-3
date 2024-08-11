import pandas as pd

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    #loads the model_predictions csv and genres csv
    model_pred_df = pd.read_csv('data/prediction_model_03.csv') 
    genres_df = pd.read_csv('data/genres.csv') 

    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
    
    #gets all genres
    genre_list = genres_df['genre']
    
    #assigns variables
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}
    genre_fn_counts = {genre: 0 for genre in genre_list}


    #goes through rows in model_pred_df
    for _, row in model_pred_df.iterrows():
        true_genres = [row['actual genres']]
        pred_genres = [row['predicted']]
        
        #ppdates true genre counts
        for genre in true_genres:
            if genre in genre_true_counts:
                genre_true_counts[genre] += 1
        
        #updates true positive and false positive counts
        for genre in genre_list:
            if genre in true_genres and genre in pred_genres:
                genre_tp_counts[genre] += 1
            if genre in pred_genres and genre not in true_genres:
                genre_fp_counts[genre] += 1
            if genre in true_genres and genre not in pred_genres:
                genre_fn_counts[genre] += 1

    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts