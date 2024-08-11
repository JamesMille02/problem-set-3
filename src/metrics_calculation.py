import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics using model predictions DataFrame and genre counts.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
        genre_true_counts (dict): Dictionary of true genre counts.
        genre_tp_counts (dict): Dictionary of true positive genre counts.
        genre_fp_counts (dict): Dictionary of false positive genre counts.
    
    Returns:
        tuple: Micro precision, recall, F1 score.
        lists of macro precision, recall, and F1 scores.
    '''
        
    #assigns 0 for genres
    true_positives = {genre: 0 for genre in genre_list}
    false_positives = {genre: 0 for genre in genre_list}
    false_negatives = {genre: 0 for genre in genre_list}

    #iterates through rows in pred_df
    for index, row in model_pred_df.iterrows():
        #gets the actual genres
        actual_genres = eval(row['actual genres']) if isinstance(row['actual genres'], str) else row['actual genres']
        #gets predicted genre
        predicted_genre = row['predicted']
        
        #checks if pred genre is the actual genre
        if predicted_genre in actual_genres:
            #if true true positive gets 1
            true_positives[predicted_genre] += 1
        else:
            #if false, false positives get 1
            false_positives[predicted_genre] += 1
        
        #goes through genres in the list
        for genre in genre_list:
            #if genre is not in the list
            if genre not in actual_genres:
                #if true, false negatives get 1
                false_negatives[genre] += 1

    #adds the sum for each values to a variable
    tp_total = sum(true_positives.values())
    fp_total = sum(false_positives.values())
    fn_total = sum(false_negatives.values())

    #calcs micro values
    micro_precision = tp_total / (tp_total + fp_total) 
    micro_recall = tp_total / (tp_total + fn_total) 
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) 

    #calc macro values
    macro_precision = []
    macro_recall = []
    macro_f1 = []

    #gets genres in genre list and assigns 0s
    for genre in genre_list:
        tp = true_positives.get(genre, 0)
        fp = false_positives.get(genre, 0)
        fn = false_negatives.get(genre, 0)
        
        #calc values
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        #adds it to the index
        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)
    
    return micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1



def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    '''
    pred_rows = []
    true_rows = []
    
    for index, row in model_pred_df.iterrows():
        true_genres = eval(row['actual genres']) if isinstance(row['actual genres'], str) else row['actual genres']
        pred_genre = row['predicted']
        
        true_row = [1 if genre in true_genres else 0 for genre in genre_list]
        pred_row = [1 if genre == pred_genre else 0 for genre in genre_list]
        
        pred_rows.append(pred_row)
        true_rows.append(true_row)
    
    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)
    true_matrix = pd.DataFrame(true_rows, columns=genre_list)
    
    y_true = true_matrix.values.flatten()
    y_pred = pred_matrix.values.flatten()
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    return macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1
