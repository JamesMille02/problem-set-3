'''
This is the main.py to run the data processing and metrics calculations
You will run this project from here, so make sure to set things up and return values accordingly
Hint: Pay attention to the print statements below for both variable names and kinda of values you should return in the other files
'''
import numpy as np
from src.preprocessing import load_data, process_data
from src.metrics_calculation import calculate_metrics, calculate_sklearn_metrics

def main():

    # Load data from CSV files
    model_pred_df, genres_df = load_data()
    
    # Process data to get genre counts and predictions
    genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts = process_data(model_pred_df, genres_df)
    
    # Calculate micro and macro metrics
    micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list = calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts)
    
    # Print micro metrics
    print("Micro-Precision:", micro_precision)
    print("Micro-Recall:", micro_recall)
    print("Micro-F1:", micro_f1)
    
    # Print macro metrics
    print("-" * 20)
    print("Macro-Precision:", np.mean(macro_prec_list))
    print("Macro-Recall:", np.mean(macro_recall_list))
    print("Macro-F1:", np.mean(macro_f1_list))
    
    # Calculate and print metrics using sklearn
    macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1 = calculate_sklearn_metrics(model_pred_df, genre_list)
    
    print("-" * 20)
    print("Macro-Precision:", macro_prec)
    print("Macro-Recall:", macro_rec)
    print("Macro-F1:", macro_f1)
    
    print("-" * 20)
    print("Micro-Precision:", micro_prec)
    print("Micro-Recall:", micro_rec)
    print("Micro-F1:", micro_f1)

if __name__ == "__main__":
    main()

