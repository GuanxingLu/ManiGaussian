'''
Usage:
python scripts/compute_results.py --file_paths ManiGaussian_results/w_geo/0.csv ManiGaussian_results/w_geo/1.csv ManiGaussian_results/w_geo/2.csv --method last
'''

import argparse
import pandas as pd
import numpy as np
from glob import glob
from termcolor import cprint
from collections import OrderedDict


TASKS = ['close_jar', 'open_drawer', 'sweep_to_dustpan_of_size', 'meat_off_grill', 'turn_tap', 'slide_block_to_color_target', 'put_item_in_drawer', 'reach_and_drag', 'push_buttons', 'stack_blocks']
CAT_GROUPS = ['Planning', 'Long', 'Tools' , 'Motion', 'Screw', 'Occulusion']

cat_group_to_task = OrderedDict({
    'Planning': ['push_buttons', 'meat_off_grill'],
    'Long': ['stack_blocks', 'put_item_in_drawer'],
    'Tools': ['slide_block_to_color_target', 'reach_and_drag', 'sweep_to_dustpan_of_size'],
    'Motion': ['turn_tap'],
    'Screw': ['close_jar'],
    'Occulusion': ['open_drawer'],
})


def calculate_average_return(df):
    """
    Calculate the average return for each checkpoint in the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame containing the data.
    
    Returns:
    Series: A Pandas Series containing the average return for each checkpoint.
    """
    # Extract columns containing 'return' in their names
    return_columns = [col for col in df.columns if 'return' in col]
    
    # Calculate the average return for each checkpoint
    df_returns = df[return_columns]
    # print df_returns with column shrink e.g., 'eval_envs/return/push_buttons' -> 'push_buttons'
    df_returns.columns = [col.split('/')[-1] for col in df_returns.columns]
    # do not fold the columns
    pd.set_option('display.expand_frame_repr', False)
    pd.options.display.float_format = '{:.1f}'.format

    avg_return = df_returns.mean(axis=1)

    # add index from 'step' column in the original df
    df_returns.insert(0, 'step', df['step'])

    # print all checkpoints and their returns on each task
    cprint("df_returns", 'green')
    print(df_returns)

    # calculate the average return for each category
    df_returns_cat = df_returns.copy()
    for cat, tasks in cat_group_to_task.items():
        df_returns_cat[cat] = df_returns[tasks].mean(axis=1)
    
    # remove original task columns
    df_returns_cat = df_returns_cat.drop(TASKS, axis=1)
    cprint("df_returns_cat", 'green')
    print(df_returns_cat)

    return avg_return


def main(file_paths, method='last'):
    """
    Main function to compute the average success rate.
    
    Parameters:
    file_paths (list): List of file paths for the seeds.
    method (str): Method to select the checkpoint ('best' or 'last').
    """
    # List to store average return for selected checkpoint across seeds
    selected_avg_return = []
    
    # Loop through each file and calculate the average return
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        avg_return = calculate_average_return(df)
        
        # Select the checkpoint based on the method
        if method == 'best':
            selected_return = avg_return.max()
            # get the index of the best checkpoint
            best_checkpoint = avg_return.idxmax()
            print(f"Best checkpoint for {file_path}: {best_checkpoint}, with average return: {selected_return:.2f}")
        elif method == 'last':
            # get the last checkpoint from step column
            last_checkpoint = df['step'].idxmax()
            print(f"last_checkpoint: {last_checkpoint}")
            selected_return = avg_return.iloc[last_checkpoint]

        elif method.isdigit():
            print(f"index method: {method}")
            method = int(method)
            selected_return = avg_return.iloc[method]
        else:
            print(f"Unknown method: {method}. Skipping this seed.")
            continue
        
        selected_avg_return.append(selected_return)
    
    # Calculate the average and standard deviation over all seeds
    avg_over_seeds = np.mean(selected_avg_return)
    std_over_seeds = np.std(selected_avg_return)
    
    cprint(f"Average return over all seeds: {avg_over_seeds:.2f}", 'cyan')
    cprint(f"Standard deviation over all seeds: {std_over_seeds:.2f}", 'cyan')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate average return across seeds and tasks.')
    parser.add_argument('--file_paths', nargs='+', required=True, help='List of file paths for the seeds.')
    parser.add_argument('--method', help='Method to select the checkpoint ("best" or "last" or "<specific index>").', default='last')
    
    args = parser.parse_args()
    main(args.file_paths, args.method)
