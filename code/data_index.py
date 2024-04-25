#%% The speed of profit realization time; the denominator is divided by the number of hands 
import pandas as pd
import numpy as np

def calculate_distance_to_close(signals,lots):
    start_idx = 0
    end_idx = 0
    start_lot = []
    distances = []
    open_position = False

    for i, signal in enumerate(signals):
        print(i)
        if signal != 0 and not open_position:  # First open position
            open_position = True
            start_idx = i
            start_lot.append(lots[i])
            print(start_idx,start_lot)
        elif open_position and signals[i] != signals[i-1]:  # Closing the position
            end_idx = i
            span_idx = end_idx - start_idx 
            print(span_idx)
            distances.append(span_idx/sum(start_lot))
            open_position = False
            start_idx = 0
            start_lot = []
        elif open_position and signals[i] == signals[i-1]:  # Increasing the position
            start_lot.append(lots[i])
            print(start_idx,start_lot)
    distances = np.mean(distances)
    return distances

#%%
def WinRatio(x1,x2,ic=None):
    '''
    Win Rate Calculation - The number of correct predictions / total number of open positions
    Parameters:
    x1: Prediction signal
    x2: Actual signal
    Returns:
    Win rate
    '''
    sumlen = sum(x1!=0)
    if sumlen == 0:
        count = np.nan
    else :
        count = (sum((x1==1)&(x2==1)) + sum((x1==-1)&(x2==-1)))  / sumlen
    return count