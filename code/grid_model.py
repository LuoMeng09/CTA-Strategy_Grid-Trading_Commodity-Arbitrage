'''
Calculation Method for Ratios:
1. Balance according to the value of goods.
2. Balance according to volatility.
Regression Test:
1. ADF Test: To determine whether the series is non-stationary.
In the ADF test, the null hypothesis usually is that the series contains a unit root (i.e., it is non-stationary), and the alternative hypothesis is that the series does not contain a unit root (i.e., it is stationary).
If the p-value of the ADF test is less than the significance level (e.g., 0.05), the null hypothesis is rejected, and the series is considered stationary.
2. Hurst Index: An indicator used to assess the long-term memory of a time series.
The value of the Hurst Index lies between 0 and 1. If the Hurst Index is greater than 0.5, it indicates that the series has a characteristic of persistence or trend reinforcement; if it is equal to 0.5, it indicates that the series is a random walk process; if it is less than 0.5, it indicates that the series has a characteristic of anti-persistence or mean reversion.
The Hurst Index can help determine how the future trend of a series may be influenced by past trends.
'''
import pandas as pd
import numpy as np
from data_processor import *
from data_index import *
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc

def Ratio(data,name1,name2,name,f1,f2,method='residuals',weight=None):
    '''
    Weight: Methods of Balancing

    Value-Based Balancing
    Volatility Balancing: If the volatility of one asset is significantly higher than another, it can lead to excessive volatility in the ratio. To reduce this volatility, the weights of the assets in the ratio can be adjusted to balance their volatilities. f1 and f2 are multiplier factors. Calculation Methods: 1. Value, 2. Volatility, 3. OLS Balancing
    '''
    p1 = pd.DataFrame(data[name1]['open'].dropna()) 
    p1.index = pd.to_datetime(p1.index)
    p2 = pd.DataFrame(data[name2]['open'].dropna())
    p2.index = pd.to_datetime(p2.index)
    p1,p2 = DateSame(p1,p2)
    p1 = f1*p1
    p2 = f2*p2
    
    a = 1
    b = (p1.mean() / p2.mean())[0]
    new1 = a*p1 
    new2 = b*p2
    
    # correlation = new1.iloc[:,0].corr(new2.iloc[:,0])
    vol_new1_ori = new1.std()
    vol_new2_ori = new2.std()
    ratio_ori = new1/new2
    adf,coef = ADF_test(new1,new2,name,method)
    if adf.loc['p-value',name] > 0.05:

        std1 = np.std(p1)
        std2 = np.std(p2)
        # counting weight 
        weight1 = 1 / std1
        weight2 = 1 / std2
        # standardize
        total_weight = weight1 + weight2
        a /= total_weight
        b /= total_weight 
        new1 = a * p1
        new2 = b * p2
        new1 = new1.replace(0,np.nan).dropna()
        new2 = new2.replace(0,np.nan).dropna()
        
        adf,coef = ADF_test(new1,new2,name,method)
        if adf.loc['p-value',name] > 0.05:
            print(name+'ratio does not pass the cointegration test; an OLS test is conducted.')
            adf,coef = ADF_test(p1,p2,name,method='ols')
            a = coef['open']
            b = 1
            new1 = a*p1
            new2 = b*p2
            print ('ols_adf',adf)
            if adf.loc['p-value',name] > 0.05:
            
                print(name+'The ratio does not pass the cointegration test in any of the three methods; differencing is performed.')
                p1 = p1.diff().replace(0,np.nan).dropna()
                p2 = p2.diff().replace(0,np.nan).dropna()
        
                b = (p1.mean() / p2.mean())[0]
                a = 1
                new1 = a*p1 
                new2 = b*p2 
                method = 'residuals'
                adf,coef = ADF_test(new1,new2,name,method)

                if adf.loc['p-value',name] > 0.05:
      
                    std1 = np.std(p1)
                    std2 = np.std(p2)
                
                    weight1 = 1 / std1
                    weight2 = 1 / std2
         
                    total_weight = weight1 + weight2
                    a /= total_weight
                    b /= total_weight
                    new1 = a * p1
                    new2 = b * p2
                    new1 = new1.replace(0,np.nan).dropna()
                    new2 = new2.replace(0,np.nan).dropna()
                    adf,coef = ADF_test(new1,new2,name,method)

                    if adf.loc['p-value',name] > 0.05:
                        adf,coef = ADF_test(p1,p2,name,method='ols')
                        a = coef['open']
                        b = 1
                        new1 = a*p1
                        new2 = b*p2
                        adf,coef = ADF_test(new1,new2,name,method)
                        print('diff_ols_adf',adf)
                        method = 'residuals'
            
    new_ratio = new1/new2
    # Hurst
    new_ratio = new_ratio.replace(np.inf,np.nan)
    new_ratio = new_ratio.replace(-np.inf,np.nan)
    new_ratio = new_ratio.dropna()
    
    if np.any(new_ratio< 0):
        hurst = None
    else:
        hurst, c, df_tp = compute_Hc(new_ratio, kind='price', simplified=True)
    adf ,coef= ADF_test(new1,new2,name,method)

    vol_new1 = new1.std()
    vol_new2 = new2.std()
    return ratio_ori,new_ratio,hurst,adf,a,b,vol_new1_ori,vol_new2_ori,vol_new1,vol_new2


def sequence_ratio_transformation(x_values, y_values):
    """
    Calculate the ratio of each pair of values in the x and y sequences,
    and also the first order difference ratios.

    :param x_values: List of x values.
    :param y_values: List of y values.
    :return: Tuple of two lists - (ratios, difference_ratios).
    """
    ratios = [x / y for x, y in zip(x_values, y_values)]
    difference_ratios = [(x_values[i] - x_values[i-1]) / (y_values[i] - y_values[i-1]) for i in range(1, len(x_values))]

    return ratios, difference_ratios

def IniticalMulti(ratio):
    # Observe the Data Distribution
    mean = np.mean(ratio)[0]
    std = np.std(ratio)

    low_sig1 = (mean - 1 * std)[0]
    up_sig1 = (mean + 1 * std)[0]
    low_sig2 = (mean - 2 * std)[0]
    up_sig2 = (mean + 2 * std)[0]

    low_sig3 = (mean - 3 * std)[0]
    up_sig3 = (mean + 3 * std)[0]
    low_sig4 = (mean - 4 * std)[0]
    up_sig4 = (mean + 4 * std)[0]
    ratio_adj = ratio.copy()
    ratio_adj[ratio > up_sig3] = up_sig3
    ratio_adj[ratio < low_sig3] = low_sig3
    ratio_fin = ratio.copy()
    ratio_fin[ratio > up_sig4] = up_sig4
    ratio_fin[ratio < low_sig4] = low_sig4
    step = np.mean(abs(ratio_adj.diff()).dropna())[0] # Use the mean of the ratio's changes to avoid excessive intraday positions and low capital utilization.
    skew = ratio.skew()[0]
    kurt = ratio.kurtosis()[0]
    
    diff_up = abs(up_sig3 - up_sig1)
    diff_low = abs(low_sig3 - low_sig1)
    multi = max(diff_up,diff_low)//step

    return multi,mean,low_sig1,up_sig1,low_sig2,up_sig2,low_sig3,up_sig3,low_sig4,up_sig4,step,skew,kurt,ratio_fin


'''
Strategy Concept:
Initial Capital: The maximum threshold range for the gold-silver ratio.
Gold: 1000 grams per hand, Silver: 15 kilograms per hand.
Strategy: If the gold-silver ratio falls below the threshold b, go long, and trade one contract when it drops by step size; if it rises above the threshold a, go short on the gold-silver ratio, and trade one contract when it rises by step size, do not operate within the 80-90 range.
Grid Trading Method:
Signal Generation: When the gold-silver ratio falls below the threshold, it generates a long position; when it rises above the threshold a, it goes short on the gold-silver ratio, generating a short position, and the signal is 0 in the intermediate state.
'''

def Threhold(threshold_a,threshold_b,ratio,step,out=None):
    '''
    Threshold/Signal Generation:
    Open positions based on the percentile thresholds of the current data.
    a: Represents the upper boundary of the regression.
    b: Represents the lower boundary of the regression.
    step: Represents the unit of change in the ratio.
    '''
    signal = ratio.copy() 
    signal = signal.applymap(lambda x: 0) 
    signal.columns = ['signal']
    lots = []

    trigger_a = None
    trigger_b = None
    
    for i,value in enumerate(ratio.iloc[:,0]):
        print(i,value)
        if value>threshold_a: 
            if out == 'return':
                sgl1 = -1 
            else:
                sgl1 = 1
            signal.iloc[i,0] = sgl1 
            if trigger_a is None: 
                lot = (ratio.iloc[i,0]-threshold_a)//step 
                if lot == 0.0:
                    lot = 1.0
                lots.append(lot)
                trigger_a = ratio.iloc[i,0]        
            elif (trigger_a is not None) and (signal.iloc[i-1,0] == sgl1) and (ratio.iloc[i,0] >= (trigger_a+step)):
                 lot = (ratio.iloc[i,0]-trigger_a)//step
                 lots.append(lot)      
                 trigger_a = ratio.iloc[i,0]
            elif (trigger_a is not None) and (signal.iloc[i-1,0] != sgl1) :
                 lot = (ratio.iloc[i,0]-threshold_a)//step
                 if lot == 0.0:
                     lot = 1.0
                 lots.append(lot)
                 trigger_a = ratio.iloc[i,0]
            else:
                lots.append(0.0)
                
        elif value<threshold_b:
            if value<=threshold_b:
                if out =='return':
                    sgl2 = 1
                else:
                    sgl2 = -1
                signal.iloc[i,0] = sgl2 
                if trigger_b is None:
                    lot = (threshold_b-ratio.iloc[i,0])//step
                    if lot == 0.0:
                        lot = 1.0
                    lots.append(lot)
                    trigger_b = ratio.iloc[i,0]
                elif (trigger_a is not None) and (signal.iloc[i-1,0] == sgl2) and ratio.iloc[i,0] <= (trigger_b-step):
                     lot = (trigger_b-ratio.iloc[i,0])//step
                     lots.append(lot)
                     trigger_b = ratio.iloc[i,0]
                elif (trigger_b is not None) and (signal.iloc[i-1,0] != sgl2):
                     lot = (threshold_b-ratio.iloc[i,0])//step
                     if lot == 0.0:
                         lot = 1.0
                     lots.append(lot)
                     trigger_b = ratio.iloc[i,0]
                else:
                    lots.append(0.0)
        else:
            signal.iloc[i,0] = 0
            lots.append(0.0)
            trigger_a = None
            trigger_b = None
    lots = pd.DataFrame(lots,columns=['lots'],index=signal.index)
    signal = pd.concat([ratio,signal,lots],axis=1)
    signal.columns = ['ratio','signal','lots']
    return signal     


def RatioThed(ratio,price,step,mean,low,up,name,date,out=None,st=0.1):
    dtn,thre_a,thre_b,winr=[],[],[],[]
    ratio = pd.DataFrame(ratio)
    if st == 1:
        mean = int(mean)
        up = int(up)
        low = int(low)
        step = int(step)
        for a in range(mean,up):
            for b in range(low,mean):
                print(a,b)
                signal = Threhold(a,b,ratio,step,out)
                signal.index = pd.to_datetime(signal.index)
                price.index = pd.to_datetime(price.index)
        
                signal = signal[~signal.index.duplicated(keep='first')]
                signal = pd.concat([signal,price],axis=1)
                signal.dropna(inplace=True)
                temp_sgl = signal['signal']
                temp_lot = signal['lots']
                
                mark_signal = price['open'].pct_change().dropna().apply(lambda x: 1 if x>0 else(-1 if x<0 else 0)) # 三分类价格数据
                x1 = temp_sgl  # 测试信号
                x2 = mark_signal  # 实际信号
                winratio = WinRatio(x1,x2)
                
                distance = calculate_distance_to_close(temp_sgl,temp_lot)
                print(distance)
                dtn.append(distance)
                thre_a.append(a)
                thre_b.append(b)
                winr.append(winratio)
    else:
        for a in np.arange(mean,up,st):
            for b in np.arange(low,mean,st):
                print(a,b)
                signal = Threhold(a,b,ratio,step,out)
                signal.index = pd.to_datetime(signal.index)
                price.index = pd.to_datetime(price.index)
        
                signal = signal[~signal.index.duplicated(keep='first')]
                signal = pd.concat([signal,price],axis=1)
                signal.dropna(inplace=True)
                temp_sgl = signal['signal']
                temp_lot = signal['lots']
                
                mark_signal = price['open'].pct_change().dropna().apply(lambda x: 1 if x>0 else(-1 if x<0 else 0))
                x1 = temp_sgl  
                x2 = mark_signal  
                winratio = WinRatio(x1,x2)
                
                distance = calculate_distance_to_close(temp_sgl,temp_lot)
                print(distance)
                dtn.append(distance)
                thre_a.append(a)
                thre_b.append(b)
                winr.append(winratio)
    
    count = pd.DataFrame({'upper_thres':thre_a,'lower_thres':thre_b,'Liquidation Time':dtn,'winratio':winr})
    count = count.replace(np.inf,np.nan)
    count.dropna(inplace=True)
    count = count.sort_values(by='Liquidation Time').reset_index()

    return count


def Select(count,up=0,low=0,time=False):

    filtered_df = count.copy()
    if time == True:
        filtered_df = count[count['Liquidation Time'] > 1]
    if up != 0:
        filtered_df = filtered_df[filtered_df['upper_thres'] < up]
    if low != 0:
        filtered_df = filtered_df[filtered_df['lower_thres'] > low]
    

    # Sort these records in descending order based on the win rate
    sorted_filtered_df = filtered_df.sort_values(by='winratio', ascending=False)

    # elect the top 10 records with the highest win rate
    top_10 = sorted_filtered_df.head(10)

    # among these 10 records, identify the value with the shortest liquidation time.
    if top_10.size==0:
        count_sel = count.sort_values(by='winratio', ascending=False).iloc[0,:]
    else:
        count_sel = top_10.sort_values(by='Liquidation Time').reset_index().iloc[0,:]
    return count_sel