import pandas as pd
import numpy as np

def DateChg(dfs,dfs_tl,ctm):
    '''
    The main contract price determines the first contract for trading.
    Calculate the average rollover time for a single commodity—the average holding period for each contract.
    Switch to the next contract n days before expiration.
    '''

    code = dfs[ctm]['trade_hiscode'] 
    code_chg = dfs[ctm][code!=code.shift(-1)]
    diff = pd.to_datetime(code_chg['ltdate_new'])-pd.to_datetime(code_chg.index)
    day = diff[:-1].mean().days 

    ctm_name = ctm.split('.')[0]
    dfs_main = {key: value for key, value in dfs_tl.items() if ctm_name in key}
    code_start = code[0]
    ct_now = code_start.split('.')[0][:2]+code_start.split('.')[0][-2:]+'M.'+code_start.split('.')[1] # 初始合约
    dfs_key = list(dfs_main.keys())
    ct_pos = dfs_key.index(ct_now)
    price = pd.DataFrame() 
    end_date = dfs_main[ct_now].index[0]
    begin_date = end_date
    begin_date = pd.to_datetime(begin_date).strftime('%Y-%m-%d')
    
    while end_date < dfs[ctm].index[-1]:
        print(ct_now)
        print(begin_date)
        pri_now = dfs_main[ct_now] 
        pri_now.index = pd.to_datetime(pri_now.index)
        if pri_now.loc[begin_date,'ltdate_new'] <=dfs[ctm].index[-1]:
            end_date = np.busday_offset(np.datetime64(pri_now.loc[begin_date,'ltdate_new'],'D'), -day, roll='forward')
            while len(np.where(dfs[ctm].index == end_date)[0]) == 0:
                end_date = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            if end_date > dfs[ctm].index[-1]:
                end_date = np.datetime64(dfs[ctm].index[-1],'D')
            else:
                end_pos = np.where(pri_now.index == end_date)[0][0]
            #end_date = dfs[ctm].index[end_pos]
        else: 
            end_date = np.datetime64(dfs[ctm].index[-1],'D')
        print(end_date)
        price = pd.concat([price,pri_now[begin_date:end_date]])
        if end_date != dfs[ctm].index[-1]:
            #begin_date = np.busday_offset(end_date, 1, roll='forward')
            begin_date = pri_now.index[end_pos+1]
            ct_pos = (ct_pos + 1) % len(dfs_key) 
            ct_now =  dfs_key[ct_pos]
    return day,price


def ADF_test(new1,new2,name,method='residuals'):
    rlt = pd.DataFrame()
    new1,new2 = DateSame(pd.DataFrame(new1), pd.DataFrame(new2))
    x = new1.iloc[:,0].astype(float)
    y = new2.iloc[:,0].astype(float)
    if method =='ols':
        #x = sm.add_constant(x)  # 添加截距
        model = sm.OLS(y, x).fit()
        residuals = model.resid
        adf_test = adfuller(residuals)
        coefficients = model.params
    elif method=='residuals':
        residuals = x-y
        # ADF检验
        adf_test = adfuller(residuals)
        coefficients = 0
    rlt.loc['ADF Statistic',name] = adf_test[0]
    rlt.loc['p-value',name] = adf_test[1]
#    print('ADF Statistic:', adf_test[0])
#    print('p-value:', adf_test[1])
    return rlt,coefficients


def DateSame(factor1,factor2,dtype='df'):
 
    f1 = factor1.copy()
    f2 = factor2.copy()
    if dtype == 'df':
        f1 = factor1.loc[factor1.index.isin(factor2.index),:]
        f2 = factor2.loc[factor2.index.isin(factor1.index),:]
    elif dtype == 'sr':
        f1 = factor1[factor2.index]
        f2 = factor2[factor1.index]
    return f1,f2
