
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from typing import List, Dict
from linear_diagnostics import LinearRegDiagnostic

class DataModel():
    '''Wrapper for `statsmodels.formula.api` with additional methods for assumption checking.'''

    def __init__(self, data: pd.DataFrame, other_event: int, model_type: str) -> None:
        '''Parameters:
          -  data (`pd.DataFrame`): a DataFrame with 800m and [other_event]m data
          -  other_event (int): the other event being analyzed. Options: 400 or 1500
          -  model_type: the type of model to call from `statsmodels.formula.api`. Currently supports 'ols' and 'rlm'
        '''
        self.data = data
        self.other_event = other_event
        self.model_type = model_type
        
    @property
    def model(self):
        '''Returns a fitted `statsmodels.formula.api` model based on the specified `model_type` in the initializer. Compatible with other `statsmodels.formula.api` attributes and methods'''
        assert self.model_type in smf.__all__, 'Choose a valid `statsmodels.formula.api` model'

        match self.model_type:
            case 'ols':
                return smf.ols(f'time_800 ~ time_{self.other_event}', data=self.data).fit()
            case 'rlm':
                return smf.rlm(f'time_800 ~ time_{self.other_event}', data=self.data).fit() 

    @property
    def model_summary(self) -> statsmodels.iolib.summary.Summary:
        '''Returns a summary of a `statsmodels.formula.api` model based on the specified `model_type` in the initializer'''
        return self.model.summary()
    

    def check_assumptions(self) -> None:
        '''Check the assumptions of linear regression. That is: linearity, normally-distributed residuals, constant variance, and the model describes all observations.'''

        if self.model_type == 'ols':
            model_diagnostic = LinearRegDiagnostic(self.model)
            model_diagnostic(context='seaborn-v0_8-whitegrid', high_leverage_threshold=True)
    

    def plot_dist(self, data: pd.DataFrame | None = None, other_event: int | None = None) -> None:
        '''Plot a histogram with overlying KDE for 400m and 800m times in a pd.DataFrame.
        
        Parameters:
          -  data (`pd.DataFrame`): a pd.DataFrame with 800m and 'other_event' times
          -  other_event (int): the event besides the 800m that's in the dataset'''
        
        sns.set_theme(style='whitegrid')

        if data is None:
            data = self.data

        if other_event is None:
            other_event = self.other_event

        # Sturge's Rule
        BINS = int(np.ceil(np.log2(len(data)) + 1))

        plt.figure(figsize = (15, 5))
        plt.suptitle(f'{other_event}m vs 800m\nn={len(data):,}')

        plt.subplot(1, 3, 1)
        plt.scatter(x=data[f'time_{other_event}'], y=data['time_800'], alpha=0.5)
        plt.xlabel(f'{other_event}m')
        plt.ylabel('800m')

        plt.subplot(1, 3, 2)
        sns.histplot(data['time_800'], bins=BINS, kde=True)

        plt.subplot(1, 3, 3)
        sns.histplot(data[f'time_{other_event}'], bins=BINS, kde=True)  


    def plot_conf_int(self, data: pd.DataFrame | None = None, other_event: int | None = None) -> None:
        '''Plot a regression plot with 95% Confidence Intervals for an ols model
        
        Parameters:
          -  data (`pd.DataFrame`): a pd.DataFrame with 800m and 'other_event' times
          -  other_event (int): the event besides the 800m that's in the dataset'''
        
        if self.model_type not in ['ols', 'rlm']:
            raise ValueError('This function can only be run on OLS models')
        
        sns.set_theme(style='whitegrid')

        if data is None:
            data = self.data

        if other_event is None:
            other_event = self.other_event

        match self.model_type:
            case 'ols':
                plt.figure(figsize=(5, 5))
                sns.regplot(data, x=f'time_{other_event}', y='time_800',
                            line_kws={'color': 'red'})
            case 'rlm':
                plt.figure(figsize=(5, 5))
                sns.regplot(data, x=f'time_{other_event}', y='time_800', robust=True,
                            line_kws={'color': 'red'})
        

def plot_eda(data: pd.DataFrame, other_event: int, **kwargs) -> None:
    '''Returns a scatter plot and two boxplots of the data in the `pd.DataFrame`
    
    Parameters:
      -  data (pd.DataFrame): a `pd.DataFrame` of 800m and `other_event` data
      -  other_event (int): the length of the other event in the data frame in meters'''
    
    sns.set_theme(style = 'whitegrid')

    kwargs.setdefault('color', 'lightblue')
    kwargs.setdefault('linecolor', 'black')
    kwargs.setdefault('thresh', 0.01)
    
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(x=data[f'time_{other_event}'], 
                y=data['time_800'], 
                alpha = 0.3)

    plt.subplot(2, 2, 2)
    sns.kdeplot(x=data[f'time_{other_event}'], 
                y=data['time_800'], 
                thresh=kwargs.get('thresh'),
                cmap='cividis')

    plt.subplot(2, 2, 3)
    sns.boxplot(y=data['time_800'], 
                width=0.3, 
                color=kwargs.get('color'), 
                linecolor=kwargs.get('linecolor'))
    plt.title('800m')

    plt.subplot(2, 2, 4)
    sns.boxplot(y=data[f'time_{other_event}'], 
                width=0.3, 
                color=kwargs.get('color'), 
                linecolor=kwargs.get('linecolor'))
    plt.title(f'{other_event}m')
