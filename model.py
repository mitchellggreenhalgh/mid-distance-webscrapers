
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

class BivariateModel():
    '''Wrapper for `statsmodels.formula.api` with additional methods for assumption checking.'''

    def __init__(self, data: pd.DataFrame, other_event: int, model_type: str) -> None:
        '''Parameters:
          -  data (`pd.DataFrame`): a DataFrame with 800m and [other_event]m data
          -  other_event (int): the other event being analyzed. Options: 400 or 1500
          -  model_type (str): the type of model to call from `statsmodels.formula.api`. Currently supports 'ols' and 'rlm'
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
                sns.regplot(data, x=f'time_{other_event}', y='time_800', 
                            robust=True,
                            line_kws={'color': 'red'})
                
    
    def predict_time(self,  time: str, event: int | str | None = None) -> float:
        '''Use the model's parameters to predict the average 800m time for a runner who runs a certain event in a certain time. No protection against extrapolation
        
        Parameters:
          -  event (int | str): possible events to predict 800m time from. Options: '400', '1600', 'mile'
          -  time (str): time elapsed in the specified event. Format: 'm:ss.xx' 
        
        Returns:
          -  estimate (float): the estimated time according to the parameters'''
        
        if event is None:
            event = self.other_event

        # Grab coefficients
        beta_0 = self.model.params['Intercept']

        match str(event).lower():
            case '400':
                beta_1_index = 'time_400'
            case '1500' | '1600' | 'mile':
                beta_1_index = 'time_1500'

        beta_1 = self.model.params[beta_1_index]

        # Convert time to seconds
        time_sec = float(time.split(':')[0]) * 60 + float(time.split(':')[1])

        # Add 1600m and Mile conversions to 1500m
        if str(event) == '1600':
            time_sec = time_sec * 0.9375
        elif str(event) == 'mile':
            time_sec = time_sec * 0.93205678835

        return round(beta_0 + beta_1 * time_sec, 2)
    

class MultivariateModel():
    '''Wrapper for `statsmodels.formula.api` with additional methods for assumption checking.'''

    def __init__(self, data: pd.DataFrame, model_type: str, outcome_event: int = 800, other_events: List[int] = [400, 1500]) -> None:
        '''Parameters:
          -  data (`pd.DataFrame`): a DataFrame with 800m and [other_event]m data
          -  outcome_event (int): the outcome variable in the analysis. Select one element from the following list: [400, 800, 1500]
          -  other_events (int): the other events being analyzed. Any two event combination of [400, 800, 1500] where the outcome event is removed from the list
          -  model_type (str): the type of model to call from `statsmodels.formula.api`. Currently supports 'ols' and 'rlm'
        '''
        self.data = data
        self.model_type = model_type
        self.outcome_event = outcome_event
        self.other_events = other_events
        
    @property
    def model(self):
        '''Returns a fitted `statsmodels.formula.api` model based on the specified `model_type` in the initializer. Compatible with other `statsmodels.formula.api` attributes and methods. Currrently only includes OLS and RLM without interactions.'''
        assert self.model_type in smf.__all__, 'Choose a valid `statsmodels.formula.api` model'

        predictor_formula = ' + '.join(['time_' + str(event) for event in self.other_events])

        match self.model_type:
            case 'ols':
                return smf.ols(f'time_{self.outcome_event} ~ {predictor_formula}', data=self.data).fit()
            case 'rlm':
                return smf.rlm(f'time_{self.outcome_event} ~ {predictor_formula}', data=self.data).fit() 

    @property
    def model_summary(self) -> statsmodels.iolib.summary.Summary:
        '''Returns a summary of a `statsmodels.formula.api` model based on the specified `model_type` in the initializer'''
        return self.model.summary()
    

    def check_assumptions(self) -> None:
        '''Check the assumptions of linear regression. That is: linearity, normally-distributed residuals, constant variance, and the model describes all observations.'''

        if self.model_type == 'ols':
            model_diagnostic = LinearRegDiagnostic(self.model)
            
            return model_diagnostic(context='seaborn-v0_8-whitegrid', high_leverage_threshold=True, vif=True)
    

    def plot_partial_regressors(self) -> None:
        '''Plot partial regression plots for the model'''
        
        sns.set_theme(style='whitegrid')

        fig = plt.figure(figsize=(10,10))
        sm.graphics.plot_partregress_grid(self.model, grid=(2,2), fig=fig)

    
    def plot_dist(self, **kwargs) -> None:
        '''Plots the distributions of the outcome event and the predictor events.'''
        
        sns.set_theme(style='whitegrid')

        kwargs.setdefault('color', 'lightblue')
        kwargs.setdefault('linecolor', 'black')
        kwargs.setdefault('width', 0.3)

        # Sturge's Rule for histograms
        BINS: int = int(np.ceil(np.log2(len(self.data)) + 1))

        NROWS: int = 1 + len(self.other_events)
        NCOLS: int = 2 


        plt.figure(figsize = (10, 5 * NROWS))
        plt.suptitle(f'Distributions of Outcome and Predictors')

        # Outcome Distribution
        plt.subplot(NROWS, NCOLS, 1)
        sns.histplot(self.data[f'time_{self.outcome_event}'], 
                     bins=BINS, 
                     kde=True)

        plt.subplot(NROWS, NCOLS, 2)
        sns.boxplot(y=self.data[f'time_{self.outcome_event}'], 
                    width=kwargs.get('width'),
                    color=kwargs.get('color'), 
                    linecolor=kwargs.get('linecolor'))

        # Predictor Distributions
        for i in range(len(self.other_events)):
            plt.subplot(NROWS, NCOLS, i*2 + 3)
            sns.histplot(self.data[f'time_{self.other_events[i]}'], 
                         bins=BINS, 
                         kde=True)
            
            plt.subplot(NROWS, NCOLS, i*2 + 4)
            sns.boxplot(y=self.data[f'time_{self.other_events[i]}'], 
                    width=kwargs.get('width'),
                    color=kwargs.get('color'), 
                    linecolor=kwargs.get('linecolor'))

    # TODO: #1 Add influence plots, fit plot, ccpr
    # https://www.statsmodels.org/devel/examples/notebooks/generated/regression_plots.html#Using-robust-regression-to-correct-for-outliers.
         
        
    def predict_time(self, times: List[str], events: List[int] | List[str] = None) -> float:
        '''Use the model's parameters to predict the average 800m time for a runner who runs a certain event in a certain time. No protection against extrapolation
        
        Parameters:
          -  times (List[str]): time elapsed in the events specified in the events argument. The times must be in the same order as the events they correspond to in the events argument. The time format must follow: 'm:ss.xx' 
          -  events (List[int] | List[str]): list of 2 events to predict the outcome event time. Options: '400', '800', '1600', 'mile'
        
        Returns:
          -  estimate (float): the estimated time according to the parameters'''
        
        if events is None:
            events = self.other_events

        # Grab coefficients
        beta_0 = self.model.params['Intercept']

        match str(events[0]).lower():
            case '400':
                beta_1_index = 'time_400'
            case '1500' | '1600' | 'mile':
                beta_1_index = 'time_1500'

        beta_1 = self.model.params[beta_1_index]

        match str(events[1]).lower():
            case '400':
                beta_2_index = 'time_400'
            case '1500' | '1600' | 'mile':
                beta_2_index = 'time_1500'

        beta_2 = self.model.params[beta_2_index]

        # Convert time to seconds
        time_1_sec = float(times[0].split(':')[0]) * 60 + float(times[0].split(':')[1])
        time_2_sec = float(times[1].split(':')[0]) * 60 + float(times[1].split(':')[1])

        # Add 1600m and Mile conversions to 1500m
        if str(events[0]) == '1600':
            time_1_sec = time_1_sec * 0.9375
        elif str(events[0]) == 'mile':
            time_1_sec = time_1_sec * 0.93205678835

        if str(events[1]) == '1600':
            time_2_sec = time_2_sec * 0.9375
        elif str(events[1]) == 'mile':
            time_2_sec = time_2_sec * 0.93205678835

        return f'{self.outcome_event}m Prediction: {round(beta_0 + beta_1 * time_1_sec + beta_2 * time_2_sec, 2)} seconds'

        

def plot_bivariate_eda(data: pd.DataFrame, other_event: int, **kwargs) -> None:
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
    plt.xlabel(f'time_{other_event}')
    plt.ylabel('time_800')

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
