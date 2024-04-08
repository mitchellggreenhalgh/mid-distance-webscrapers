
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
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tools.tools import add_constant


class BivariateModel():
    '''Wrapper for `statsmodels.api` with additional methods for assumption checking.'''

    def __init__(self, 
                 data: pd.DataFrame,
                 *, 
                 outcome_event: int, 
                 predictor_event: int, 
                 model_type: str) -> None:
        '''Initializer for the BivariateModel class.
        
        Parameters:
          -  data (`pd.DataFrame`): a DataFrame with outcome and predictor data
          -  outcome_event (`int`): the outcome event to be analyzed. Options: 400, 800, or 1500
          -  predictor_event (`int`): the predictor event being analyzed. Options: 400, 800, or 1500, exclusive of the outcome event
          -  model_type (`str`): the type of model to call from `statsmodels.formula.api`. Currently supports 'ols', 'quad', 'rlm', and 'quantreg'
        '''

        self.data = data
        self.outcome_event = outcome_event
        self.predictor_event = predictor_event
        self.model_type = model_type
        
    @property
    def model(self):
        '''Returns a fitted `statsmodels.api` model based on the specified `model_type` in the initializer. Compatible with other `statsmodels.api` attributes and methods.'''

        assert self.model_type in smf.__all__ + ['quad'], 'Choose a valid `statsmodels.api` model'

        y = self.data[f'time_{self.outcome_event}']
        X = self.data[[f'time_{self.predictor_event}']]

        match self.model_type:
            case 'ols':
                return sm.OLS(y, add_constant(X)).fit()
            case 'quad':
                X_quad = PolynomialFeatures(degree=2).fit_transform(X)
                return sm.OLS(y, X_quad).fit()            
            case 'rlm':
                return sm.RLM(y, add_constant(X)).fit()
            case 'quantreg':
                model = smf.quantreg(f'time_{self.outcome_event} ~ time_{self.predictor_event}', data=self.data)
                quantiles = np.arange(0.05, 0.96, 0.1)

                model_list = [self.fit_quantile(q=i, model=model) for i in quantiles]
                model_df = pd.DataFrame(
                    model_list, 
                    columns=[
                        'q', 
                        'intercept', 
                        f'time_{self.predictor_event}', 
                        f'time_{self.predictor_event}_ll', 
                        f'time_{self.predictor_event}_ul'
                        ]
                    )
                
                ols = smf.ols(f'time_{self.outcome_event} ~ time_{self.predictor_event}', data=self.data).fit()
                ols_ci_b = ols.conf_int().loc[f'time_{self.predictor_event}'].tolist()
                ols = dict(
                    a=ols.params['Intercept'], 
                    b=ols.params[f'time_{self.predictor_event}'], 
                    bll=ols_ci_b[0], 
                    bul=ols_ci_b[1], 
                )

                return model_df, ols
            

    @property
    def model_summary(self) -> statsmodels.iolib.summary.Summary:
        '''Returns a summary of a `statsmodels.formula.api` model based on the specified `model_type` in the initializer'''
        if self.model_type == 'quantreg':
            return self.model[0]
        else:
            return self.model.summary()
    

    def check_assumptions(self) -> None:
        '''Check the assumptions of linear regression. That is: linearity, normally-distributed residuals, constant variance, and the model describes all observations.'''

        if self.model_type in ['ols', 'quad']:
            model_diagnostic = LinearRegDiagnostic(self.model)
            model_diagnostic(context='seaborn-v0_8-whitegrid', high_leverage_threshold=True)
    

    def plot_dist(self, 
                  data: pd.DataFrame | None = None, 
                  outcome_event: int | None = None, 
                  predictor_event: int | None = None) -> None:
        '''Plot a histogram with overlying KDE for 400m and 800m times in a pd.DataFrame.
        
        Parameters:
          -  data (`pd.DataFrame`): a pd.DataFrame with 800m and 'predictor_event' times
          -  outcome_event (`int` | `None`): the outcome event being modelled
          -  predictor_event (`int` | `None`): the predictor event in the dataset
        '''
        
        sns.set_theme(style='whitegrid')

        if data is None:
            data = self.data

        if predictor_event is None:
            predictor_event = self.predictor_event

        if outcome_event is None:
            outcome_event = self.outcome_event

        # Sturge's Rule
        BINS = int(np.ceil(np.log2(len(data)) + 1))

        plt.figure(figsize = (15, 5))
        plt.suptitle(f'{predictor_event}m vs {outcome_event}m\nn={len(data):,}')

        plt.subplot(1, 3, 1)
        plt.scatter(x=data[f'time_{predictor_event}'], y=data[f'time_{outcome_event}'], alpha=0.5)
        plt.xlabel(f'{predictor_event}m')
        plt.ylabel(f'{outcome_event}m')

        plt.subplot(1, 3, 2)
        sns.histplot(data[f'time_{outcome_event}'], bins=BINS, kde=True)

        plt.subplot(1, 3, 3)
        sns.histplot(data[f'time_{predictor_event}'], bins=BINS, kde=True)  


    def plot_conf_int(self, 
                      data: pd.DataFrame | None = None, 
                      outcome_event: int | None = None, 
                      predictor_event: int | None = None) -> None:
        '''Plot a regression plot with 95% Confidence Intervals for a model
        
        Parameters:
          -  data (`pd.DataFrame`): a pd.DataFrame with [outcome_event]m and [predictor_event]m times
          -  outcome_event (`int` | `None`): the outcome event being modelled
          -  predictor_event (`int` | `None`): the predictor event in the dataset
        '''
        
        sns.set_theme(style='whitegrid')

        if data is None:
            data = self.data

        if outcome_event is None:
            outcome_event = self.outcome_event

        if predictor_event is None:
            predictor_event = self.predictor_event

        match self.model_type:
            case 'ols':
                plt.figure(figsize=(5, 5))
                sns.regplot(data, x=f'time_{predictor_event}', y=f'time_{outcome_event}',
                            line_kws={'color': 'red'},
                            scatter_kws={'alpha': 0.3})
            case 'quad':
                plt.figure(figsize=(5, 5))
                sns.regplot(data, x=f'time_{predictor_event}', y=f'time_{outcome_event}',
                            line_kws={'color': 'red'},
                            scatter_kws={'alpha': 0.3},
                            order=2)
            case 'rlm':
                plt.figure(figsize=(5, 5))
                sns.regplot(data, x=f'time_{predictor_event}', y=f'time_{outcome_event}', 
                            robust=True,
                            line_kws={'color': 'red'},
                            scatter_kws={'alpha': 0.3})
                            
    
    def fit_quantile(self, q: float, model: statsmodels.regression.quantile_regression.QuantReg) -> List[List[float]]:
        '''Fit a linear model for a given quantile.
        
        Parameters:
          -  q (float): the quantile to regress on
          -  model (statsmodels.regression.quantile_regression.QuantReg): The unfit instantiation of a Quantile Regression model
        '''
        
        results = model.fit(q=q)

        return [q, results.params['Intercept'], results.params[f'time_{self.predictor_event}']] + \
            results.conf_int().loc[f'time_{self.predictor_event}'].tolist()
    

    def plot_quantiles_by_parameter(self, quantile_data: pd.DataFrame | None = None, ols_data: dict | None = None) -> None:
        '''Docstring'''

        sns.set_theme(style='whitegrid')

        if quantile_data is None:
            quantile_data = self.model[0]

        if ols_data is None:
            ols_data = self.model[1]

        n = quantile_data.shape[0]
        plt.figure(figsize=(5,5))  
        plt.title(f'Conditional Parameter Estimates across {self.outcome_event}m Quantiles')

        p1 = plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_event}'], color='black', label=f'Quantile Reg {self.predictor_event}m')
        p2 = plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_event}_ul'], linestyle='dotted', color='black')
        p3 = plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_event}_ll'], linestyle='dotted', color='black')
        p4 = plt.plot(quantile_data['q'], [ols_data['b']] * n, color='red', label=f'OLS {self.predictor_event}m')
        p5 = plt.plot(quantile_data['q'], [ols_data['bll']] * n, linestyle='dotted', color='red')
        p6 = plt.plot(quantile_data['q'], [ols_data['bul']] * n, linestyle='dotted', color='red')
        plt.ylabel(fr'$\beta_{{time_{{{self.predictor_event}}}}}$')
        plt.xlabel(f'Quantiles of the conditional {self.outcome_event}m distribution')
        plt.legend()
    

    def predict_time(self,  
                     time: str, 
                     event: int | str | None = None) -> float:
        '''Use the model's parameters to predict the average 800m time for a runner who runs a certain event in a certain time. No protection against extrapolation
        
        Parameters:
          -  event (`int` | `str`): possible events to predict 800m time from. Options: '400', '1600', 'mile'
          -  time (`str`): time elapsed in the specified event. Format: 'm:ss.xx' 
        
        Returns:
          -  estimate (`float`): the estimated time according to the parameters
        '''
        
        if event is None:
            event = self.predictor_event

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
    '''Wrapper for `statsmodels.formula.api` with additional methods for assumption checking, plotting distributions, and using model parameters to predict a time.'''

    def __init__(self, 
                 data: pd.DataFrame, 
                 *,
                 outcome_event: int, 
                 predictor_events: List[int],
                 model_type: str) -> None:
        '''Parameters:
          -  data (`pd.DataFrame`): a DataFrame with [outcome_event]m and [predictor_events]m data
          -  outcome_event (`int`): the outcome variable in the analysis. Select one element from the following list: [400, 800, 1500]
          -  predictor_events (`int`): the predictor events being analyzed. Any two event combination of [400, 800, 1500] exclusive of the outcome event
          -  model_type (`str`): the type of model to call from `statsmodels.formula.api`. Currently supports 'ols', 'rlm', 'quad', 'quantreg'
        '''
        self.data = data
        self.model_type = model_type
        self.outcome_event = outcome_event
        self.predictor_events = predictor_events


    def __call__(self):
        self.plot_dist()
        self.plot_partial_regressors()
        self.check_assumptions(vif=False)
        return self.model_summary
        

    @property
    def model(self):
        '''Returns a fitted `statsmodels.formula.api` model based on the specified `model_type` in the initializer. Compatible with other `statsmodels.formula.api` attributes and methods. Currrently only includes OLS and RLM without interactions.'''

        # TODO: #10 Convert from formula to regular, it should be easier to generalize that way
        assert self.model_type in smf.__all__ + ['quad'], 'Choose a valid `statsmodels.formula.api` model'

        predictor_formula = ' + '.join(['time_' + str(event) for event in self.predictor_events])

        match self.model_type:
            case 'ols':
                return smf.ols(f'time_{self.outcome_event} ~ {predictor_formula}', data=self.data).fit()

            case 'rlm':
                return smf.rlm(f'time_{self.outcome_event} ~ {predictor_formula}', data=self.data).fit() 

            case 'quantreg':
                model = smf.quantreg(f'time_{self.outcome_event} ~ {predictor_formula}', data=self.data)
                # TODO: #4 Add option for different quantiles
                quantiles = np.arange(0.05, 0.96, 0.1)

                model_list = [self.fit_quantile(q=i, model=model) for i in quantiles]
                # TODO: #3 Make columns more generalizable for more than two parameters
                model_df = pd.DataFrame(
                    model_list, 
                    columns=[
                        'q', 
                        'intercept', 
                        f'time_{self.predictor_events[0]}', 
                        f'time_{self.predictor_events[1]}', 
                        f'time_{self.predictor_events[0]}_ll', 
                        f'time_{self.predictor_events[0]}_ul', 
                        f'time_{self.predictor_events[1]}_ll', 
                        f'time_{self.predictor_events[1]}_ul'
                        ]
                    )
                
                ols = smf.ols(f'time_{self.outcome_event} ~ {predictor_formula}', data=self.data).fit()
                # TODO: #5 Make generalizable for more than 2 parameters
                ols_ci_b = ols.conf_int().loc[f'time_{self.predictor_events[0]}'].tolist()
                ols_ci_c = ols.conf_int().loc[f'time_{self.predictor_events[1]}'].tolist()
                ols = dict(
                    a=ols.params['Intercept'], 
                    b=ols.params[f'time_{self.predictor_events[0]}'], 
                    c=ols.params[f'time_{self.predictor_events[1]}'], 
                    bll=ols_ci_b[0], 
                    bul=ols_ci_b[1], 
                    cll=ols_ci_c[0],
                    cul=ols_ci_c[1],
                )

                return model_df, ols
                
            case 'quad':
                y = self.data[f'time_{self.outcome_event}']
                X = self.data[[f'time_{i}' for i in self.predictor_events]]

                X_poly = PolynomialFeatures(degree=2).fit_transform(X)
                X_poly = add_constant(X_poly)

                return sm.OLS(endog=y, exog=X_poly).fit()


    @property
    def model_summary(self) -> statsmodels.iolib.summary.Summary:
        '''Returns a summary of a `statsmodels.formula.api` model based on the specified `model_type` in the initializer'''

        if self.model_type == 'quantreg':
            return self.model[0]
        else:
            return self.model.summary()
    

    def check_assumptions(self, **kwargs) -> None:
        '''Check the assumptions of linear regression. That is: linearity, normally-distributed residuals, constant variance, and the model describes all observations.'''

        kwargs.setdefault('vif', True)

        if self.model_type in ['ols', 'quad']:
            model_diagnostic = LinearRegDiagnostic(self.model)
            
            return model_diagnostic(context='seaborn-v0_8-whitegrid', high_leverage_threshold=True, vif=kwargs.get('vif'))
    

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

        NROWS: int = 1 + len(self.predictor_events)
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
        for i in range(len(self.predictor_events)):
            plt.subplot(NROWS, NCOLS, i*2 + 3)
            sns.histplot(self.data[f'time_{self.predictor_events[i]}'], 
                         bins=BINS, 
                         kde=True)
            
            plt.subplot(NROWS, NCOLS, i*2 + 4)
            sns.boxplot(y=self.data[f'time_{self.predictor_events[i]}'], 
                    width=kwargs.get('width'),
                    color=kwargs.get('color'), 
                    linecolor=kwargs.get('linecolor'))

        
    def fit_quantile(self, 
                     q: float, 
                     model: statsmodels.regression.quantile_regression.QuantReg) -> List[List[float]]:
        '''Fit a linear model for a given quantile.
        
        Parameters:
          -  q (float): the quantile to regress on
          -  model (statsmodels.regression.quantile_regression.QuantReg): The unfit instantiation of a Quantile Regression model'''
        
        results = model.fit(q=q)

        # TODO: #2 Make the return statement more generalizable for more parameters
        return [q, results.params['Intercept'], results.params[f'time_{self.predictor_events[0]}'], results.params[f'time_{self.predictor_events[1]}']] + \
            results.conf_int().loc[f'time_{self.predictor_events[0]}'].tolist() + \
            results.conf_int().loc[f'time_{self.predictor_events[1]}'].tolist()
    

    def plot_quantiles_by_parameter(self, 
                                    quantile_data: pd.DataFrame | None = None, 
                                    ols_data: dict | None = None) -> None:
        # TODO: #6 add docstring
        # https://www.statsmodels.org/dev/examples/notebooks/generated/quantile_regression.html#Second-plot
        # https://www.statsmodels.org/dev/generated/statsmodels.formula.api.quantreg.html#
        '''Docstring'''

        sns.set_theme(style='whitegrid')

        if quantile_data is None:
            quantile_data = self.model[0]

        if ols_data is None:
            ols_data = self.model[1]

        n = quantile_data.shape[0]
        plt.figure(figsize=(12,5))  # TODO: #7 Make figsize, rest of plotting generalizable
        plt.suptitle('Conditional Parameter Estimates across Quantiles')

        plt.subplot(1, 2, 1)
        p1 = plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_events[0]}'], color='black', label=f'Quantile Reg {self.predictor_events[0]}m')
        p2 = plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_events[0]}_ul'], linestyle='dotted', color='black')
        p3 = plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_events[0]}_ll'], linestyle='dotted', color='black')
        p4 = plt.plot(quantile_data['q'], [ols_data['b']] * n, color='red', label=f'OLS {self.predictor_events[0]}m')
        p5 = plt.plot(quantile_data['q'], [ols_data['bll']] * n, linestyle='dotted', color='red')
        p6 = plt.plot(quantile_data['q'], [ols_data['bul']] * n, linestyle='dotted', color='red')
        plt.ylabel(fr'$\beta_{{time_{{{self.predictor_events[0]}}}}}$')
        plt.xlabel(f'Quantiles of the conditional {self.outcome_event}m distribution')
        plt.title(f'{self.predictor_events[0]}m')
        plt.legend()

        plt.subplot(1, 2, 2)
        p7 =  plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_events[1]}'], color='blue', label=f'Quantile Reg {self.predictor_events[1]}m')
        p8 =  plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_events[1]}_ul'], linestyle='dotted', color='blue')
        p9 =  plt.plot(quantile_data['q'], quantile_data[f'time_{self.predictor_events[1]}_ll'], linestyle='dotted', color='blue')
        p10 = plt.plot(quantile_data['q'], [ols_data['c']] * n, color='red', label=f'OLS {self.predictor_events[1]}m')
        p11 = plt.plot(quantile_data['q'], [ols_data['cll']] * n, linestyle='dotted', color='red')
        p12 = plt.plot(quantile_data['q'], [ols_data['cul']] * n, linestyle='dotted', color='red')
        plt.ylabel(fr'$\beta_{{time_{{{self.predictor_events[1]}}}}}$')
        plt.xlabel(f'Quantiles of the conditional {self.outcome_event}m distribution')
        plt.title(f'{self.predictor_events[1]}m')
        plt.legend()

        plt.show()
    
    
    def predict_time(self, 
                     times: List[str], 
                     events: List[int] | List[str] = None) -> float:
        '''Use the model's parameters to predict the average 800m time for a runner who runs a certain event in a certain time. No protection against extrapolation
        
        Parameters:
          -  times (List[str]): time elapsed in the events specified in the events argument. The times must be in the same order as the events they correspond to in the events argument. The time format must follow: 'm:ss.xx' 
          -  events (List[int] | List[str]): list of 2 events to predict the outcome event time. Options: '400', '800', '1600', 'mile'
        
        Returns:
          -  estimate (float): the estimated time according to the parameters'''
        
        if events is None:
            events = self.predictor_events

        # Grab coefficients
        try:
            beta_0 = self.model.params['Intercept']
        except:
            beta_0 = self.model.params['const']

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

        
def plot_bivariate_eda(data: pd.DataFrame, title: str, outcome_event: int, predictor_event: int, **kwargs) -> None:
    '''Returns a scatter plot and two boxplots of the data in the `pd.DataFrame`
    
    Parameters:
      -  data (pd.DataFrame): a `pd.DataFrame` of outcome_event and predictor_event data
      -  title (str): title for the group of plots
      -  outcome_event (int): the distance in meters of the outcome event of interest
      -  predictor_event (int): the distance in meters of the predictor event of interest'''
    
    sns.set_theme(style = 'whitegrid')

    kwargs.setdefault('color', 'lightblue')
    kwargs.setdefault('linecolor', 'black')
    kwargs.setdefault('thresh', 0.01)
    
    plt.figure(figsize=(10, 10))
    plt.suptitle(title)

    plt.subplot(2, 2, 1)
    plt.scatter(x=data[f'time_{predictor_event}'], 
                y=data[f'time_{outcome_event}'], 
                alpha = 0.3)
    plt.xlabel(f'time_{predictor_event}')
    plt.ylabel(f'time_{outcome_event}')

    plt.subplot(2, 2, 2)
    sns.kdeplot(x=data[f'time_{predictor_event}'], 
                y=data[f'time_{outcome_event}'], 
                thresh=kwargs.get('thresh'),
                cmap='cividis')

    plt.subplot(2, 2, 3)
    sns.boxplot(y=data[f'time_{outcome_event}'], 
                width=0.3, 
                color=kwargs.get('color'), 
                linecolor=kwargs.get('linecolor'))
    plt.title(f'{outcome_event}m')

    plt.subplot(2, 2, 4)
    sns.boxplot(y=data[f'time_{predictor_event}'], 
                width=0.3, 
                color=kwargs.get('color'), 
                linecolor=kwargs.get('linecolor'))
    plt.title(f'{predictor_event}m')
