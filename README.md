# Predicting Temperature in London (Python)

**Background:** As the climate changes, predicting the weather becomes ever more important for businesses. Since the weather depends on a lot of different factors, you will want to run a lot of experiments to determine what the best approach is to predict the weather.

**Purpose:** In this project, you will run experiments for different regression models predicting the mean temperature in London, England, using a combination of `sklearn` and `MLflow`.

This project was done in January, 2026. Machine learning was utilized in this project.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Objectives:** Use machine learning to predict the mean temperature in London, England, logging root mean squared error (RMSE) metrics using `mlflow`.
- Build a model to predict `"mean_temp"` with a RMSE of 3 or less.
- Use MLflow to log any models that are trained, their hyperparameters, & respective RMSE scores (include `"rmse"` as part of the metric name).
- Search all `mlflow` runs & store the results (`experiment_results`).


### Brief Summary
A brief evaluation of the dataset revealed that there were only a handful of missing values in a dataset with 15,341 data points. The dataset contained daily metrics from London in 1979-2020. Over time, the temperature increased by about 0.065 degrees Celsius per year.  
The variables with the greatest correlation with the average temperature were the amount of sunshine & global radiation. In other words, the more sunlight & radiation that London saw during this time period, the greater the average temperatures were.

Three machine learning algorithms were utilized to cast predictions: linear regression, decision tree regression, & random forest regression. A subset of the variables in the dataset were used to predict the average temperature including: the month, amount of cloud cover, sunshine, precipitation, pressure, & global radiation. The max-depth was the only parameter tuned. As such, nine total runs were produced in the MLflow experiment, three for each method.

Of these nine models, only one returned a root mean square error less than 3.0; the **random forest** model with a `max_depth` of 10 had a RMSE of about **2.848**.
- In comparing the three MLFlow runs, a higher `max_depth` value was definitely more beneficial in terms of producing a more accurate model, particularly for the decision tree & random forest models. The linear regression models had the same RMSE in all three runs because linear regression models don't use such a hyperparameter.
- The three linear regression models had the lowest average RMSE (~ 3.757), just barely better than that of the random forest models (~ 3.760). The decision tree models had the highest average RMSE; ~ 3.873.

### Recommendations
Given the results, the random forest models can be considered to produce the best fits for this dataset given the specific feature variables included, particularly those with a larger `max_depth`.

Further analysis & experimentation could explore additional hyperparater tuning options as well as altering what variables were used (or not used) in predicting the average temperature. It is likely that including the variables with the strongest correlation with 'mean_temp' will produce more accurate models because they change in relation to one another. In this case, those variables are 'sunshine' & 'global_radiation.'
