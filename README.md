# Predicting Movie Attendance

This Jupyter notebook tests different classifiers/regressors from a sold movie tickets dataset. The goal is to forecast at daily level with at least one week in advance. These results will help the Operations Department with day-to-day inventory management that could benefit their selling concessions activities.

# Abstract
This work presents the methodology to predict movie attendance based on ticket sales and types of movies. Various techniques were used and the random forest approach performed better when comparing R^2 results. Movie tickets sales are from Cinepolis, and the dataset was obtained from 2016 to 2017. We used the following predictor variables: movie studios, type, exposure; month, day of the week, type of day, weekday / weekend and budget to construct the model.
We selected random forest approach after studing the dataset, mainly because we did not have to deselect variables and it is a technique the does not overfit the data. The baseline was a Linear Model. The random forest prediction was compared against Gradient Boost regressor and then both hypertuned. The random forest performed better. Also, a comparative analysis against an ARIMA model tells us that forecasting with this model reflects better the impact of the variables on the movie attendance and tends to follow the empirical results.

# Results
The following graph shows the 7-day in advance forecasting.
![](images/Figure_1.png.png)

# Conclusion
This analysis uses a tickets sales combined with the movie data and holidays over a period of two years. Using feature engineerig, categories were created to bin the budget, time displayed in theaters, number of theathers where the movie was shown and the studio that produced the movie. Analazing this dataset, the statistical tests show that there are features that skew the sample. Mainly some weekdays and weekends (Friday to Sunday). When constructing the linear models, a baseline was a Linear Regression Model according to the R^2 score and the residuals vs predictors. Then we tried a boosting technique and a classifier: Gradient Boosting and Random Forests. Random Forest regression probed to have a better score than linear regression. Also we treated the data as a time series and used an ARIMA model to test the forecasting. But this model goes againt the empirical and statistical findings about the importance of the end of the week (Friday to Sunday). We can conclude that random forests works better to predict attendance performed better with this data set. Random Forest is a machine learning technique that does not require preselecting the covariates and it naturally avoids the hazard of overfitting. It also captures the importance of holidays and weekends in the attendance to theaters forecast.
