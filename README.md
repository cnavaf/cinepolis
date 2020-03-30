# Predicting Movie Attendance

This Jupyter notebook tests different classifiers/regressors from a sold movie tickets dataset. The goal is to forecast at daily level with at least one week in advance. These results will help the Operations Department with day-to-day inventory management that could benefit their selling concessions activities.

# Abstract
This work presents the methodology to predict movie attendance based on ticket sales and types of movies. Various techniques were used and the random forest approach performed better when comparing R^2 results. Movie tickets sales are from Cinepolis, and the dataset was obtained from 2016 to 2017. We used the following predictor variables: movie studios, type, exposure; month, day of the week, type of day, weekday / weekend and budget to construct the model.
We selected random forest approach after studing the dataset, mainly because we did not have to deselect variables and it is a technique the does not overfit the data. The baseline was a Linear Model. The random forest prediction was compared against Gradient Boost regressor and then both hypertuned. The random forest performed better. Also, a comparative analysis against an ARIMA model tells us that forecasting with this model reflects better the impact of the variables on the movie attendance and tends to follow the empirical results.

# Results
Acording to the model we get the following forecasting values (including min/max/mean):

Date      |    ARIMA |    RndFR |   GradBR |       LR |      Min |      Max |      Mean
----------|----------|----------|----------|----------|----------|----------|----------
2018-01-01|  2.324806|  0.597408|  0.504941|  0.996014|  0.504941|  2.324806|  1.105792
2018-01-02|  1.948318|  0.622260|  0.747829|  0.778737|  0.622260|  1.948318|  1.024286
2018-01-03|  2.270043|  0.889823|  0.902034|  1.048118|  0.889823|  2.270043|  1.277504
2018-01-04|  1.871790|  0.604766|  0.701904|  0.664115|  0.604766|  1.871790|  0.960644
2018-01-05|  1.941841|  1.655673|  1.534452|  1.579336|  1.534452|  1.941841|  1.677825
2018-01-06|  2.007034|  2.296866|  2.112396|  2.377848|  2.007034|  2.377848|  2.198536
2018-01-07|  0.652987|  2.769174|  2.647798|  2.870293|  0.652987|  2.870293|  2.235063


And, the following graph shows the 7-day in advance forecasting.

![Results](/images/Figure_1.png)

# Conclusion
This analysis uses a tickets sales combined with the movie data and holidays over a period of two years. Using feature engineerig, categories were created to bin the budget, time displayed in theaters, number of theathers where the movie was shown and the studio that produced the movie. Analazing this dataset, the statistical tests show that there are features that skew the sample. Mainly some weekdays and weekends (Friday to Sunday). When constructing the linear models, a baseline was a Linear Regression Model according to the R^2 score and the residuals vs predictors. Then we tried a boosting technique and a classifier: Gradient Boosting and Random Forests. Random Forest regression probed to have a better score than linear regression. Also we treated the data as a time series and used an ARIMA model to test the forecasting. But this model goes againt the empirical and statistical findings about the importance of the end of the week (Friday to Sunday). We can conclude that random forests works better to predict attendance performed better with this data set. Random Forest is a machine learning technique that does not require preselecting the covariates and it naturally avoids the hazard of overfitting. It also captures the importance of holidays and weekends in the attendance to theaters forecast.

We can conclude: 1) Cinema attendance is highly stationary. Weekends, holidays, summer and winter vacations is when we have higher rates of attendance. 2) Blockbusters from some studios tend to attract higher numbers of fans. 3) Using the Random Forest Classifier to forecast a week in advance can probe to be valuable for operations and their objetive to manage the inventory of selling concessions; avoid losses and get more profit.
