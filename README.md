# Predicting Movie Attendance

This Jupyter notebook tests different classifiers/regressors from a sold movie tickets dataset. The goal is to forecast at daily level with at least one week in advance. These results will help the Operations Department with day-to-day inventory management that could benefit their selling concessions activities.

# Abstract
This work presents the methodology to predict movie attendance based on ticket sales and types of movies. Various techniques were used and the random forest approach performed better when comparing R^2 results. Movie tickets sales are from Cinepolis, and the dataset was obtained from 2016 to 2017. We used the following predictor variables: movie studios, type, exposure; month, day of the week, type of day, weekday / weekend and budget to construct the model.
We selected random forest approach after studing the dataset because we did not have to deselect variables and it is a technique the does not overfit the data. The random forest prediction was compared against another regressors. Also, a comparative analysis against an ARIMA model tells us that forecasting with this model reflects the impact of the variables on the movie attendance.
