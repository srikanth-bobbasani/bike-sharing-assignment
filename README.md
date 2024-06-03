# Shared Bike Demand Prediction

> This project builds a multiple linear regression model to predict the demand for shared bikes based on various influencing factors.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Data Description](#data-description)
* [Model Building and Evaluation](#model-building-and-evaluation)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

## General Information
This project addresses the challenge of predicting demand for shared bikes in the American market. BoomBikes, a bike-sharing company, aims to understand this demand to optimize their business strategies post-pandemic.

The model will be used to:
- Identify significant factors affecting bike demand.
- Analyze how these factors influence demand variations.
- Data Source: The data for this project is assumed to be provided by BoomBikes. (Replace with the actual source if different)

## Technologies Used
- Python 3
- Pandas library 
- Scikit-learn library
- Matplotlib library (Optional, for visualization)

## Data Description

### day.csv have the following fields:
	
	- instant: record index
	- dteday : date
	- season : season (1:spring, 2:summer, 3:fall, 4:winter)
	- yr : year (0: 2018, 1:2019)
	- mnth : month ( 1 to 12)
	- holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
	- weekday : day of the week
	- workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
	+ weathersit : 
		- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
		- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
		- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
		- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
	- temp : temperature in Celsius
	- atemp: feeling temperature in Celsius
	- hum: humidity
	- windspeed: wind speed
	- casual: count of casual users
	- registered: count of registered users
	- cnt: count of total rental bikes including both casual and registered

 
### License
Use of this dataset in publications must be cited to the following publication:

[1] Fanaee-T, Hadi, and Gama, Joao, "Event labeling combining ensemble detectors and background knowledge", Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg, doi:10.1007/s13748-013-0040-3.

@article{
	year={2013},
	issn={2192-6352},
	journal={Progress in Artificial Intelligence},
	doi={10.1007/s13748-013-0040-3},
	title={Event labeling combining ensemble detectors and background knowledge},
	url={http://dx.doi.org/10.1007/s13748-013-0040-3},
	publisher={Springer Berlin Heidelberg},
	keywords={Event labeling; Event detection; Ensemble learning; Background knowledge},
	author={Fanaee-T, Hadi and Gama, Joao},
	pages={1-15}
}

### Contact
 - For further information about this dataset please contact Hadi Fanaee-T (hadi.fanaee@fe.up.pt)

## Model Building and Evaluation
### Data Preprocessing:

Load the data.
Handle missing values.
Perform feature engineering as necessary (e.g., encoding categorical variables).
Split the data into training and testing sets.
Model Training:

Train a multiple linear regression model on the training data.
Evaluate the model's performance using metrics like R-squared and mean squared error (MSE).
Model Evaluation:

Make predictions on the testing set.
Calculate R-squared score on the test set.


## Conclusions
The analysis will reveal key factors influencing shared bike demand (based on model results).
The model's R-squared score will indicate how well these factors describe bike demands.
The model can be used for demand forecasting and business strategy optimization.

## Acknowledgements
BoomBikes for providing the dataset and problem statement.

Contact
Created by [@githubusername] - feel free to contact me!
