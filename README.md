# Data Science Midterm Project

## Project Goals
The aim of this project is to develop a machine learning model to predict house prices based on various features.

## Process Overview

### Exploratory Data Analysis (EDA)
- **Data Transformation**: Convert data from JSON to a pandas DataFrame.
- **Data Cleaning**: 
  - Remove columns with more than 50% missing values.
  - Eliminate duplicate entries.
- **Feature Engineering**:
  - Convert categorical features to numerical.
  - Fill missing values with the median.
  - Remove outliers.
- **Data Splitting**: Divide data into training and testing sets.
- **Data Visualization**:
  - Plot the distribution of the target variable.
  - Display the correlation matrix.
  - Create scatter plots between features and the target variable.
- **Data Export**: Save the cleaned data to a new CSV file.

### Model Building
- **Baseline Models**:
  - Linear Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - XGBoost

### Feature Selection
- Use `SelectKBest` to identify the top 10 features with the highest correlation to the target variable.
- Apply regularization methods for feature selection.

### Model Tuning
- Utilize GridSearch for hyperparameter tuning.
- Implement a pipeline for model building.

## Results

### Best Model: XGBRegressor
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.9966             |
| Test Score     | 0.9904             |
| MSE            | 824,864,461.44     |
| MAE            | 3725.76            |
| R²             | 0.9904             |

### Descriptive Statistics for `MAE`
| Statistic | Value     |
|-----------|-----------|
| Count     | 1330.00   |
| Mean      | 3725.76   |
| Std Dev   | 28488.48  |
| Min       | 0.50      |
| 25%       | 146.09    |
| 50%       | 346.73    |
| 75%       | 888.86    |
| Max       | 544763.88 |

## Challenges 
1. **EDA**: Time-consuming data cleaning and transformation to ensure readiness for model building.
2. **Model Tuning**: Extensive time required to tune hyperparameters and identify the best model configuration.

## Future Goals
1. Explore building more complex models, such as deep learning, to enhance performance.
2. Use the model for real-world house price predictions and evaluate its effectiveness.