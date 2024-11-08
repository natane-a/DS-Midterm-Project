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
| Test Score     | 0.7425             |
| MSE            | 60,786,794,030.98  |
| MAE            | 9,667.04           |
| RÂ²            | 0.7425             |

### Linear Regression
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.8802             |
| Test Score     | 0.9125             |
| MSE            | 80994.67           |

### Random Forest
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.9892             |
| Test Score     | 0.9898             |
| RMSE           | 27628.42           |

### Gradient Boosting
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.9898             |
| Test Score     | 0.9899             |
| RMSE           | 27462.38           |

### Support Vector Machine
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.4107             |
| Test Score     | 0.4632             |
| RMSE           | 200655.40          |
### Descriptive Statistics for `description.sold_price`
| Statistic | Value      |
|-----------|------------|
| Count     | 1343.00    |
| Mean      | 9667.04    |
| Std Dev   | 246451.96  |
| Min       | 0.00       |
| 25%       | 8.16       |
| 50%       | 20.53      |
| 75%       | 44.16      |
| Max       | 8975225.00 |

## Challenges 
1. **EDA**: Time-consuming data cleaning and transformation to ensure readiness for model building.
2. **Model Tuning**: Extensive time required to tune hyperparameters and identify the best model configuration.

## Future Goals
1. Explore building more complex models, such as deep learning and time series models, to enhance performance.
2. Use the model for real-world house price predictions and evaluate its effectiveness.
