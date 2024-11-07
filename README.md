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
| Training Score | 0.8251             |
| Test Score     | 0.9135             |
| MSE            | 142,855.84         |

### Random Forest
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.9894             |
| Test Score     | 0.7269             |
| RMSE           | 6253,870.21        |

### Gradient Boosting
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.9947             |
| Test Score     | 0.7527             |
| RMSE           | 241,608.52         |

### Support Vector Machine
| Metric         | Value              |
|----------------|--------------------|
| Training Score | 0.2965             |
| Test Score     | 0.2712             |
| RMSE           | 414,743.61         |
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
1. Explore building more complex models, such as deep learning, to enhance performance.
2. Use the model for real-world house price predictions and evaluate its effectiveness.
3. Fix issues concerning overfitting.