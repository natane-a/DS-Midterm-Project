import pandas as pd
import itertools
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def encode_tags(df, freq_threshold=1000):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    unique_tags = {}
    for tags in df['tags']:
        if isinstance(tags, list):
            for tag in tags:
                unique_tags[tag] = unique_tags.get(tag, 0) +1
    
    new_columns = {}
    for tag in unique_tags.keys():
        if unique_tags[tag] > freq_threshold:
            new_columns[tag] = df['tags'].apply(lambda x: 1 if isinstance(x, list) and tag in x else 0)
    
    new_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_df], axis=1)
    return df

def custom_cross_validation(training_data, n_splits =5):
    '''creates n_splits sets of training and validation folds

    Args:
      training_data: the dataframe of features and target to be divided into folds
      n_splits: the number of sets of folds to be created

    Returns:
      A tuple of lists, where the first index is a list of the training folds, 
      and the second the corresponding validation fold

    Example:
        >>> output = custom_cross_validation(train_df, n_splits = 10)
        >>> output[0][0] # The first training fold
        >>> output[1][0] # The first validation fold
        >>> output[0][1] # The second training fold
        >>> output[1][1] # The second validation fold... etc.
    '''
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_folds = []
    val_folds =[]
    
    for training_index, val_index in kfold.split(X_train):
        train_fold = X_train.iloc[training_index]
        val_fold = X_train.iloc[val_index]
        train_folds.append(train_fold)
        val_folds.append(val_fold)
    
    return train_folds, val_folds

def hyperparameter_search(train_folds, val_folds, param_grid, model, metric=r2_score):
    '''outputs the best combination of hyperparameter settings in the param grid, 
    given the training and validation folds

    Args:
      training_folds: the list of training fold dataframes
      validation_folds: the list of validation fold dataframes
      param_grid: the dictionary of possible hyperparameter values for the chosen model
      Example:
            param_grid = {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
      model (estimator): The model instance to be trained and evaluated.
      metric (callable, optional): The evaluation metric function to optimize. Default is `r2_score` from sklearn.metrics.

    Returns:
      A list of the best hyperparameter settings based on the chosen metric

    Example:
        >>> best_hyperparams = hyperparameter_search(train_folds, val_folds, param_grid, RandomForestRegressor(), metric=r2_score)
        >>> best_hyperparams
        {'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2'}
    '''
    
    hyperparams = list(itertools.product(*param_grid.values()))
    hyperparam_scores = []

    for hyperparam_combo in hyperparams:
        param_dict = dict(zip(param_grid.keys(), hyperparam_combo))
        scores = []

        for train_fold, val_fold in zip(train_folds, val_folds):
            X_train_fold = train_fold.drop(columns=['target'])
            y_train_fold = train_fold['target']
            X_val_fold = val_fold.drop(columns=['target'])
            y_val_fold = val_fold['target']
            
            # set parameters and train the model
            model.set_params(**param_dict)
            model.fit(X_train_fold, y_train_fold)
            
            # predict and evaluate the model
            y_pred = model.predict(X_val_fold)
            score_fold = metric(y_val_fold, y_pred)  # r-squared score for this fold
            scores.append(score_fold)

        # average score across all folds
        score = np.mean(scores)
        hyperparam_scores.append(score)

    # find the index of the best hyperparameters based on the highest R-squared score
    best_index = np.argmax(hyperparam_scores)
    best_hyperparams = hyperparams[best_index]

    return best_hyperparams
