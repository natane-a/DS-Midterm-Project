import pandas as pd

def encode_tags(df):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
    unique_tags = set()
    for tags in df['tags']:
        if isinstance(tags, list):
            unique_tags.update(tags)
    
    new_columns = {}
    for tag in unique_tags:
        new_columns[tag] = df['tags'].apply(lambda x: 1 if isinstance(x, list) and tag in x else 0)
    
    new_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_df], axis=1)
    return df