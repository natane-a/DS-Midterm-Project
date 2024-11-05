import pandas as pd

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