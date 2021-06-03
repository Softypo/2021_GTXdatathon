# some usefull pandas dataframe extra functions for data cleaning
import pandas as pd
from os import listdir

# load csv and excel files from a folder and store them in a dictionary
def loader(path=None, index_col=None, ext=False):
    """load csv and excel files from a folder and store them in a dictionary.
    
    parameters
    ----------
    path : folder path containg the desired csv and/or excel files.
    index_col : int, column number to be used as index.
    """
    try:
        files = {}
        for filename in listdir(path):
            if ext==False: name = filename.rpartition('.')[0].replace(' ', '_')
            else: name =filename.replace(' ', '_')
            if filename.endswith('.xlsx'):
                files[name] = pd.read_excel(path+'\\'+filename, index_col=index_col)
            elif filename.endswith('.csv'):
                files[name] = pd.read_csv(path+'\\'+filename, index_col=index_col)
        return files
    except Exception:
        print ('Exception: No files loaded for \\', path)


# drop columns or rows with all unique values
def drop_unique (df=None, axis=1, inplace=False):
    """drop columns or rows with all unique values.
    
    parameters
    ----------
    df : pandas dataframe or mutable array like containing them.
    axis : 0 for rows wise or 1 for columns wise.
    inplace : bool, default False, if True, do operation inplace and return None.
    """
    try:
        if isinstance(df, pd.DataFrame):
            return df.drop(df.nunique()[(df.nunique() == 1)].index, axis=axis, inplace=inplace)
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                lst.append(d.drop(d.nunique()[(d.nunique() == 1)].index, axis=axis, inplace=inplace))
            if inplace==False: return lst
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                lst[k] = d.drop(d.nunique()[(d.nunique() == 1)].index, axis=axis, inplace=inplace)
            if inplace==False: return lst
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a list/tuple containing them')


# boost of original pd.dropna funtion,
def drop_nan (df=None, axis=1, how='all', inplace=False):
    """boost of original pd.dropna funtion,
    by saving time by adding the feature of running several dataframe.
    
    parameters
    ----------
    df : pandas dataframe or mutable array like containing them.
    axis : 0 for rows wise or 1 for columns wise.
    how : {'any', 'all'}, default 'any'.
    * 'any' : If any NA values are present, drop that row or column.
    * 'all' : If all values are NA, drop that row or column.
    inplace : bool, default False, if True, do operation inplace and return None.
    """
    try:
        if isinstance(df, pd.DataFrame):
            return df.dropna(axis=axis, how=how, inplace=inplace)
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                lst.append(d.dropna(axis=axis, how=how, inplace=inplace))
            if inplace==False: return lst
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                lst[k] = d.dropna(axis=axis, how=how, inplace=inplace)
            if inplace==False: return lst
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a list/tuple containing them')


# thanks to https://www.geeksforgeeks.org/how-to-find-drop-duplicate-columns-in-a-pandas-dataframe/
# This function take a dataframe
# as a parameter and returning list
# of column names whose contents 
# are duplicates.
def getDuplicateColumns(df):
    """return a labels list of equal value columns.
    
    parameters
    ----------
    df : pandas dataframe or mutable array like containing them.
    """
    # Create an empty set
    duplicateColumnNames = set()

    # comparing columns agains index
    for x in range(df.shape[1]):
        col = df.iloc[:, x]
        tempdf = df.set_index(col)
        if tempdf.index.equals(df.index):
            duplicateColumnNames.add(df.columns.values[x])

    # Iterate through all the columns 
    # of dataframe
    for x in range(df.shape[1]):

        # Take column at xth index.
        col = df.iloc[:, x]

        # Iterate through all the columns in
        # DataFrame from (x + 1)th index to
        # last index
        for y in range(x + 1, df.shape[1]):

            # Take column at yth index.
            otherCol = df.iloc[:, y]

            # Check if two columns at x & y
            # index are equal or not,
            # if equal then adding 
            # to the set
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
                  
    # Return list of unique column names 
    # whose contents are duplicates.
    return list(duplicateColumnNames)

# drop equal value columns
def dropEqualColumns (df=None, inplace=False):
    """drop equal value columns.
    
    parameters
    ----------
    df : pandas dataframe or mutable array like containing them.
    inplace : bool, default False, if True, do operation inplace and return None.

    Note: returns none for array like inputs.
    """
    try:
        if isinstance(df, pd.DataFrame):
            dupes = getDuplicateColumns(df)
            print('duplicates: ',dupes)
            return df.drop(columns = dupes, inplace=inplace)
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                dupes = getDuplicateColumns(d)
                print(dupes)
                lst.append(d.drop(columns = dupes, inplace=inplace))
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                dupes = getDuplicateColumns(d)
                print('duplicates in ',k,dupes)
                lst[k] = d.drop(columns = dupes, inplace=inplace)
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a list/tuple containing them')



# rename index title name
def renameIndex (name, df=None, inplace=False):
    """rename index column title.
    
    parameters
    ----------
    name: string, new name
    df : pandas dataframe or mutable array like containing them.
    inplace : bool, default False, if True, do operation inplace and return None.

    Note: returns none for array like inputs.
    """
    try:
        if isinstance(df, pd.DataFrame):
            return df.index.set_names(name, inplace=inplace)
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                lst.append(d.index.set_names(name, inplace=inplace))
            if inplace==False: return lst
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                lst[k] = d.index.set_names(name, inplace=inplace)
            if inplace==False: return lst
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a list/tuple containing them')


def main ():
    pass

if __name__ == "__main__":
    main()
