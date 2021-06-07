# some usefull pandas dataframe extra functions for data cleaning
import pandas as pd
import numpy as np
from os import listdir

# load csv and excel files from a folder and store them in a dictionary
def loader(path=None, index_col=None, ext=False, output='dict'):
    """load csv and excel files from a folder and store them in a dictionary.
    
    parameters
    ----------
    path : folder path containg the desired csv and/or excel files.
    index_col : int, column number to be used as index.
    ext: bol, include extention in file name
    output: str, 'dict' or 'list'
    """
    try:
        if output=='dict':
            files = {}
            for filename in listdir(path):
                if ext==False: name = filename.rpartition('.')[0].replace(' ', '_')
                else: name =filename.replace(' ', '_')
                if filename.endswith('.xlsx'):
                    files[name] = pd.read_excel(path+'\\'+filename, index_col=index_col,)
                elif filename.endswith('.csv'):
                    files[name] = pd.read_csv(path+'\\'+filename, index_col=index_col)
            return files
        elif output=='list':
            files = []
            for filename in listdir(path):
                if ext==False: name = filename.rpartition('.')[0].replace(' ', '_')
                else: name =filename.replace(' ', '_')
                if filename.endswith('.xlsx'):
                    files.append([name, pd.read_excel(path+'\\'+filename, index_col=index_col)])
                elif filename.endswith('.csv'):
                    files.append([name, pd.read_excel(path+'\\'+filename, index_col=index_col)])
            return files
    except:
        print ('No files loaded for \\', path)


# boost of original pd.merge funtion,
def multiadd (dfs=None):
    """takes al dataframes in a array like object and returns it's sum.

    Parameters
    ----------
    dfs : mutable array like containing DataFrames
        Object to merge with.
    """
    #try:
    if isinstance(dfs, dict): dfs = list(dfs.values())
    if isinstance(dfs, (list, tuple)):
        df = pd.DataFrame(index=dfs[0].index)
        for d in dfs:
            df = df+d
        return df
    #except:
    #    print ('Exception: input is not a dict/list/tuple containing them')


# boost of original pd.merge funtion,
def multimerge (dfs=None, **Kargs):
    """adding the option of multiple merge.
    
    ( dfs, how: str = 'inner', on=None, left_on=None, right_on=None, left_index: bool = False, right_index: bool = False, sort: bool = False, suffixes=('_x', '_y'), copy: bool = True, indicator: bool = False, validate=None, ) -> 'DataFrame'
Merge DataFrame or named Series objects with a database-style join.

    Parameters
    ----------
    dfs : mutable array like containing DataFrames
        Object to merge with.
    how : {'left', 'right', 'outer', 'inner', 'cross'}, default 'inner'
        Type of merge to be performed.

        * left: use only keys from left frame, similar to a SQL left outer join;
        preserve key order.
        * right: use only keys from right frame, similar to a SQL right outer join;
        preserve key order.
        * outer: use union of keys from both frames, similar to a SQL full outer
        join; sort keys lexicographically.
        * inner: use intersection of keys from both frames, similar to a SQL inner
        join; preserve the order of the left keys.
        * cross: creates the cartesian product from both frames, preserves the order
        of the left keys.

        .. versionadded:: 1.2.0

    on : label or list
        Column or index level names to join on. These must be found in both
        DataFrames. If `on` is None and not merging on indexes then this defaults
        to the intersection of the columns in both DataFrames.
    left_on : label or list, or array-like
        Column or index level names to join on in the left DataFrame. Can also
        be an array or list of arrays of the length of the left DataFrame.
        These arrays are treated as if they are columns.
    right_on : label or list, or array-like
        Column or index level names to join on in the right DataFrame. Can also
        be an array or list of arrays of the length of the right DataFrame.
        These arrays are treated as if they are columns.
    left_index : bool, default False
        Use the index from the left DataFrame as the join key(s). If it is a
        MultiIndex, the number of keys in the other DataFrame (either the index
        or a number of columns) must match the number of levels.
    right_index : bool, default False
        Use the index from the right DataFrame as the join key. Same caveats as
        left_index.
    sort : bool, default False
        Sort the join keys lexicographically in the result DataFrame. If False,
        the order of the join keys depends on the join type (how keyword).
    suffixes : list-like, default is ("_x", "_y")
        A length-2 sequence where each element is optionally a string
        indicating the suffix to add to overlapping column names in
        `left` and `right` respectively. Pass a value of `None` instead
        of a string to indicate that the column name from `left` or
        `right` should be left as-is, with no suffix. At least one of the
        values must not be None.
    copy : bool, default True
        If False, avoid copy if possible.
    indicator : bool or str, default False
        If True, adds a column to the output DataFrame called "_merge" with
        information on the source of each row. The column can be given a different
        name by providing a string argument. The column will have a Categorical
        type with the value of "left_only" for observations whose merge key only
        appears in the left DataFrame, "right_only" for observations
        whose merge key only appears in the right DataFrame, and "both"
        if the observation's merge key is found in both DataFrames.

validate : str, optional
    If specified, checks if merge is of specified type.

    * "one_to_one" or "1:1": check if merge keys are unique in both

    """
    if isinstance(dfs, dict): dfs = list(dfs.values())
    if isinstance(dfs, (list, tuple)):
        df = pd.DataFrame(index=dfs[0].index)
        for d in dfs:
            df = pd.merge(df, d, **Kargs)
        return df


# drop columns or rows with all unique values
def drop_unique (df=None, inplace=False, **kargs):
    """drop columns or rows with all unique values.
    
    parameters
    ----------
    df : pandas dataframe or mutable array like containing them.
    axis: {0 or ‘index’, 1 or ‘columns’}, default 0
        Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).
    indexs: ingle label or list-like
        Alternative to specifying axis (labels, axis=0 is equivalent to index=labels).
    columns: single label or list-like
        Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).
    level: int or level name, optional
        For MultiIndex, level from which the labels will be removed.
    inplace: bool, default False
        If False, return a copy. Otherwise, do operation inplace and return None.
    errors: {‘ignore’, ‘raise’}, default ‘raise’
        If ‘ignore’, suppress error and only existing labels are dropped.
    """
    try:
        if isinstance(df, pd.DataFrame):
            return df.drop(df.nunique()[(df.nunique() == 1)].index,inplace=inplace, **kargs)
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                lst.append(d.drop(d.nunique()[(d.nunique() == 1)].index,inplace=inplace, **kargs))
            if inplace==False: return lst
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                lst[k] = d.drop(d.nunique()[(d.nunique() == 1)].index,inplace=inplace, **kargs)
            if inplace==False: return lst
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a dict/list/tuple containing them')


# boost of original pd.dropna funtion,
def drop_nan (df=None, inplace=False, **kargs):
    """boost of original pd.dropna funtion,
    by saving time by adding the feature of running several dataframes.
    
    parameters
    ----------
    df : pandas dataframe or mutable array like containing them.
    axis : 0 for rows wise or 1 for columns wise.
    how : {'any', 'all'}, default 'any'.
        'any' : If any NA values are present, drop that row or column.
        'all' : If all values are NA, drop that row or column.
    inplace : bool, default False, if True, do operation inplace and return None.
    thresh: int, optional
        Require that many non-NA values.
    subset: array-like, optional
        Labels along other axis to consider, e.g. if you are dropping rows these would be a list of columns to include.
    inplace: bool, default False
        If True, do operation inplace and return None.
    """
    try:
        if isinstance(df, pd.DataFrame):
            return df.dropna(inplace=inplace, **kargs)
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                lst.append(d.dropna(inplace=inplace, **kargs))
            if inplace==False: return lst
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                lst[k] = d.dropna(inplace=inplace, **kargs)
            if inplace==False: return lst
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a dict/list/tuple containing them')


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
                print('duplicates: ',dupes)
                lst.append(d.drop(columns = dupes, inplace=inplace))
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                dupes = getDuplicateColumns(d)
                print('duplicates in ',k,dupes)
                lst[k] = d.drop(columns = dupes, inplace=inplace)
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a dict/list/tuple containing them')


# drop equal value columns
def dropEqualRows (df=None, keep='first', view=False):
    """drop equal value columns.
    
    parameters
    ----------
    df : pandas dataframe or mutable array like containing them.
    keep{‘first’, ‘last’, False}, default ‘first’
        The value or values in a set of duplicates to mark as missing.
            ‘first’ : Mark duplicates as True except for the first occurrence.
            ‘last’ : Mark duplicates as True except for the last occurrence.
            False : Mark all duplicates as True.
    view : bool, default False, if True, do operation returns the same input like object.
    """
    try:
        if isinstance(df, pd.DataFrame):
            dupes = df.loc[df.index.duplicated()].index
            print('duplicated indexes: ',dupes)
            if view==True:
                df = df[~df.index.duplicated(keep=keep)]
                return df
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                dupes = d.loc[d.index.duplicated()].index
                print('duplicated indexes: ',dupes)
                if view==True: d = d[~d.index.duplicated(keep=keep)]
                lst.append(d)
                return lst
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                dupes = d.loc[d.index.duplicated()].index
                print('duplicated indexes in ',k,dupes)
                if view==True:
                    lst[k] = d[~d.index.duplicated(keep=keep)]
                    return lst
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a dict/list/tuple containing them')


# rename index title name
def renameIndex (names, df=None, inplace=False, **kargs):
    """rename index column title.
    
    parameters
    ----------
    names: label or list of label
        Name(s) to set.
    df : pandas dataframe or mutable array like containing them.
    level: int, label or list of int or label, optional
        If the index is a MultiIndex, level(s) to set (None for all levels). Otherwise level must be None.
    inplace: bool, default False
        Modifies the object directly, instead of creating a new Index or MultiIndex.

    Note: returns none for array like inputs.
    """
    try:
        if isinstance(df, pd.DataFrame):
            return df.index.set_names(names,inplace=inplace, **kargs)
        elif isinstance(df, (list, tuple)):
            lst=[]
            for d in df:
                lst.append(d.index.set_names(names,inplace=inplace, **kargs))
            if inplace==False: return lst
        elif isinstance(df, (dict)):
            lst={}
            for k, d in df.items():
                lst[k] = d.index.set_names(names,inplace=inplace, **kargs)
            if inplace==False: return lst
    except Exception:
        print ('Exception: input is not a pandas dataframe nor a dict/list/tuple containing them')


def main ():
    pass

if __name__ == "__main__":
    main()
