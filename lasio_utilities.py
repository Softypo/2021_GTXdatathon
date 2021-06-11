import concurrent.futures
import pandas as pd
#import numpy as np
import lasio
from os import listdir
from tqdm.notebook import tqdm

def filt_m(df, look=()):
    name = df[0]
    d = df[1]
    add = []
    for col in d.columns:
        if col.startswith(look):
            add.append(col)
    if len(add)>=1: return [name, d[add]]


# look for mnemonics
def filter_multiprocess(df, look=()):
    """load well logs las files from a folder and store them in a dictionary.
    
    parameters
    ----------
    path : folder path containg the desired las files.
    ext: bol, include extention in file name
    """

    look_lst = [look for _ in range(len(df))]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        filtered = {f[0]: f[1] for f in tqdm(executor.map(filt_m, df.items(), look_lst), total=len(df), desc='Searching') if isinstance(f, list)}
        #df_files = {f[0]: f[1] for f in tqdm(executor.map(todf, list(las_files.items())), total=len(list(las_files.items())))}
    return filtered

# look for mnemonics
def filter(df, look=(), mean=True):
    filtered = {}
    for name, d in tqdm(df.items(), desc='Searching'):
        add = []
        for col in d.columns:
            if col.startswith(look):
                add.append(col)
        if len(add)>=1:
            filtered[name] = d[add]

    if mean==True:
        filteredmean = {}
        for name, d in tqdm(filtered.items(), desc='Calculating mean'):
            nd = pd.DataFrame(index=d.index)
            for mnemonic in look:
                add = []
                for col in d.columns:
                    if isinstance(look, (str)):
                        if col.startswith(look): add.append(col)
                    elif col.startswith(mnemonic):
                        add.append(col)
                if isinstance(look, (str)):
                    nd[look] = d[add].mean(axis=1)
                    break
                else: nd[mnemonic] = d[add].mean(axis=1)
            filteredmean[name] = nd
        return filteredmean
    else: return filtered


# transform lasio to dataframe
def todf(las_file):
    """transform lasio object to pandas dataframe.
    
    parameters
    ----------
    las_file : lasio object.
    """ 
    if isinstance(las_file, (list, tuple)):
        name = las_file[1].well.uwi.value
        unit = las_file[1].well.step.unit
        return [name, unit, las_file[1].df()]
    else: return las_file.df()

# load las file from a folder for lasio
def load(filename, path=None, ext=False):
    """load las file.
    
    parameters
    ----------
    filename : name of the file including extension.
    path : file path.
    ext : bol, default False. include extencion.
    """ 
    if ext==False: name = filename.rpartition('.')[0].replace(' ', '_')
    else: name =filename.replace(' ', '_')
    if filename.endswith(('.las', '.LAS')): return [name, lasio.read(path+'\\'+filename)]


# load las files from a folder for lasio multiprocessing enabled
def loader_lasio_multiprocess(path=None, ext=False, todrop=None, outlas=True, outdf=True):
    """load well logs las files from a folder and store them in a dictionary.
    
    parameters
    ----------
    path : folder path containg the desired las files.
    ext: bol, include extention in file name
    todrop: list/tuple of filenames not to be included in the dataframe dictionary
    """
    path_lst = [path for _ in range(len(listdir(path)))]
    ext_lst = [ext for _ in range(len(listdir(path)))]
    if todrop==None: todrop = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        las_files = {f[0]: f[1] for f in tqdm(executor.map(load, listdir(path), path_lst, ext_lst), total=len(listdir(path)), desc='Loading LAS files') if f[0] not in todrop}
        if outdf==True: df_files = {f[0]: [f[1], f[2]] for f in tqdm(executor.map(todf, list(las_files.items())), total=len(list(las_files.items())), desc='LAS to Pandas Dataframe')}
    if outlas==True and outdf==True: return las_files, df_files
    elif outlas==True and outdf==False: return las_files
    elif outlas==False and outdf==True: return df_files


# load las files from a folder for lasio
def loader_lasio(path=None, ext=False):
    """load well logs las files from a folder and store them in a dictionary.
    
    parameters
    ----------
    path : folder path containg the desired las files.
    ext: bol, include extention in file name
    """
    try:
        lasio_files, las_df= {}, {}
        for filename in tqdm(listdir(path), desc='Loading LAS files'):
            if ext==False: name = filename.rpartition('.')[0].replace(' ', '_')
            else: name =filename.replace(' ', '_')
            if filename.endswith(('.las', '.LAS')):
                lasio_files[name] = lasio.read(path+'\\'+filename)
                las_df[lasio_files[name].well.UWI.value] = lasio_files[name].df()
        return lasio_files, las_df
    except:
        print ('No files loaded for \\',path)

def main ():
    las_files, lasdf = loader_lasio_multiprocess('Data for Datathon\well_log_files\Clean_LAS')
    filter(lasdf, ('GR', 'DT'))
    pass

if __name__ == "__main__":
    main()
