from os import listdir
from os.path import isfile, join
import numpy as np
from skimage.transform import resize
from tqdm import tqdm_notebook, tnrange
import dask.array as da

## Loading training data

def load_training_data(folder, chunks='auto'):

    path_folder_cl = folder + '/CL/'
    path_folder_bse = folder + '/BSE/'
    path_folder_mm = folder + '/MM/'
    path_folder_mask = folder + '/mask_cleaned/'

    onlyfiles_cl = [f for f in listdir(path_folder_cl) if isfile(join(path_folder_cl, f))]
    onlyfiles_bse = [f for f in listdir( path_folder_bse) if isfile(join(path_folder_bse, f))]
    onlyfiles_mm = [f for f in listdir(path_folder_mm) if isfile(join(path_folder_mm, f))]
    onlyfiles_mask = [f for f in listdir(path_folder_mask) if isfile(join(path_folder_mask, f))]   

    print('Found' , len(onlyfiles_cl), 'files in CL folder:')
    print('Found' , len(onlyfiles_bse), 'files in BSE folder:')
    print('Found' , len(onlyfiles_mm), 'files in MM folder:')
    print('Found' , len(onlyfiles_mask), 'files in MASK folder:')

    im_list = []
    h, w  = (768, 1024)


    X = np.zeros([len(onlyfiles_mask), w*2, w*2])
    y = np.zeros(X.shape)


    k=0
    for im_name in onlyfiles_cl:
        # Check if image is in BSE, MM and MASK folder
        #print(im_name)
        if isfile(join(path_folder_bse, im_name)) and isfile(join(path_folder_mm, im_name)) and isfile(join(path_folder_mask, im_name.replace('.tif' , '_mask_cleaned.tif'))) == True:
            im_list.append(im_name)
    # dataset_list[k]['CL_'+im_name]   = cv2.imread(path_folder_cl   + im_name , 0)
    # dataset_list[k]['BSE_'+im_name]  = cv2.imread(path_folder_bse  + im_name , 0)
    # dataset_list[k]['MM_'+im_name]   = cv2.imread(path_folder_mm   + im_name , 0)
    # dataset_list[k]['MASK_'+im_name] = cv2.imread(path_folder_mask + im_name.replace('.tif' , '_mask_cleaned.tif') , 0)

            # Mering images into large frames (2048 x 2048)
            input_dummy = np.zeros([ w*2 , w*2]) 
            mask_dummy  = np.zeros([ w*2 , w*2]) - 1
            
            input_dummy[:h, :w]    = cv2.imread(path_folder_bse  + im_name , 0)
            input_dummy[:h, w:w*2] = cv2.imread(path_folder_cl   + im_name , 0)
            input_dummy[h:h*2, :w] = cv2.imread(path_folder_mm   + im_name , 0)
            mask_dummy[:h, :w]    = cv2.imread(path_folder_mask + im_name.replace('.tif' , '_mask_cleaned.tif') , 0)

            X[k,:,:] = input_dummy
            y[k,:,:] = mask_dummy

            k += 1 

    X = X.astype('uint8')
    X = X / 255 

    y = y.astype('int8')
    y[y==0] = -1
    print('Unique mask IDs:' , np.unique(y), 'of dtype:', y.dtype)
    print('Done!')


    if chunks is not 'auto':
        X = X.rechunk(chunks)
        y = y.rechunk(chunks)
    return X, y, im_list


def train_test_plit(X, y, im_list)
    ids = np.arange(X.shape[0])
    np.random.shuffle(ids)
    test_size =int(X.shape[0]*0.8)
    train_ind = ids[:test_size]
    test_ind = ids[test_size:]

    X_train = X[train_ind]
    X_test = X[test_ind]
    y_train = y[train_ind]
    y_test = y[test_ind]

    im_list_train = np.array(im_list)[train_ind]
    im_list_test = np.array(im_list)[test_ind]

    return X_train, y_train, im_list_train, X_test, y_test, im_list_test

 #   train_gen = DaskGenerator(X_train, y_train)
 #   test_gen = DaskGenerator(X_test, y_test)