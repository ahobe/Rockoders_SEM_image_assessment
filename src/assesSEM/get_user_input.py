import os
import sys
import glob
from os.path import isfile, join


def get_folder_names():
    value = input("Please choose folders:\n" +
                  "0: All folders\n" +
                  "1: dataset1\n" +
                  "2: dataset2\n" +
                  "3: dataset3\n" +
                  "4: dataset4A\n"
                  "5: dataset4B\n"
                  )
    if value == "0":
        names = ['dataset1', 'dataset2', 'dataset3', 'dataset4A', 'dataset4B']
    elif value == "1":
        names = ['dataset1']
    elif value == "2":
        names = ['dataset2']
    elif value == "3":
        names = ['dataset3']
    elif value == "4":
        names = ['dataset4A']
    elif value == "5":
        names = ['dataset4B']
    else:
        raise ValueError
    return names


def deal_with_folder_availability(path):
    if not os.path.exists(path):
        os.mkdir(path)
        empty_folder = True
    else:
        files_in_dir = os.listdir(path)
        if len(files_in_dir) == 0:
            empty_folder = True
        else:
            empty_folder = False

    overwrite_ok = False
    if not empty_folder:
        overwrite_ok = get_ok_for_overwrite(path)

    if overwrite_ok or empty_folder:  # either or both True
        allowed_to_continue = True
        return allowed_to_continue


def get_ok_for_overwrite(folder_name):
    answer = input(f"Folder {folder_name} not empty. Is it ok to overwrite? [Y/n]")
    if answer == "Y" or answer == "y" or answer == "":  # only enter was pressed, thus choosing the default
        overwrite_ok = True
    elif answer == "N" or answer == "n":
        print('Aborting...')
        sys.exit()
    else:
        print("Unexpected input. Input should be either 'y' or 'n'. Aborting")
        sys.exit()
    return overwrite_ok


def get_desired_nr_of_images_per_folder(names):
    nr_per_folder = []
    for name in names:
        folder, _, _ = get_names_for_image_type_folders(name)
        max_nr_of_images = int(get_nr_of_images_in_folder(folder))
        value = input(f"Please enter desired # of images to load for {name} (max {max_nr_of_images}):")
        try:
            check = int(value)
            if check <= max_nr_of_images:
                nr_per_folder.append(check)
            else:
                nr_per_folder.append(max_nr_of_images)
        except Exception:
            raise ValueError

    return nr_per_folder


def get_nr_of_images_in_folder(folder_path):
    nr_of_images = len(get_names_of_images_in_folder(folder_path))
    return nr_of_images


def get_names_of_images_in_folder(folder_path):
    print(folder_path)
    image_names = glob.glob1(folder_path, "*.tif")
    print(image_names)
    return image_names


def get_common_image_nrs_from_image_types(path_folder_bse, path_folder_cl, path_folder_mm=None):
    onlyfiles_cl = get_names_of_images_in_folder(path_folder_cl)
    print('Found', len(onlyfiles_cl), 'files in CL folder:')
    onlyfiles_bse = get_names_of_images_in_folder(path_folder_bse)
    print('Found', len(onlyfiles_bse), 'files in BSE folder:')

    if path_folder_mm: # not None
        onlyfiles_mm = get_names_of_images_in_folder(path_folder_mm)
        print('Found', len(onlyfiles_mm), 'files in mm folder:')

    common_images = []
    for file in onlyfiles_cl:
        if isfile(join(path_folder_bse, file)):
            if path_folder_mm and not isfile(join(path_folder_mm, file)):
                continue
            common_images.append(file)
    return common_images


def get_names_for_image_type_folders(folder):
    print('Opening folder', folder, '..')
    path_folder_cl = folder + '/CL/'
    path_folder_bse = folder + '/BSE/'
    path_folder_mm = folder + '/MM/'
    return path_folder_bse, path_folder_cl, path_folder_mm
