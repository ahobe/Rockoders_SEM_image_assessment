import os
import sys


def get_folder_names():
    value = input("Please choose folders:\n" +
                  "0: All folders\n" +
                  "1: dataset1\n" +
                  "2: dataset2\n" +
                  "3: dataset3\n" +
                  "4: dataset4"
                  )
    if value == "0":
        names = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
    elif value == "1":
        names = ['dataset1']
    elif value == "2":
        names = ['dataset2']
    elif value == "3":
        names = ['dataset3']
    elif value == "4":
        names = ['dataset4']
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
        overwrite_ok = get_ok_for_overwrite()

    if overwrite_ok or empty_folder:  # either or both True
        allowed_to_continue = True
        return allowed_to_continue


def get_ok_for_overwrite():
    answer = input("Folder not empty. Is it ok to overwrite? [Y/n]")
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
        value = input(f"Please enter desired # of images to load for {name}:")
        # get total nr of images in folder
        # todo: add total nr for consideration and error handling.
        nr_per_folder.append(int(value))

    return nr_per_folder
