import numpy as np


def get_percentage_values_for_labels(iSample, im_name, percentage_table, predictions_path, test_argmax):
    total_area = test_argmax.shape[0] * test_argmax.shape[1]
    percentage_table['path'][iSample] = predictions_path + '/' + im_name
    percentage_table['quartz_rel_area'][iSample] = np.round(np.count_nonzero(test_argmax == 4) / total_area * 100, 2)
    percentage_table['otherminerals_rel_area'][iSample] = np.round(
        np.count_nonzero(test_argmax == 3) / total_area * 100, 2)
    percentage_table['overgrowth_rel_area'][iSample] = np.round(np.count_nonzero(test_argmax == 2) / total_area * 100,
                                                                2)
    percentage_table['pores_rel_area'][iSample] = np.round(np.count_nonzero(test_argmax == 1) / total_area * 100, 2)
    print('Done and saved!')
    return percentage_table