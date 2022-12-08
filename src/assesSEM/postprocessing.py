import numpy as np


def get_percentage_values_for_labels(im_name, percentage_table_row, predictions_path, label_predictions):
    total_area = label_predictions.shape[0] * label_predictions.shape[1]    # is this needed for every image?
    percentage_table_row['path'] = predictions_path + '/' + im_name
    percentage_table_row['quartz_rel_area'] = np.round(np.count_nonzero(label_predictions == 4) / total_area * 100, 2)
    percentage_table_row['otherminerals_rel_area'] = np.round(np.count_nonzero(label_predictions == 3) /
                                                              total_area * 100, 2)
    percentage_table_row['overgrowth_rel_area'] = np.round(np.count_nonzero(label_predictions == 2) / total_area * 100, 2)
    percentage_table_row['pores_rel_area'] = np.round(np.count_nonzero(label_predictions == 1) / total_area * 100, 2)
    print('Done and saved!')
    return percentage_table_row


def get_maximum_likelihood_label_for_each_pixel(predictions_for_all_labels):
    max_likelihood_image = np.argmax(predictions_for_all_labels, axis=2)
    return max_likelihood_image