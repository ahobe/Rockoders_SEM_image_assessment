# Created by ah at 08.12.22
Feature: Predict_4_classes_using_2_images
  # Enter feature description here

  Scenario: Predicted image from BSE and CL
    Given I have a BSE image and a CL image
    And I have a model
    When I use "predict_from_images" and "get_maximum_likelihood_label_for_each_pixel"
    Then I get a predicted image
    And the predicted image has 4 classes

#    Scenario: Reconstruct full image after using nxn unet
#      Given I have a Unet
#      And I have a BSE image and a CL image
#      Then when I use

#  Scenario: Quantify 4 classes for predicted image
#    Given I have a predicted image
#    When I use "get_percentage_values_for_labels"
#    Then I get percentage values for all 4 labels