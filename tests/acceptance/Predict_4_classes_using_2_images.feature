# Created by ah at 08.12.22
Feature: Predict_4_classes_using_2_images
  # Enter feature description here

  Scenario: Predicted image from BSE and CL
    Given I have a BSE image and a CL image
    And I have a model
    When I use "predict_from_images"
    Then I have a predicted image