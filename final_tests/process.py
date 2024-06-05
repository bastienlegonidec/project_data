from feature_extraction import process_images_to_csv
from model import process_bug_data

# Extract features from images and save to CSV
process_images_to_csv("project_data/test/images/", "project_data/test/masks/", "project_data/final_tests/final_feature.csv")
#process_bug_data('project_data/dataVisualization/result.csv', "project_data/final_tests/final_feature.csv", "project_data/final_tests/final_output.csv")

