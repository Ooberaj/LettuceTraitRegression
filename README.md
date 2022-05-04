# LettuceTraitRegression

Given an RGB-D image of a lettuce head, the model predicts its fresh weight, dry weight, height, diameter, and leaf area. The model is trained on the [2021 Autonomous Greenhouse Challenge dataset]
(https://data.4tu.nl/articles/dataset/3rd_Autonomous_Greenhouse_Challenge_Online_Challenge_Lettuce_Images/15023088#!).

After testing a variety of Early Fusion, Mid Fusion, and Late Fusion architectures, a Mid Fusion architecture with subnetworks for each output performed the best under my limited compuatational resources(Google Colab), acheiving an NMSE loss of 0.114. Each architecture tested can be found in `models.py`. 

The top-performing architecture consists of a subnetwork for each output. Each subnetwork consists of a separate ResNet18 for RGB and Depth images that both feed into a single fully connected network, producing a single output.
