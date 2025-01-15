==============
SEMART DATASET
==============

This dataset contains painting images associated with attributes and comments for semantic art understanding. Data is collected from the Web Gallery of Art (https://www.wga.hu/).


FILES & DIRECTORIES
===================
- Images/		JPG files with painting images
- human_difficult.csv	Human evaluation difficult task
- human_easy.csv	Human evaluation easy task
- semart_test.csv	Test split with 1069 samples
- semart_train.csv	Train split with 19244 samples
- semart_val.csv	Validation split with 1069 samples


DATA FILES
==========
semart_{split}.csv files contain the information of each data split (test/train/val). Each row in the file (tab separated) contains the following elements:
- IMAGE_FILE		filename of the image in the Images/ directory
- DESCRIPTION		artistic comment
- AUTHOR		author
- TITLE			title 
- TECHNIQUE		technique and format
- DATE			date when the painting was created
- TYPE			genre
- SCHOOL		nationality
- TIMEFRAME		50 year time period


HUMAN EVALUATION FILES
======================
human_{level}.csv files contain the information of the data used in the human evaluation experiments. Each row in the file (coma separated) continas the following elements:
- id 			experiment id
- author		displayed author
- type			displayed type
- school		displayed school
- timeframe		displayed timeframe
- description		displayed description
- image_1_url		filename of image no.1
- image_2_url		filename of image no.2
- image_3_url		filename of image no.3
- image_4_url		filename of image no.4
- image_5_url		filename of image no.5
- image_6_url		filename of image no.6
- image_7_url		filename of image no.7
- image_8_url		filename of image no.8
- image_9_url		filename of image no.9
- image_10_url		filename of image no.10
- gt			id of the correct image


CITATION
========
If you use this dataset, please cite the following paper:

@InProceedings{Garcia2018How,
author    = {Noa Garcia and George Vogiatzis},
title     = {How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval},
booktitle = {Proceedings of the European Conference in Computer Vision Workshops},
year      = {2018},
}
