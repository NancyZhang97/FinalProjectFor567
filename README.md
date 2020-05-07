# Final Project For 567

This project is aimed to use deep neural network to predict significant contacts for chromatin regions with the use of corresponding epigenomic information and sequences.

This folder has two models used in this project:
The file "model_auc.py" includes model building and training of model only use epigenomic matrices.
The file "model_auc.py" includes model building and training of model use epigenomic matrices and sequence matrices.

Because the training dataset and saved model files are too big to upload, I just upload codes about model. Before training the model, in order to get input data and targets, you need to do:
1. Get intra-chromosome contact matrix from Hi-C data and fit Weibull distribution to select significant contacts.
2. Randomly select the same number of non-significant contacts as negative outcomes.Label positive outcomes as 1 and negative outcomes as 0.
3. Get each chromatin region pair's epigenomic data array (including DNase-seq data and Chip-seq data for histone modification), standardize them and combine them into a 7-row matrix.
4. Get base sequences for each chromatin region pair and change them into a 4-row matrix.
