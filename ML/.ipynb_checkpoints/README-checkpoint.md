# Data
Store on gpu 140.114.94.240 /home/ko-yan/ML4NO/Data -- Select data from all channels
* n1000000_0910_all_flat.npz -- fixed theta12 and sdm, flat on theta13, theta23, delta (the preferred dataset for regression)
* n1000000_0910_classification.npz -- fixed theta12 and sdm, flat on theta13, theta23, delta has peak on 0 and 180  (the preferred dataset for classifying cpc and cpv)

* sample_NuFit0911.npz -- Index 0 for IO spectrum and index 1 for NO spectrum.

# Features files
A python notebook to inspect the features of the input data.
* The name of the features
* The histogram of the feature
* The correlation matrix of the ve_dune

# Others
* Poisson_Compare_Train.ipynb -- Compare the poisson generated with the Train spectrum

# Classification Folder

* ak -- build ak model on all 12 classes (octant = {-1, 0, 1}, cpv = {0, 1}, mo = {-1, 1})
* stat -- generate poisson bins and get statistics

# Regression Folder

* ak-delta-combine.ipynb -- Combination of ak-delta, ak-delta-furthurTrain, and ak-delta-poisson.
* ak-delta-furthurTrain.ipynb -- Input trained deltacp model and train with perturbation.
* ak-delta-poisson.ipynb -- Train deltacp model with perturbation directly.
* ak-delta.ipynb -- Most common training deltacp model.
* sigmoid-delta-poisson.ipynb --  Hand written structure to make the last layer sigmoid.


* ak-theta23-combine.ipynb -- Combination of ak-theta23, ak-theta23-furthurTrain, and ak-theta23-poisson.
* ak-theta23-furthurTrain.ipynb -- Input trained theta23 model and train with perturbation.
* ak-theta23-poisson.ipynb -- Train theta23 model with perturbation directly.
* ak-theta23.ipynb -- Most common training theta23 model.


* models.ipynb -- to check the models created (the siffix represent the data used to train the model)
* contour.ipynb -- to generate 1 and 2 sigma contour
* stats.ipynb -- generate poisson bins and get statistics

## models
usedDataset-parameter-number

## models_furthurTrain
usedDataset-parameter-(origin_model_number)-number
usedDataset-parameter-(origin_model_number)-number_result -- the furthurTrain loss during the training process

## models_PoissonTrain
usedDataset-parameter-(origin_model_number)-(furthurTrain_number)-number