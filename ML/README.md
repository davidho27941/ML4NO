# Data
Store on gpu 140.114.94.240
* n1000000_0716.npz -- gaussian on theta12, theta13, theta23, and sdm, deltacp half on 0 and 180  
* n1000000_0728.npz -- flat on theta12, theta13, sdm, theta23 half on 45 and else flat, deltacp half on 0 and 180
* n1000000_0803.npz -- fixed theta12 and sdm, flat on theta13, theta23 half on 45 and else flat, deltacp half on 0 and 180 (the preferred dataset for classification)
* n1000000_0804_all_flat.npz -- fixed theta12 and sdm, flat on theta13, theta23, delta (the preferred dataset for regression)

* NuFit_IO.npz -- the spectrum created by NuFit inverse ordering parameters
* NuFit_NO.npz -- the spectrum created by NuFit normal ordering parameters

# Feature files
A python notebook to inspect the features of the input data.
* The name of the features
* The histogram of the feature
* The correlation matrix of the ve_dune


* Poisson_Compare_Train.ipynb -- Compare the poisson generated with the Train spectrum

# Classification Folder

* ak -- build ak model on all 12 classes (octant = {-1, 0, 1}, cpv = {0, 1}, mo = {-1, 1})
* stat -- generate poisson bins and get statistics

# Regression Folder

* ak-delta-furthurTrain.ipynb -- Input trained deltacp model and train with perturbation.
* ak-delta-poisson.ipynb -- Train deltacp model with perturbation directly.
* ak-delta-testcut.ipynb -- Train deltacp model with selected inputs.
* ak-delta.ipynb -- Most common training deltacp model.
* sigmoid-delta-poisson.ipynb --  Hand written structure to make the last layer sigmoid.


* ak-theta23-furthurTrain.ipynb -- Input trained theta23 model and train with perturbation.
* ak-theta23-poisson.ipynb -- Train theta23 model with perturbation directly.
* ak-theta23-testcut.ipynb -- Train theta23 model with selected inputs.
* ak-theta23.ipynb -- Most common training theta23 model.


* models.ipynb -- to check the models created (the siffix represent the data used to train the model)
* contour.ipynb -- to generate 1 and 2 sigma contour
* stats.ipynb -- generate poisson bins and get statistics

## models_furthurTrain
usedDataset-parameter-number
generateProcess.md

## models_all
usedDataset-parameter-number

# Planning
* Merge delta training files.
* Delete unusable models.