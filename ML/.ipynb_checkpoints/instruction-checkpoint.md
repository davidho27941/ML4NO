input_state == 0 for ve_dune
            == 1 for all_dune (ve_dune, vu_dune, vebar_dune, vubar_dune)
            == 2 for all('ve_dune', 'vu_dune', 'vebar_dune', 'vubar_dune', 've_t2hk', 'vu_t2hk', 'vebar_t2hk', 'vubar_t2hk')

Regression:

Feature -- to get understanding the features of original data
models -- to check the models created (the siffix represent the data used to train the model)
ak-delta -- build ak model on delta
ak-theta23 -- build ak model on theta23
ak -- build ak model on multiple parameters together
stat -- generate poisson bins and get statistics

Classification:

ak -- build ak model on all 12 classes (octant = {-1, 0, 1}, cpv = {0, 1}, mo = {-1, 1})
stat -- generate poisson bins and get statistics