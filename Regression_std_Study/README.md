# Regression std Study
Aim to study the detail of gradual learning.   
Currently, we facus on the DUNE experiment.

---
## Script Description:

* std_study_full_energy.py
    * energy range: full specturm 
    * input: <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{e}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{e}}" title="\nu_{\bar{e}}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\mu}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\mu}" title="\nu_{\mu}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{\mu}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{\mu}}" title="\nu_{\bar{\mu}}" /></a> 
    * model: one model is for both of parameters
    * training precedure: asimov -> add gaussian noise with std=0.001 -> add gaussian noise with std=0.001215 -> ...... -> add gaussian noise with std=2.000 -> poisson(X10)

* std_study_to_5GeV.py
    * energy range: up to 5 GeV 
    * input: <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{e}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{e}}" title="\nu_{\bar{e}}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\mu}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\mu}" title="\nu_{\mu}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{\mu}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{\mu}}" title="\nu_{\bar{\mu}}" /></a> 
    * model: there are two kinds of model in the scripts. One is for $\theta_{23}$, the other is for $\delta$.
    * training precedure: asimov -> add gaussian noise with std=0.001 -> add gaussian noise with std=0.001215 -> ...... -> add gaussian noise with std=2.000 -> poisson(X10)
    
    
* std_study_to_5GeV_e_only.py
    * energy range: up to 5 GeV 
    * input: <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{e}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{e}}" title="\nu_{\bar{e}}" /></a> 
    * model: one model is for both of parameters
    * training precedure: asimov -> add gaussian noise with std=0.001 -> add gaussian noise with std=0.001215 -> ...... -> add gaussian noise with std=2.000 -> poisson(X10)
    
* poisson_loop_study_full_energy.py
    * energy range: full specturm 
    * input: <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{e}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{e}}" title="\nu_{\bar{e}}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\mu}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\mu}" title="\nu_{\mu}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{\mu}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{\mu}}" title="\nu_{\bar{\mu}}" /></a> 
    * model: one model is for both of parameters
    * training precedure: using previous trained model(after adding gaussian noise with std=0.92) and then trained with poisson noise from once to 20 times
    
* poisson_loop_study_to_5GeV.py
    * energy range: up to 5 GeV  
    * input: <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{e}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{e}}" title="\nu_{\bar{e}}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\mu}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\mu}" title="\nu_{\mu}" /></a>   <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{\mu}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{\mu}}" title="\nu_{\bar{\mu}}" /></a> 
    * model: one model is for both of parameters
    * training precedure: using previous trained model(after adding gaussian noise with std=0.92) and then trained with poisson noise from once to 20 times
    
    
* std_study_to_5GeV_new_model.py
    * energy range: up to 5 GeV 
    * input: [<a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{e}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{e}}" title="\nu_{\bar{e}}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\mu}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\mu}" title="\nu_{\mu}" /></a> , <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{\mu}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{\mu}}" title="\nu_{\bar{\mu}}" /></a>] & [<a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a> - <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{e}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{e}}" title="\nu_{\bar{e}}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\mu}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\mu}" title="\nu_{\mu}" /></a> - <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{\bar{\mu}}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{\bar{\mu}}" title="\nu_{\bar{\mu}}" /></a>]
    * model: there are two kinds of model in the scripts. One is for $\theta_{23}$, the other is for $\delta$.
    * training precedure: asimov -> add gaussian noise with std=0.001 -> add gaussian noise with std=0.001215 -> ...... -> add gaussian noise with std=2.000 -> poisson(X10)
    
    
## Usage:
```
python3 xxxxxxx.py experiment physics_parameter
```
    * xxxxxxx.py: the above scripts above
    * experiment: DUNE or T2HK
    * physics_parameter: delta or theta23
    
e.g.
```
python3 std_study_full_energy.py dune delta
```

## Results:
All results are in corresponding notebooks.

e.g.   
The result of `std_study_full_energy.py` is in `contour_full_energy.ipynb`.



