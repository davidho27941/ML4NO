# Regression std Study
Aim to study the detail of gradual learning. 
Currently, we facus on the DUNE experiment.

---
## Script Description:

* std_study_full_energy.py
    * energy range: full specturm 
    * neutrino type: $\nu_e$ + $\nu_\bar{e}$ + $\nu_\mu$ + $\nu_\bar{\mu}$ <a href="https://www.codecogs.com/eqnedit.php?latex=\nu_{e}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nu_{e}" title="\nu_{e}" /></a>
    * model: one model is for both of parameters
    * training precedure: asimov -> add gaussian noise with std=0.001 -> add gaussian noise with std=0.001215 -> ...... -> add gaussian noise with std=2.000 -> poisson(X20)

* std_study_to_5GeV.py
    * energy range: up to 5 GeV 
    * neutrino type: $\nu_e$ + $\nu_\bar{e}$ + $\nu_\mu$ + $\nu_\bar{\mu}$
    * model: there are two kinds of model in the scripts. One is for $\theta_{23}$, the other is for $\delta$.
    * training precedure: asimov -> add gaussian noise with std=0.001 -> add gaussian noise with std=0.001215 -> ...... -> add gaussian noise with std=2.000 -> poisson(X20)
    
    
    
* std_study_to_5GeV_e_only.py
    * energy range: up to 5 GeV 
    * neutrino type: $\nu_e$ + $\nu_\bar{e}$
    * model: one model is for both of parameters
    * training precedure: asimov -> add gaussian noise with std=0.001 -> add gaussian noise with std=0.001215 -> ...... -> add gaussian noise with std=2.000 -> poisson(X20)
    
* poisson_loop_study_full_energy.py
    * energy range: full specturm 
    * neutrino type: $\nu_e$ + $\nu_\bar{e}$ + $\nu_\mu$ + $\nu_\bar{\mu}$
    * model: one model is for both of parameters
    * training precedure: using previous trained model(after adding gaussian noise with std=0.92) and then trained with poisson noise from once to 20 times
    
* poisson_loop_study_to_5GeV.py
    * energy range: up to 5 GeV  
    * neutrino type: $\nu_e$ + $\nu_\bar{e}$ + $\nu_\mu$ + $\nu_\bar{\mu}$
    * model: one model is for both of parameters
    * training precedure: using previous trained model(after adding gaussian noise with std=0.92) and then trained with poisson noise from once to 20 times
    
    
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


