# VibrationPattern
Software and hardware for the onboard point-wise vibration pattern production
(For the manuscript "Point-Wise Vibration Pattern Production via a Sparse Actuator Array for Surface Tactile Feedback" submitted to ICRA 2024.)

hardware: voltage drive circuit board of the sparse actuator array. 

softeare: 
(1) datasets: run each python script in this folder firstly to generate the basic vibration pattern series (need COMSOL 6.0).
(2) algorithm_SA: the whole simulated annealing process for search the optimal phase spectrum (need the server memory no less than 100G).
(3) algorithm_SA_target_4: set any target feedback point to get its corresponding optimal phase spectrum (can change the target point in datasets/pattern_target_4).
