# updated ensemble HW design

This is the project I started after finishing the main ensemble spiking neural network implementation.

This project aims to optimise mainly the design of the analogue input layer so that the computation could be improved and save power sonsumption with less address switching.

In the meantime, this will only choose 5 nets from the SNN ensemble to save power.

## 3 Nov

Main idea of this project:

Old design loops through the time steps and within that loop it will loop through all the addresss again to yield the same membran voltage change to the neuron at the new time step.

This could be optimised with an expanded voltage memory to save the voltage from last time step and the voltage difference from the first time step.


