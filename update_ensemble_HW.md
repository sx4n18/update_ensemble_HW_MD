# updated ensemble HW design

This is the project I started after finishing the main ensemble spiking neural network implementation.

This project aims to optimise mainly the design of the analogue input layer so that the computation could be improved and save power sonsumption with less address switching.

In the meantime, this will only choose 5 nets from the SNN ensemble to save power.

## 3 Nov

Main idea of this project:

Old design loops through the time steps and within that loop it will loop through all the addresss again to yield the same membran voltage change to the neuron at the new time step.

This could be optimised with an expanded voltage memory to save the voltage from last time step and the voltage difference from the first time step.


Now I redesigned the first layers IF neuron to support both multiplication-accumulation and both simply accumulation.

![Simulation waveform of the updated IF neuron](./img/updated_neuron_simulation_waveform_3_Nov.png)

Newly added ports to the neuron module are:
+ mem_vol_diff_2_be_add; 16-bit input acting as the difference of the voltage change
+ arithm;  1-bit input arithmatic selection, 0 as MAC, 1 as ACC. 
+ post_mem_vol_diff; 16-bit output of the voltage change at last time step


But in the meantime, this change once implemented, it will mean that the analogue input layer will act faster than the second layer in the second phase.

To keep the work minimised and the second layer design unchanged, I could either:
+ insert a FIFO
+ slow down the first layer at the second phase


I will slow down the first layer's process.



Another question to be considered is where should I save the voltage change.

According to the simulation, voltage change should be available the same time as the post_mem_vol.

I could make modification on the voltage_mem in the first layer and keep the second voltage mem unchanged.

I will redesign the voltage mem for the first layer with a new module where the width will be doubled and depth stays unchanged.

Each row of memory will be formatted like this:

32'h  AAAA_BBBB where AAAA will be the voltage change BBBB will be the initial voltage.


The newly designed sate machine should be something like this:

```python

for t in range(4):
	if t==0: ## t = 0
		w_i = 0
		for neuron_i in range(40): ## for each neuron
			load_voltage(0) ## load the raw voltage with arithm = 0
			N = load_offset() ## check how many computations it needs
			for i in range(w_i, w_i+N+1, 1):
				index, w = load_weight(i)
				act = load_act(index)
				compute(act, w)
			output_spk_dump_voltage()
			w_i += N
	else:
		for neuron_i in range(40):
			load_voltage(1) ## load the processed voltage with arithm = 1 
			compute(pre_vol, vol_diff)
			output_spk_dump_voltage()

```

## 4 Nov

The newly designed state machine chart should probably look like this:

![New controller state machine](img/newly_designed_controller_statemachine_FSM_chart.jpeg)

The blue part belongs to the old controller state machine, the crimson part is the extra logic I need to add to the state machine.

The only extra port I need to add is simply just the arithm port to the neuron.

Finished the main design and did the simulation. 

So far the functionality looks alright, but the computation looks wrong.

Will verify this tomorrow.


## 6 Nov

Since the state machine has already been setup, I will check the behaviour now.

I will first check the simulation of the old version of the simulation and see the activation for this specific sample.

Just reran the simulation and collected the key data from the simulation, will compare them against the new simulation.

It turns out that I forgot to declare the weight_out_mem as an 8-bit wire so that it was treated as a 1 bit wire.

After fixing this bug, this layer is now giving correct spike AER.


I should check how fast could the second stage be in terms of the spike so that the following layer could catch up.

This would only leave the second layer 6 clock cycles to deal with the spike.

Example simulation shows that it needs:

37, cycles 31 cycles.

These two numbers correspond to the processing unit of 6 and 5.

The clock cycles it needs is 6\*n+2 (one extra cycle for the complete display of the valid signal)

But this would totally depends on how many neurons this spike will trigger.

Normally one spike will trigger 0.3\*18 neurons that is 5.4 neurons.

What I could do probably is to check the distribution of the numbers of neurons triggered by spikes for each spike from the hidden layer.

After checking on the distribution of the count of each possible bin, it could be seen that the possible connection ranges from 1 to 12, so I should at least leave 12 connections for the possible computation.

that is at least 12\*6 +2 = 74 cycles.


