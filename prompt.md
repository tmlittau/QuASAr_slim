This is not working. Now when I'm running the benchmarks it takes an eternity. 

Since I have to generate the result plots for my paper, I want to simply create a slimmed down version of the examples I want to give.

So instead of using the at this point giant codebase, let's try to come up with a very slim version of QuASAr, starting with just taking in a circuit generated from a file benchmark_circuits.py, similar to what we have in QuASAr. 
Then we can build QuASAr that should be structured as follows:
QuASAr/
-- analyzer.py
-- planner.py
-- SSD.py
-- backends/
    -- tableau.py
    -- dd.py
    -- sv.py
-- conversion/
    -- tab2sv.py
    -- tab2dd.py
    -- dd2sv.py

This should be enough for now. 

## Analyzer
So the idea would be that the analyzer identifies independent sub circuits in the provided circuit. You can use any available libraries that already exist here, maybe qiskit provides a quick way to identify those using the dependency graph, check the documentation. Otherwise create your own algorithm based on the existing QuASAr implementation to do this. 
Besides this, the analyzer should also record the circuit metrics (either it total or for each subcircuit), depending on the incoming gates. Circuit metrics means the metrics that later decide which method to use (Clifford-gates, gate-rotation-metrics etc.).

## Planner
The planner is supposed to take the metrics and assign the final methods by estimating the cost for feasible partitioning plans and picking the best. It creates the final execution-ready annotated SSD.

## SSD
This is out hierarchical data structure that on the top level provides the partitioning of the circuit where each node represents one partition with the simulation method assigned, holding the metrics that were 'filled in' by the analyzer and all other meta data like qubit subsystem and fingerprint. The next level would be the gates or sub circuits within each node. 

## Backends
Here should be the helper classes for each "plugged in" simulator. 
stim for Tableau / stabilizer
MQT DD for Decision Diagram simulation
Qiskit Aer for statevector

## conversion
Here should be helper classes to convert from one representation to the other. For now, we simply use the library internal functions to convert to statevector. In case of tab2dd (tableau to decision diagram) for now we go the unfortunate route of converting to statevector first and then to decision diagram. If you can find a way to convert directly from tableau to decision diagram, that would be ideal, have a look online.


Please go ahead and provide the necessary files for this implementation such that I can run this slimmed down version of QuASAr to get results in time for the paper deadline.
