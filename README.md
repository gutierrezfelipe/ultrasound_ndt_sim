# Ultrasound NDT Simulator

Examples demonstrating ultrasound NDT simulation with Perfectly Matched Layers in GPU implementation 
and some of the code for the tests and results of the paper.


# Instructions

First compile the CUDA code with the makefile. The "cuda_interface*" files are the interfaces between 
the python code and the CUDA functions and must be included in any script that will run simulations.

Then you should be able to run the example codes:
1. reflection_comparison.py
2. domain_without_PML.py
3. domain_with_PML.py
4. TT_Times.py
5. glycerol_setup.py

The "Analyses.py" is useful to visualize the recorded data if you save them.

The "glycerol_setup.py" is set to simulate one test we did in our lab. The recorded data from 6 emitters are on the 
"data" folder with a simulated version too. This is still a work in progress.

The "data" folder contains some other files, as the recorded pulse of our Olympus Ultrasound transducer we used as 
source in the above mentioned file and the recorded acquisition from a test without any reflectors from the same 
6 emitters and other versions of the transducer pulse.

If the video output is desired you must specify it in the initialization of the cuda_interface class, the finished 
video will be in the output folder.
