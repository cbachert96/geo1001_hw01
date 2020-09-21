The file geo1001_hw01.py contains the whole code for the Assignment 01.

The code presents a statistical analysis of 
a heat stress measurement dataset. The data is collected by
five Kestrel 5400 Sensors.
It creates all the plots and statistical information that are asked for in the Assignment.

The paths to the input files are absolute, meaning that they need to be altered to the specific device used.

Each Sensor dataset is assigned to its own Variable in the beginning:
    S1 = Sensor A (HEAT -A_final.xls)
    S2 = Sensor B (HEAT -B_final.xls)
    S3 = Sensor C (HEAT -C_final.xls)
    S4 = Sensor D (HEAT -D_final.xls)
    S5 = Sensor E (HEAT -E_final.xls)
Followingly, if any pre- or suffixes occur, it will always refer to Sensor A. As an example tS1 refers to the Temperature values of Sensor A.

The code is separated in 5 parts, one for each lesson plus the bonus question.

To adapt the bin size of the Histograms, please change the variable "b" in line 69 accordingly.
To get the 5-bins-Histograms as well as the 50-bins-Histograms the code needs to run twice with the changed variable.

The file "95% Confidence Intervals.txt" from A4, will always be created in the current working directory.

The bonus question is only answered in theory, there is no corresponding code to put it into practice.