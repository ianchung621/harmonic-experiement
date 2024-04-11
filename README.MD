# DATA STORAGE

1. **folder name (SampleName_ExperimentOrder)**

* the folder name clarifies the thickness of pt, sample number, experiment order.
* example: NiPS3_100nm_5_2 means 100nm Pt, sample 5, experiment 2
2. **file name (expID_xK_xT_xmA_FC_CC.csv)**

* the file name clarifies the experiment id, tempurature, field, current, condition
* the string FC,ZFC,CC,ZCC means field cool, zero field cool, current cool, zero current cool (current cool: apply current during cooling process)
* if not specified 2nd, it is first harmonic
* example: exp1_20K_3T_5mA_FC_CC.csv means exp1, measure under 20 Kelvin, 3 Tesla, 5mA AC current, field cool, current cool, first harmonic
3. **file content**

* columns: external field angle, amplitude, amplitude error, phase, phase error, *voltage history
* for first harmonic, the actual amplitude should be multiply by 10^-6 to convert to Volt (e.g. 2.2 --> 2.2*10^-6 V)
* for second harmonic, the actual amplitude should be multiply by 10^-8 to convert to Volt (e.g. 2.2 --> 2.2*10^-8 V)

# EXPERIMENT SET UP

1. **preset**

* file manager -> E -> maurice(保留) -> QDInstrument_LabVIEW_1.0.4_Pharos_20131010 -> open QDInstrument_server.exe
* windows start -> NI-MAX -> GPIB-USB-HS 'gpib0' -> scan instrument (make sure 6221, SR830, PPMS, PPMS option box is connected)
* MultiView -> instrument -> bridge config: only channel 4 is opened, connect sample to external circuit

2. **6221**

* make sure to test on standard resistant before passing current to sample
* file manager -> E -> DATA -> SF_Lee -> Daniel -> open 6221_current_source.vi

3. **SR830**

* suggest params: Time constant 1s, Slope/Oct 24dB Ground float, Input A, Coupling AC, Trig neg_edge, Reserve low_noise
* use expand 1 for 1st, expand 100 for 2nd

4. **PPMS sample installation**

* install rotator (O-ring no need)
* install rotator rod, make sure its at 0 degree 
* MultiView -> instrument -> motion: make sure rotator is at 0 degree
* connect rotator and rotater rod, purge

5. **measurement**

* file manager -> E -> DATA -> SF_Lee -> Daniel -> posision_scan -> posscan.vi
* MultiView -> instrument -> motion: make sure rotator is at 0 degree
* set the angle table file and data target file. you can directly type out the data target filename, it will create csv file if it is not exist (but will not create folder)
* set the waiting time, sampling time, sampling num from block diagram (shortcut: ctrl+E)
* suggested folder and filename: folder: SampleName_ExperimentOrder file: expID_xK_xT_xmA_FC_CC.csv