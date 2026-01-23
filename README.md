# TritonNeuroTech-EMG-Prosthetics

Project Details:


EMG Prosthetic Arm
A wearable prosthetic arm that can be controlled using EMG signals.  The system should detect different muscle activations (e.g. hand open, fist, wrist flex) and map them to prosthetic movements. A manual activation button or gesture could “enable” control mode to prevent accidental triggers.


Data Acquisition & Pre-processing (Filtering)
MindRove Connect
LibEMG (https://www.libemg.com/)
Band Pass filtering + Low and high pass filters
Ex: Apply a notch filter at 60 Hz and a bandpass filter from 10 Hz to 200 Hz.
MVC(Calibration)


Gesture Recognition & Mapping (Classification)
RNN (LSTM)
TensorFlow
GWO: Grey Wolf Optimization (metaheuristic algorithm)
Once we get results from our ML model we need to set a threshold, run some tests, determine true positive rate and false positive rate, and revise as needed


Actuation, Safety & Communication
Raspberry Pi 5 and/or Arduino


Prosthetics Movement 
Linear actuator/spring system from last year for springs 
Ball joint or hinge joint for wrist 


Material 
PLA print for palm, arm base, fingers 
Hollow arm + bar for gripping during the demo 
Silicone cushion inside the arm for wearability 
Metal constraints at wrist, so it can hold weight 


Design 
Fusion for structure 
Maybe model with Ansys? (bit of a stretch but can look into it) 


# Collaborators:
- Enrique Aranda - earanda4
- Skye Belcher (skyebel)
- Peter Little (PeterLittle670)
- Byron Chen - HashbrownPNG
- Ziqing Zhu (zikoole)
- Bora Vanli (bvanli)
- Dhruv Sehgal - dhrutube
- Shivani Rajanala - Shiv-Code123
- Abhay Korlapati - abhay784
- Siddhant Gulati (Sid6154)
- Rishab Kolan - rkolan-alt
- Tristan Lee (tmlee06)




# Links
[TNT - Basic Content Guide
](https://docs.google.com/document/d/1KJiq6qIKDhxJ-azZowQby1tJ9o-k_Vsh72BCCRTu_YU/edit?tab=t.0)

[TNT - Members Managing Document
](https://docs.google.com/document/d/1dWdjA4xGRYG7SnRga2GyC9wFmkC64npIvnQQYizD2NQ/edit?tab=t.0)

[TNT - Programming Planning Document](https://docs.google.com/document/d/1JNd9rQi58S-O2Ss6KbTQ3FcP-HhZHSv_BTzB1oUUAZE/edit?tab=t.0)
