# Solution to MATLAB and Simulink Challenge project 245 Classifying RF signals using AI

[Program link](https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub)

[Project description link](<https://github.com/mathworks/MATLAB-Simulink-Challenge-Project-Hub/blob/main/projects/Classify%20RF%20Signals%20Using%20AI/README.md>)


# Project details
To address the challenges of an increasingly congested electromagnetic spectrum where traditional static radios struggle, this project establishes a real-time signal classification framework using an NI USRP B210 Software-Defined Radio (SDR) and a Support Vector Machine (SVM) classifier implemented via MATLAB and Simulink. The development is split into an offline training phase and a real-time deployment phase. During training, a diverse database of complex baseband IQ samples is generated across multiple modulation types under varying channel impairments like noise, fading, and interference. These raw inputs are preprocessed through Automatic Gain Control (AGC), DC offset removal, and filtering before extracting statistical features (such as mean, variance, skewness, and kurtosis), higher-order moments, or spectrograms to construct a specialized feature matrix that trains a supervised SVM algorithm to define optimal high-dimensional decision boundaries.

In the deployment and testing phase, target waveforms including Sine Waves, AM, FM, BPSK, and QPSK are generated in Simulink, transmitted via a USRP B210 (Tx) hardware node over an impaired wireless medium, and captured using a USRP B210 (Rx) receiver module. Alternatively, the live system seamlessly connects to the interactive RFClassify web interface, which serves as a dedicated deployment dashboard. Through this interface, unknown captured IQ data arrays can be pasted into the application to run instant classification against the loaded model boundaries. The system immediately extracts the necessary feature characteristics and pipes them through the pre-trained SVM hyperplane, dynamically predicting, displaying, and logging the identified modulation scheme alongside its real-time confidence percentage.


# How to run section
To initialize the workflow, open the RFClassify dashboard application. If the status indicator in the top-right corner reads ● no model - train first, navigate to the 01 / Train tab to load your training feature matrix and initialize your SVM classifier workspace.
Next, prepare your IQ values by running your signal simulation or transceiver model in MATLAB to capture baseband complex IQ data samples. Convert the workspace signal array into a format accepted by the app's clipboard parser—ensuring a minimum of 8 IQ pairs for valid inference—by running the following command in your MATLAB command window to generate a standard column file.
writematrix([real(sig)' imag(sig)'], 'iq.csv')
Finally, to run the classification, click on the 02 / Classify tab and paste your copied MATLAB IQ values directly into the text box inside the IQ Input card. The system flexibly accepts complex notation (0.707+0.707i), line-by-line pairs (0.707 0.707), or matrix row-vectors ([0.707 0.707; -0.707 -0.707]). Alternatively, you can use the quick-load sample preset buttons (BPSK, QPSK, 16-QAM, 64-QAM) and click the teal Classify button to execute the inference model, or click Clear to reset the field.<img width="1334" height="543" alt="Screenshot 2026-05-30 011355" src="https://github.com/user-attachments/assets/5b86dd26-7389-4049-b8e6-e11c6adc5ebf" />
<img width="1322" height="454" alt="Screenshot 2026-05-30 011427" src="https://github.com/user-attachments/assets/78de4f25-4cbb-49ef-a2d4-71354f2962f0" />







# Demo/Results
The RFClassify web dashboard provides an interactive workspace for validating real-time signal recognition, accommodating testing for both baseline and higher-order digital modulations including BPSK, QPSK, 16-QAM, and 64-QAM. Within the user interface, the Status Banner in the top-right corner dynamically tracks model readiness, updating from ● no model - train first to an active state once a workspace framework is initialized. To verify a waveform, the IQ Input Panel ingests complex multi-row array text blocks formatted directly from standard MATLAB workspace matrix conversions using the writematrix function, or via quick-load sample presets. Upon clicking the teal "Classify" button, the backend evaluates the pasted coordinates against the trained Support Vector Machine (SVM) hyperplane, instantly updating the right-hand Result Panel Display from its idle state ("Paste IQ values and click Classify") to showcase the finalized text prediction string detailing the identified modulation scheme.
<img width="1263" height="742" alt="image" src="https://github.com/user-attachments/assets/f86ea11a-4102-4226-b217-0a29d5d7c7e2" />
<img width="1298" height="865" alt="image" src="https://github.com/user-attachments/assets/fe24a0e8-aecc-403d-b5b5-a24726716f2c" />
<img width="1332" height="928" alt="image" src="https://github.com/user-attachments/assets/d5556e19-5660-4b3c-ab04-cc2eed64bcf6" />
<img width="1236" height="915" alt="image" src="https://github.com/user-attachments/assets/d7f17629-25ac-451d-9929-a49d4e7175b4" />




  
# Reference
Christopher Gravelle and Ruolin Zhou, "SDR Demonstration of Signal Classification in Real-Time using Deep Learning".
Daniil Stafeev and Mikhail Ronkin, "Data Collection for Classification of Radio Signals Using SDR Transceivers".
Shadman Rahman Doha and Ahmed Abdelhadi, "Artificial Intelligence in Software Defined Radio: A Survey".

Training data: RadioML 2016.10A (DeepSig), 600 samples/class, SNR ≥ 0dB
Source: https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom
