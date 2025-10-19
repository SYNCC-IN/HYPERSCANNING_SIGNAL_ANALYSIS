# HYPERSCANNING_SIGNAL_ANALYSIS
A set of tools to analyze multimodal data recorded in hyperscanning experiments of diads in SYNCC-IN project.


In this repo, we develop Python tools to operate and analyze multimodal data, i.e.:
- EEG
- ECG
- IBI - Inter bit intervals
- ET - eye-trackers
  
Currently, the tools are tailored for the experimental setup executed at the University of Warsaw as a part of the SYNCC-IN project.

These experiments consist of three major parts: SECORE, passive MOVIE viewing, and free TALK.
The exemplary processing 'scripts warsaw_pilot_data.py' and 'warsaw_pilot_data_with_ICA.py' require that in your local repo there is a folder 'DATA' containing exemplary diade data 'W_010'

We hope that they can be adapted to the paradigms of other Partners.

[See full specification](docs/data_structure_spec.md)
