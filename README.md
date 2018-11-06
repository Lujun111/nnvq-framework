# Basic Speech Recogniton Pipeline

| Step |  Task  |  Toolkit  |  
|  :--:  |  :--:  |  :--:  |  
| 1 | Raw-Data e.g. TEDLium | kaldi |
| 2 | Feature creation (MFCC, PLP, etc.) | kaldi |
| 3 | Preparing Data for tensorflow | python + tensorflow |
| 4 | Building Acoustic Model | tensorflow |
| 5 | Training Acoustic Model | tensorflow |
| 6 | Decoding | kaldi |
| 7 | Evaluation | kaldi |

# Define a new Task

- Change settings in NeuralNetHelper/Settings.py