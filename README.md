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

# Generate train/dev/test data for tensorflow
## Prerequisite
Before you can create the data one must ensure that a basic model (mono/triphone)
is already trained.
## generate_tf_data.sh

# Define a new Task

- Change settings in NeuralNetHelper/Settings.py:
    - path to data (train/dev/test)
    - path for tensorboard/checkpoint
    - apply Hyperparameter
    - Very important: Define Identifier!

- Change Model in NeuralNetHelper/ModelHelper.py:
    - create Model
    - build model by using your predefined Identifier

- Change Loss in NeuralNetHelper/LossHelper.py:
    - create Loss
    - select Loss by checking for Identifier

- Change train_dict in NeuralNetHelper/TrainHelper.py:
    - Add a train_dict for the Identifier
    - The train_dict is executed during training and returns the result of the 
    train_op

- Change Saver in NeuralNetHelper/SaverHelper.py:
    - Define your metric for saving a better model weights
    - Use Identifier for this

- Change Summary in NeuralNetHelper/SummaryHelper.py:
    - Depending on your Identifier add the logging data which should be
    added to tensorboard

# TODOs
- Improve Inference