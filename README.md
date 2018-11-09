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
- Before you can create the data one must ensure that a basic model (mono/triphone)
is already trained.
- Install the package *tensorflow*/*tensorflow-gpu* in python
## Start process
- Open the script *scripts/generate_tf_data.sh*
- *kaldi_path*: Set your kaldi root folder of the dataset
- *framework_path*: Set your path to the nnvq-framework in here
- If you are using a virtual environment for python, change the source folder:
    ```
    source ~/folder_to_your_virtual_environment/bin/activate
    ```otherwise comment out this line

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