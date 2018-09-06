from NeuralNetHelper.ManagementHelper import Management
from NeuralNetHelper import Settings

if __name__ == "__main__":
    train_model = Management()
    train_model.restore_model('model_checkpoint')
    for i in range(Settings.epoch_size):
        print('Epoch ' + str(i))
        train_model.train_single_epoch()

        print('Doing Validation...')
        train_model.do_validation()
