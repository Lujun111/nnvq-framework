from NeuralNetHelper.ManagementHelper import Management
from NeuralNetHelper import Settings

if __name__ == "__main__":
    train_model = Management()
    train_model.restore_model('model_checkpoint')

    if not Settings.inference:
        print('Training model...')
        for i in range(Settings.epoch_size):
            print('Epoch ' + str(i))
            train_model.train_single_epoch()

            print('Create P(s_k|m_j) from training data...')
            train_model.create_p_s_m()

            print('Doing Validation...')
            train_model.do_validation()
    else:
        print('Doing inference...')
        train_model.do_inference()
