from NeuralNetHelper.ManagementHelper import Management
from NeuralNetHelper import Settings

if __name__ == "__main__":
    train_model = Management()
    train_model.restore_model('model_checkpoint')

    if not Settings.inference:
        print('Training model...')
        for i in range(Settings.epoch_size):
            print('Epoch ' + str(i))

            print('Training base network')
            train_model.train_single_epoch(identifier='combination')

            # print('Creating P(s_k|m_j)...')
            # train_model.create_p_s_m()
            #
            # # print('Training output layer')
            # # train_model.train_single_epoch(train_last_layer=True)
            #
            print('Doing Validation...')
            train_model.do_validation()

    else:
        print('Doing inference...')
        train_model.do_inference()
