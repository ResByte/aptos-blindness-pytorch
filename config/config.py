
def config():
    cfg = {
        # raw csv data
        'train_csv_path': '/workspace/kaggle/aptos-data/train.csv',
        'test_csv_path': '/workspace/kaggle/aptos-data/test.csv',
        # images path
        'img_root': '/workspace/kaggle/aptos-data/train_images/',
        'test_img_root': '/workspace/kaggle/aptos-data/test_images/',
        'arch': 'resnet50',
        'batch_size': 16,
        'test_batch_size': 4,
        'num_classes': 5,
        'random_state': 123,
        'test_size': 0.2,
        'input_size': 512,
        'freeze': False,
        'lr': 3e-4,
        'logdir': '/workspace/kaggle/aptos-blindness-pytorch/resnet50_',
        'device': None,
        'verbose': True,
        'check': False,  # set this true to run for 3 epochs only
        'num_epochs': 50,
        'class_names': class_names

    }
    return cfg


class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']