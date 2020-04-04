import os
import numpy as np
import pandas as pd
import time
from pprint import pprint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from catalyst.utils import split_dataframe_train_test
from catalyst.dl.callbacks import AccuracyCallback
from catalyst.dl.callbacks.metrics.ppv_tpr_f1 import PrecisionRecallF1ScoreCallback
from catalyst.dl import SupervisedRunner


from config import config
from dataset import AptosDataset
from model import AptosModel
from utils import get_transforms


def balance_data(csv_path: str, test_size: float = 0.2, random_state: int = 123):
    df = pd.read_csv(csv_path)
    # first class has large number of samples as compares to others
    # one way to balance is by sampling smaller amount of data
    class_0 = df[df['diagnosis'] == 0]
    class_0 = class_0.sample(400)
    class_0_train, class_0_test = split_dataframe_train_test(
        class_0, test_size=test_size, random_state=random_state)
    df_train = class_0_train
    df_test = class_0_test

    class_1 = df[df['diagnosis'] == 1]
    class_1_train, class_1_test = split_dataframe_train_test(
        class_1, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_1_train)
    df_test = df_test.append(class_1_test)

    # sub sampling data for Moderate category
    class_2 = df[df['diagnosis'] == 2]
    class_2 = class_2.sample(400)
    class_2_train, class_2_test = split_dataframe_train_test(
        class_2, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_2_train)
    df_test = df_test.append(class_2_test)

    class_3 = df[df['diagnosis'] == 3]
    class_3_train, class_3_test = split_dataframe_train_test(
        class_3, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_3_train)
    df_test = df_test.append(class_3_test)

    class_4 = df[df['diagnosis'] == 4]
    class_4_train, class_4_test = split_dataframe_train_test(
        class_4, test_size=test_size, random_state=random_state)
    df_train = df_train.append(class_4_train)
    df_test = df_test.append(class_4_test)

    return df_train, df_test


def main():

    cfg = config()
    cfg['device'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    timestr = time.strftime("%Y%m%d-%H%M%S")
    cfg['logdir'] += timestr
    pprint(cfg)

    train_df, test_df = balance_data(cfg['train_csv_path'])
    print("Train Stats:")
    print("No DR:", len(train_df[train_df['diagnosis'] == 0]))
    print("Mild:", len(train_df[train_df['diagnosis'] == 1]))
    print("Moderate:", len(train_df[train_df['diagnosis'] == 2]))
    print("Severe:", len(train_df[train_df['diagnosis'] == 3]))
    print("Proliferative DR:", len(train_df[train_df['diagnosis'] == 4]))
    print("\nTest Stats:")
    print("No DR:", len(test_df[test_df['diagnosis'] == 0]))
    print("Mild:", len(test_df[test_df['diagnosis'] == 1]))
    print("Moderate:", len(test_df[test_df['diagnosis'] == 2]))
    print("Severe:", len(test_df[test_df['diagnosis'] == 3]))
    print("Proliferative DR:", len(test_df[test_df['diagnosis'] == 4]))

    train_transforms, test_transforms = get_transforms()
    train_dataset = AptosDataset(
        img_root=cfg['img_root'],
        df=train_df,
        img_transforms=train_transforms,
        is_train=True,
    )

    test_dataset = AptosDataset(
        img_root=cfg['img_root'],
        df=test_df,
        img_transforms=test_transforms,
        is_train=False,
    )
    print(
        f"Training set size:{len(train_dataset)}, Test set size:{len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, cfg['batch_size'], shuffle=True, num_workers=1
    )
    test_loader = DataLoader(
        test_dataset, cfg['test_batch_size'], shuffle=False, num_workers=1
    )

    loaders = {
        'train': train_loader,
        'valid': test_loader
    }

    model = AptosModel(arch=cfg['arch'], freeze=cfg['freeze'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2)

    runner = SupervisedRunner(device=cfg['device'])

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,

        callbacks=[
            AccuracyCallback(
                num_classes=cfg['num_classes'],
                threshold=0.5,
                activation="Sigmoid"
            ),
#             PrecisionRecallF1ScoreCallback(
#                 class_names=cfg['class_names'],
#                 num_classes=cfg['num_classes']
#             )

        ],
        logdir=cfg['logdir'],
        num_epochs=cfg['num_epochs'],
        verbose=cfg['verbose'],
        # set this true to run for 3 epochs only
        check=cfg['check']
    )


if __name__ == "__main__":
    main()