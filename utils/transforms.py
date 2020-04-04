import torchvision.transforms as transforms


def get_transforms():
    # transforms
    train_transforms = transforms.Compose([
        transforms.Resize((524, 524)),
        transforms.RandomCrop(512),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((524, 524)),
        transforms.CenterCrop((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])
    return train_transforms, val_transforms