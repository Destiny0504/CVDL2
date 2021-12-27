import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageChops
import torch
import os

class MyDataset(Dataset):
    def __init__(self, samples, classes, cls_to_idx, transform=None):
        self.transform = transform

        self.calsses = classes
        self.cls_to_idx = cls_to_idx
        self.targets = [s[1] for s in samples]
        self.samples = [s[0] for s in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.targets[idx]

        img = Image.open(path)
        img = self.resize_image(img)
        try:
            img = self.transform(img)
        except:
            print(path)

        return img, label

    def resize_image(self, img):
        img_width, img_height = img.size

        left = img_width // 2 - 120
        right = img_width // 2 + 120
        top = img_height // 2 - 120
        bottom = img_height // 2 + 120
        img = img.crop((left, top, right, bottom))
        return img

def find_data_and_index(folder):

    print(list(enumerate(os.listdir(folder))))

    # the data structure of instance
    # '''
    #     instance : include two types of data (cats and dogs)
    #     instance[i][0] : The whole data in this class
    #     instance[i][1] : The amount of the data in this class
    # '''
    instances = {}

    # classes is the class that we have
    classes = os.listdir(folder)

    # class to index is a dictionary that map the class(str) into a number
    cls_to_idx = {name: i for i, name in enumerate(os.listdir(folder))}


    for cls in cls_to_idx.keys():
        cls_idx = cls_to_idx[cls]
        cls_path = os.path.join(folder, cls)
        files = os.listdir(cls_path)

        # This item stored the dataset of a class and each data in the list
        # is a tuple ('data_path', data_label)
        items = [(os.path.join(cls_path, name), cls_idx) for name in files]

        # (whole dataset(a class), the amount of the dataset)
        instances[cls_idx] = (items, len(items))

    print(f'There are {len(classes)} classes and {sum([s[1] for s in instances.values()])} images.')
    return classes, cls_to_idx, instances

def split_data(instances, seed):
    train_sample = []
    validation_sample = []
    test_sample = []

    for cls_idx, cls_samples in instances.items():
        # split the dataset into train, validation and test [9965, 1246, 1246] (the size is optional)
        # If this is augmentation dataset [15038,1880,1880]
        train, validation, test = torch.utils.data.random_split(cls_samples[0], [9965, 1246, 1246])
        train_sample += train
        validation_sample += validation
        test_sample += test

    return train_sample, validation_sample, test_sample

def prepare_everything_for_train(data_folder_path, transform, seed = 42, batch_size = 64, num_workers = 4):

    # labeling the data
    classes, cls_to_idx, instances = find_data_and_index(data_folder_path)

    #split the dataset into train validation and test
    train_data, validation_data, test_data =  split_data(instances=instances, seed=seed)

    # The dataset is created here
    train_dataset = MyDataset(train_data, classes=classes, cls_to_idx=cls_to_idx, transform=transform)
    validation_dataset = MyDataset(validation_data, classes=classes, cls_to_idx=cls_to_idx, transform=transform)
    test_dataset = MyDataset(test_data, classes=classes, cls_to_idx=cls_to_idx, transform=transform)


    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    return train_loader, validation_loader, test_loader

if __name__ == '__main__':

    # setting the training config
    SEED = 42
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    LR = 1e-4
    EPOCHS = 10
    WEIGHT_DECAY = 1e-4
    device = 'cuda:0'

    # setting the training seed
    torch.manual_seed(14)
    torch.cuda.manual_seed(14)

    # create the transform sequence (used in myDataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # ./PetImages is the place you put your data
    folder = os.path.expanduser('./PetImages_augmentation')

    train_loader, validation_loader, test_loader = prepare_everything_for_train(
        folder,
        seed=SEED,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
        )
    print(train_loader)