from matplotlib.pyplot import cla
from torchvision import models
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import cv2
import pet_test
import os
import numpy

def Q51():
    model = models.resnet50()
    print(model)

def Q52():
    img1 = cv2.imread('./exp1.png')
    cv2.imshow('training', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Q53(select):
    torch.manual_seed(14)
    torch.cuda.manual_seed(14)
    folder = os.path.expanduser('./PetImages')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    classes, cls_to_idx, instances = pet_test.find_data_and_index(folder)
    class_dict = {'0': 'Dog', '1': 'Cat'}
    train_data, validation_data, test_data =  pet_test.split_data(instances=instances, seed=14)
    test_dataset = pet_test.MyDataset(test_data, classes=classes, cls_to_idx=cls_to_idx, transform=transform)
    print(len(test_dataset))
    model = models.resnet50()
    finetune_model = torch.nn.Linear(1000, 10)
    model.load_state_dict(torch.load('./exp2/resnet50.model'))
    finetune_model.load_state_dict(torch.load('./exp2/finetune.model'))
    model.eval()
    finetune_model.eval()
    img, label = test_dataset[select]
    output = model(torch.unsqueeze(img,0))
    output = finetune_model(output)
    _, preds = torch.max(output, 1)
    img = cv2.imread(test_dataset.samples[select])
    cv2.imshow(class_dict[str(label)] + '      Predicted:' + str(preds.item()) + f'({class_dict[str(preds.item())]})', img)
    print(preds)


def Q54():
    pass