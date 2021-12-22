import pet_test
import torch
import torchvision.transforms as transforms
import os
from torch import torch
import copy
from tqdm import tqdm
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

def train_model(model, finetune, dataloaders, criterion, optimizer, writer, num_epochs=25, is_inception=False):
    val_acc_history = []
    loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_finetune_model = copy.deepcopy(finetune.state_dict())
    best_acc = 0.0
    train_steps = 1
    valid_steps = 1
    for epoch in tqdm(range(num_epochs)):
        print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in iter(dataloaders[phase]):
                inputs = inputs.to('cuda:0')
                labels = labels.to('cuda:0')
                model = model.to('cuda:0')
                finetune = finetune.to('cuda:0')
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        outputs = finetune(outputs)
                        # print(len(outputs))
                        # print(len(outputs[0]))
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    # print(labels.data)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        writer.add_scalar('training_loss', loss.item(), train_steps)
                        loss.backward()
                        optimizer.step()
                        train_steps += 1
                        #print('propagation')

                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                loss_history.append(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_finetune_model = copy.deepcopy(finetune.state_dict())
            if phase == 'val':
                epoch_acc.to('cpu')
                writer.add_scalar('training_accuracy', epoch_acc, valid_steps)
                valid_steps += 1
                val_acc = copy.deepcopy(epoch_acc)
                val_acc_history.append(val_acc)
                epoch_acc.to('cuda:0')

        print()
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    finetune.load_state_dict(best_finetune_model)
    return model, val_acc_history, finetune, optimizer,loss_history

if __name__ == '__main__':
    # setting the training config
    SEED = 42
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    LR = 1e-5
    EPOCHS = 10
    WEIGHT_DECAY = 1e-4
    device = 'cuda:0'

    # choose model
    Model = models.resnet50(pretrained=False)
    linear = torch.nn.Linear(1000, 10)

    # choose optimizer
    optimizer = torch.optim.Adam(Model.parameters(), lr=LR)

    # setting the training seed
    torch.manual_seed(14)
    torch.cuda.manual_seed(14)

    # create the transform sequence (used in myDataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # ./PetImages is the place you put your data
    folder = os.path.expanduser('./PetImages_augmentation')

    train_loader, validation_loader, test_loader = pet_test.prepare_everything_for_train(
        folder,
        seed=SEED,
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
        )

    writer = SummaryWriter('./exp3')
    resnet50_model, training_history, finetune_model, cur_optimizer, loss = train_model(Model, linear
    , dataloaders={'train':train_loader,'val':test_loader},criterion=torch.nn.CrossEntropyLoss(),optimizer=optimizer,num_epochs=10, writer=writer)

    torch.save(resnet50_model.state_dict(), './exp3/resnet50.model')
    torch.save(finetune_model.state_dict(), './exp3/finetune.model')
    torch.save(cur_optimizer.state_dict(), './exp3/vgg_optimizer')
