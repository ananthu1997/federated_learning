import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idx, idxs, logger):
        self.idx = idx
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        self.min_delta = 0.001
        self.best_loss = float('inf')
        self.patience = 10
        self.criterion = nn.CrossEntropyLoss().to(self.device)


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),shuffle=False)
        return trainloader, validloader, testloader
    
    def plot_learning_curve(self, train_losses, validation_losses):
        plt.figure(figsize=(10, 12))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def update_weights(self, model, global_round, global_weights):
        # Set mode to train model
        model.train()
        epoch_loss = []
        validation_losses = []
        train_losses = []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.9)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         )
        #### Moh ####
        if self.args.withDP:
            privacy_engine = PrivacyEngine(
                model,
                batch_size=self.args.virtual_batch_size,
                sample_size=len(self.trainloader.dataset),
                alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                noise_multiplier=self.args.noise_multiplier,
                max_grad_norm=self.args.max_grad_norm,
            )
            privacy_engine.attach(optimizer)
            assert self.args.virtual_batch_size % self.args.local_bs == 0 # VIRTUAL_BATCH_SIZE should be divisible by BATCH_SIZE
            virtual_batch_rate = int(self.args.virtual_batch_size / self.args.local_bs)
        #############
        for iter in range(self.args.local_ep):
            batch_loss = []
            loop = tqdm(self.trainloader)
            for batch_idx, (images, labels) in enumerate(loop):
                images, labels = images.to(self.device), labels.to(self.device)
                # model.zero_grad()
                # log_probs = model(images)
                # loss = self.criterion(log_probs, labels)
                optimizer.zero_grad()
                output = model(images)
                loss = self.criterion(output, labels)
                if self.args.FedProx:
                    prox_term = 0.0
                    for param, global_param in zip(model.parameters(), global_weights.values()):
                        prox_term += torch.norm(param - global_param) ** 2
                    loss += self.args.mu / 2 * prox_term  # Add the proximal term to the loss
                loss.backward()

                ### Moh ####
                if self.args.withDP:
                # optimizer.step()
                # take a real optimizer step after N_VIRTUAL_STEP steps t
                    if ((batch_idx + 1) % virtual_batch_rate == 0) or ((batch_idx + 1) == len(self.trainloader)):
                        optimizer.step()
                    else:
                        optimizer.virtual_step() # take a virtual step
                else:
                    optimizer.step()
                #############

                if self.args.verbose and (batch_idx % self.args.verbose == 0):
                    if self.args.withDP:
                        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(self.args.delta)
                        print('| Global Round : {} | User ID : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} | ε = {:.2f} | α = {:.2f} | δ = {}'.format(
                            global_round+1, self.idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item(), 
                            epsilon, best_alpha, self.args.delta))
                    else:
                        print('| Global Round : {} | User ID : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f} '.format(
                            global_round+1, self.idx, iter, batch_idx * len(images),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))  
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            avg_train_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, labels in self.validloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = model(images)
                    val_loss += self.criterion(output, labels).item()
            val_loss /= len(self.validloader)
            # Early stopping check
            if val_loss <= self.best_loss - self.min_delta:
                self.best_loss = val_loss
                count = 0
            else:
                count += 1
                if count >= self.patience:
                    print("Early stopping triggered")
                    break
        # Plot learning curve
        # self.plot_learning_curve(train_losses, validation_losses)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    actual_labels = []
    prediction = []
    device = 'cuda' if args.gpu else 'cpu'
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset,
                            shuffle=False)
    flag = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)
        # Inference
        output = model(images)
        _,pred=torch.max(output,dim=1)
        total+=(pred==labels).sum()
        actual_labels.extend(labels.cpu().numpy())
        prediction.extend(pred.cpu().numpy())

    confusion_mat = confusion_matrix(actual_labels, prediction)
    class_report = classification_report(actual_labels, prediction, output_dict=True)

    # accuracy = correct/total
    accuracy = total/len(testloader.dataset)
    return accuracy, loss, confusion_mat, class_report, actual_labels
