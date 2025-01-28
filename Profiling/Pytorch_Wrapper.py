import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

class Classifier:
    def __init__(self, model_name, num_classes=1, batch_size=32, lr=0.0001, num_epochs=10, train_data_path=None, test_data_path=None):
        """
        Initializes the Pytorch_Wrapper class.
        Args:
            model_name (str): Name of the model to be used.
            num_classes (int, optional): Number of output classes. Defaults to 2.
            batch_size (int, optional): Batch size for training and testing. Defaults to 32.
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.0001.
            num_epochs (int, optional): Number of epochs for training. Defaults to 10.
            train_data_path (str, optional): Path to the training data. Defaults to None.
            test_data_path (str, optional): Path to the testing data. Defaults to None.
        Attributes:
            train_data_path (str): Path to the training data.
            test_data_path (str): Path to the testing data.
            batch_size (int): Batch size for training and testing.
            num_epochs (int): Number of epochs for training.
            device (torch.device): Device to run the model on (CUDA if available, else CPU).
            model (torch.nn.Module): The loaded and modified model.
            criterion (torch.nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for the model's classifier parameters.
            train_loader (torch.utils.data.DataLoader or None): DataLoader for training data.
            test_loader (torch.utils.data.DataLoader or None): DataLoader for testing data.
            dataset_sizes (dict or None): Sizes of the training and testing datasets.
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the pretrained model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Freeze all layers except the classifier
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classifier layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model = self.model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)

        # Only prepare dataloaders if paths are provided
        if train_data_path and test_data_path:
            self.train_loader, self.test_loader, self.dataset_sizes = self._prepare_dataloaders()
        else:
            self.train_loader = None
            self.test_loader = None
            self.dataset_sizes = None

    def _prepare_dataloaders(self):
        """
        Prepares the dataloaders for training and testing datasets.
        This method checks if the paths for training and testing data are provided.
        If not, it returns None for both dataloaders and dataset sizes.
        It applies the following transformations to the images:
        - Resize to 224x224 pixels
        - Convert to tensor
        - Normalize with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
        The method then creates image datasets using the specified transformations
        and initializes dataloaders for both training and testing datasets.
        Returns:
            tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training dataset.
            - test_loader (DataLoader): DataLoader for the testing dataset.
            - dataset_sizes (dict): A dictionary containing the sizes of the training and testing datasets.
        """
        if not self.train_data_path or not self.test_data_path:
            return None, None, None

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        image_datasets = {
            'train': datasets.ImageFolder(self.train_data_path, data_transforms['train']),
            'test': datasets.ImageFolder(self.test_data_path, data_transforms['test'])
        }

        dataloaders = {
            'train': DataLoader(image_datasets['train'], batch_size=self.batch_size, shuffle=True),
            'test': DataLoader(image_datasets['test'], batch_size=self.batch_size, shuffle=False)
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
        return dataloaders['train'], dataloaders['test'], dataset_sizes

    def train(self):
        """
        Trains the model using the provided training and test data loaders.
        This method performs the following steps:
        1. Checks if the data loaders are initialized.
        2. Iterates over the specified number of epochs.
        3. For each epoch, iterates over the training and test phases.
        4. In the training phase, sets the model to training mode and updates the model weights.
        5. In the test phase, sets the model to evaluation mode and evaluates the model performance.
        6. Computes and prints the loss and accuracy for each phase.
        7. Tracks and updates the best model weights based on test accuracy.
        8. Loads the best model weights after training is complete.
        Raises:
            ValueError: If the data loaders are not initialized.
        Prints:
            The loss and accuracy for each phase at each epoch.
            A message indicating whether the test accuracy has improved.
            A message indicating the completion of training and the best test accuracy achieved.
        """
        if not self.train_loader or not self.test_loader:
            raise ValueError("Data loaders not initialized. Provide train and test data paths.")

        best_acc = 0.0
        best_model_wts = self.model.state_dict()

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch}/{self.num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluation mode

                running_loss = 0.0
                running_corrects = 0

                loader = self.train_loader if phase == 'train' else self.test_loader
                for inputs, labels in loader:
                    inputs = inputs.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        preds = (torch.sigmoid(outputs) > 0.5).float()

                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # Only track improvements in test accuracy
                if phase == 'test':
                    if epoch_acc > best_acc:
                        print(f'Accuracy improved from {best_acc:.4f} to {epoch_acc:.4f} at epoch {epoch}. Updating best model...')
                        best_acc = epoch_acc
                        best_model_wts = self.model.state_dict()
                    else:
                        print(f'No improvement in accuracy at epoch {epoch}. Best accuracy remains: {best_acc:.4f}')

        # Load best model weights after training
        self.model.load_state_dict(best_model_wts)
        print('Training complete. Best test accuracy:', best_acc)

    def save_model(self, file_name):
        """
        Saves the PyTorch model's state dictionary to a specified file.
        Args:
            file_name (str): The name of the file to save the model state dictionary to.
        Returns:
            None
        Side Effects:
            - Creates a directory named 'assets' if it does not already exist.
            - Saves the model state dictionary to the specified file within the 'assets' directory.
            - Prints a success message upon saving the model.
        """
        folder = 'assets'
        if not os.path.exists(folder):
            os.makedirs(folder)

        file_path = os.path.join(folder, file_name)
        torch.save(self.model.state_dict(), file_path)
        print('Pytorch Model saved successfully')

    def load_model(self, file_path):
        """
        Loads the model state dictionary from a specified file path.

        Args:
            file_path (str): The path to the file containing the model state dictionary.

        Returns:
            torch.nn.Module: The model with the loaded state dictionary.

        Prints:
            str: Confirmation message indicating successful model loading.
        """
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        print('Pytorch Model loaded successfully')
        return self.model
        
    def eval(self):
        self.model.eval()