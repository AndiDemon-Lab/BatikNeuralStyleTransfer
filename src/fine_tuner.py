import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, inception_v3, resnet50
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop

class FineTuner:
    def __init__(self, model_name, num_classes, learning_rate, optimizer_name='adam'):
        self.num_classes = num_classes 
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.create_model(model_name)
        self.model.to(self.device)

        # initialize optimizer
        if optimizer_name == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            self.optimizer = RMSprop(self.model.parameters(), lr=learning_rate)

    def create_model(self, model_name):
        if model_name == "vgg19":
            model = vgg19(pretrained=True)
            model.classifier = nn.Sequential(
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, self.num_classes)  # Now this will work
            )
        elif model_name == "inception_v3":
            model = inception_v3(pretrained=True, aux_logits=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif model_name == "resnet50":
            model = resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        return model.to(self.device)

    def fine_tune(self, train_loader, val_loader, num_epochs):
        train_accuracies = []
        val_accuracies = []
        train_losses = [] 
        val_losses = [] 

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            correct_train = 0
            total_train = 0

            # Training phase
            running_loss = 0.0
            with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)

                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        aux_logits = outputs[1]
                        aux_loss = self.criterion(aux_logits, labels) * 0.3
                    else:
                        logits = outputs
                        aux_loss = 0

                    loss = self.criterion(logits, labels) + aux_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    _, predicted = logits.max(1)
                    total_train += labels.size(0)
                    correct_train += predicted.eq(labels).sum().item()

                    running_loss += loss.item()
                    pbar.update(1)

            train_accuracy = 100 * correct_train / total_train
            train_accuracies.append(train_accuracy)
            train_losses.append(running_loss / len(train_loader))  # average loss per epoch

            # Validation phase
            self.model.eval()
            correct_val = 0
            total_val = 0
            running_val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)

                    outputs = self.model(images)

                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    _, predicted = logits.max(1)
                    total_val += labels.size(0)
                    correct_val += predicted.eq(labels).sum().item()

                    # count val loss
                    val_loss = self.criterion(logits, labels)
                    running_val_loss += val_loss.item() 

            val_accuracy = 100 * correct_val / total_val
            val_accuracies.append(val_accuracy)
            val_losses.append(running_val_loss / len(val_loader))  # average loss per epoch

        return train_accuracies, val_accuracies, train_losses, val_losses 
