import os
import json
import torch
import warnings
import time
from torch.utils.data import DataLoader
from torchvision import datasets
from src.fine_tuner import FineTuner
from src.transform import Transform
from src.visualization import Visualization
from torch.optim import Adam, SGD, RMSprop

warnings.filterwarnings("ignore")

# ------------------------Training Preparation----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = Transform()

# hyperparameters
batch_size = 8
num_epochs = 3

# for tuning
learning_rates = [1e-3]
optimizers = ['adam']

results = {}

# models to fine-tune
models_to_finetune = ["vgg19", "inception_v3", "resnet50"]

#-------------------------Fine-tuning----------------------------

model_save_dir = 'training_assets/saved_models'
plot_save_dir = 'training_assets/process_training'
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(plot_save_dir, exist_ok=True)

# training loop through each model, learning rate, and optimizer
for model_name in models_to_finetune:
    for learning_rate in learning_rates:
        for optimizer_name in optimizers:
            print(f"Fine-tuning {model_name} with learning_rate={learning_rate} and optimizer={optimizer_name}...")

            # load dataset
            data_transform = transformer.get_transform(model_name)

            train_dataset = datasets.ImageFolder(root="data/train_dikit", transform=data_transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            val_dataset = datasets.ImageFolder(root="data/test", transform=data_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            fine_tuner = FineTuner(model_name, num_classes=15, learning_rate=learning_rate, optimizer_name=optimizer_name)

            train_acc = []
            val_acc = []
            train_loss = []
            val_loss = []

            # count training time
            start_time = time.time()

            # fine-tune current model
            train_epoch_acc, val_epoch_acc, train_epoch_loss, val_epoch_loss = fine_tuner.fine_tune(train_loader, val_loader, num_epochs)

            train_acc.extend(train_epoch_acc)
            val_acc.extend(val_epoch_acc)
            train_loss.extend(train_epoch_loss)
            val_loss.extend(val_epoch_loss)

            end_time = time.time()
            training_time = end_time - start_time

            # save trained model
            model_save_path = os.path.join(model_save_dir, f'{model_name}_{optimizer_name}_lr{learning_rate}_model.pth')
            torch.save(fine_tuner.model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            # visualization
            vis = Visualization(model_name, plot_save_dir, num_epochs)
            vis.plot_accuracy(train_acc, val_acc)
            vis.plot_loss(train_loss, val_loss)

            # training and validation results
            results[f"{model_name}"] = {
                "train_accuracy": train_acc[0],
                "val_accuracy": val_acc[0],
                "train_loss": train_loss[0],
                "val_loss": val_loss[0],
                "training_time": training_time,
                "hyperparameters": {
                    "optimizer": optimizer_name,
                    "criterion": fine_tuner.criterion,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs
                }
            }

# save results
with open('training_assets/fine_tuning_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Fine-tuning completed, results saved to --> training_assets")
