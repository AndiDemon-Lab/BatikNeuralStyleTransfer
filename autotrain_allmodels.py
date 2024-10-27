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

warnings.filterwarnings("ignore")

# ------------------------Training Preparation----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer = Transform()

# hyperparameters
batch_size = 8
num_epochs = 1

# learning rates for tuning
# whatever value
learning_rates = [1e-3, 1e-4]

# optimizers to fine-tune
# the option is 'adam', 'sgd', and 'rmsprop'
optimizers = ['rmsprop', 'sgd']

# models to fine-tune
# the option is 'vgg19', 'inception_v3', 'resnet50'
models_to_finetune = ["vgg19", "inception_v3", "resnet50"]

#-------------------------Fine-tuning----------------------------

results = {}
# training loop through each model, learning rate, and optimizer
for model_name in models_to_finetune:
    results[model_name] = {}  # Create an entry for each model
    for optimizer_name in optimizers:
        results[model_name][optimizer_name] = {}  # Create an entry for each optimizer
        for learning_rate in learning_rates:
            print(f"Fine-tuning '{model_name}' with optimizer '{optimizer_name}' and learning rate '{learning_rate}'...")

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
            model_save_dir = f'training_assets/saved_models/{optimizer_name}_{learning_rate}_{num_epochs}'
            os.makedirs(model_save_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            model_save_path = os.path.join(model_save_dir, f'{model_name}_{optimizer_name}_lr{learning_rate}_{timestamp}_model.pth')
            torch.save(fine_tuner.model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

            # visualization
            plot_save_dir = f'training_assets/process_training/{optimizer_name}_{learning_rate}_{num_epochs}'
            os.makedirs(plot_save_dir, exist_ok=True)
            vis = Visualization(model_name, plot_save_dir, num_epochs)
            vis.plot_accuracy(train_acc, val_acc)
            vis.plot_loss(train_loss, val_loss)

            # store training and validation results
            sanitized_lr = str(learning_rate).replace('.', ',')
            results[model_name][optimizer_name][sanitized_lr] = {
                "train_accuracy": train_acc[0],
                "val_accuracy": val_acc[0],
                "train_loss": train_loss[0],
                "val_loss": val_loss[0],
                "training_time": training_time,
                "hyperparameters": {
                    "optimizer": optimizer_name,
                    "criterion": str(fine_tuner.criterion.__class__.__name__),
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs
                }
            }

            torch.cuda.empty_cache()

# save results for all combinations
timestamp = time.strftime("%Y%m%d%H%M%S")
results_file_path = f'training_assets/results_{timestamp}.json'
with open(results_file_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Fine-tuning completed, results saved to --> {results_file_path}")
