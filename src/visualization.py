import matplotlib.pyplot as plt
import os

class Visualization:
    def __init__(self, model_name, plot_save_dir, num_epochs):
        self.model_name = model_name
        self.plot_save_dir = plot_save_dir
        self.num_epochs = num_epochs

    def plot_accuracy(self, train_acc, val_acc):
        plt.figure()
        plt.plot(range(1, self.num_epochs + 1), train_acc, color='blue', label='Train Accuracy')
        plt.plot(range(1, self.num_epochs + 1), val_acc, color='orange', label='Validation Accuracy')
        plt.title(f'Training and Validation Accuracy for {self.model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid()
        
        # Save plot to file
        plot_save_path = os.path.join(self.plot_save_dir, f'{self.model_name}_accuracy_plot.png')
        plt.savefig(plot_save_path)
        plt.close()

    def plot_loss(self, train_loss, val_loss):
        plt.figure()
        plt.plot(range(1, self.num_epochs + 1), train_loss, color='blue', label='Train Loss')
        plt.plot(range(1, self.num_epochs + 1), val_loss, color='orange', label='Validation Loss')
        plt.title(f'Training and Validation Loss for {self.model_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()

        # Save plot to file
        plot_save_path = os.path.join(self.plot_save_dir, f'{self.model_name}_loss_plot.png')
        plt.savefig(plot_save_path)
        plt.close()
