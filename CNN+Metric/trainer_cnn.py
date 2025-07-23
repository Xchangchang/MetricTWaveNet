#!/usr/bin/env python3
"""
CNN Trainer Class for T-phase Classification

Contains the training logic for CNN baseline model.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from data_loader import get_dataloaders
from model import CNN
from loss import get_class_weights

class CNNTrainer:
    """CNN trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create data loaders
        self.train_loader, self.test_loader = get_dataloaders(
            config['data_root'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            class_names=config['class_names'],
        )

        # Create model
        self.model = CNN(num_classes=len(config['class_names'])).to(self.device)

        # Calculate class weights
        class_counts = np.zeros(len(config['class_names']), dtype=np.int64)
        for _, labels in self.train_loader:
            for c in range(len(config['class_names'])):
                class_counts[c] += (labels == c).sum().item()
        print(f"Class counts: {class_counts}")

        weights = get_class_weights(class_counts, beta=0.999)
        print(f"Class weights: {weights}")

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32).to(self.device)
        )

        # Optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=config['lr'], 
                                  weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10)

        # Training history
        self.train_losses, self.test_losses = [], []
        self.train_accs, self.test_accs = [], []

        # Create save directory
        os.makedirs(config['save_dir'], exist_ok=True)

    def train(self):
        """Main training loop"""
        best_loss = float('inf')
        best_acc = 0.0
        patience, epochs_no_improve = 15, 0

        for epoch in range(self.config['epochs']):
            # Train one epoch
            train_loss, train_acc = self._train_epoch(epoch)
            
            # Evaluate
            test_loss, test_acc = self._eval_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step(test_loss)

            # Record history
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)

            # Save best model
            if test_loss < best_loss:
                best_loss = test_loss
                best_acc = test_acc
                self._save_model('best_model.pth')
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Save latest model
            self._save_model('latest_model.pth')

            print(f"Epoch {epoch + 1}: TrainLoss {train_loss:.4f}, "
                  f"TestLoss {test_loss:.4f}, TrainAcc {train_acc:.4f}, "
                  f"TestAcc {test_acc:.4f}")

            # Early stopping
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        # Save training history
        self._save_training_history()
        
        # Final evaluation and plots
        self._plot_training_process()
        self._final_evaluation()
        
        return best_acc

    def _train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, targets in tqdm(self.train_loader, desc=f"Train Epoch {epoch + 1}"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.ce_loss(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(self.train_loader), correct / total

    def _eval_epoch(self, epoch):
        """Evaluate one epoch"""
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc=f"Test Epoch {epoch + 1}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                logits = self.model(inputs)
                loss = self.ce_loss(logits, targets)

                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return total_loss / len(self.test_loader), correct / total

    def _save_model(self, filename):
        """Save model checkpoint"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }, os.path.join(self.config['save_dir'], filename))

    def _save_training_history(self):
        """Save training history as numpy arrays"""
        np.save(os.path.join(self.config['save_dir'], 'train_losses.npy'), 
                np.array(self.train_losses))
        np.save(os.path.join(self.config['save_dir'], 'test_losses.npy'), 
                np.array(self.test_losses))
        np.save(os.path.join(self.config['save_dir'], 'train_accs.npy'), 
                np.array(self.train_accs))
        np.save(os.path.join(self.config['save_dir'], 'test_accs.npy'), 
                np.array(self.test_accs))

    def _plot_training_process(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Testing Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.test_accs, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Testing Accuracy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['save_dir'], 'training_process.png'))
        plt.close()

    def _final_evaluation(self):
        """Final evaluation with confusion matrix and classification report"""
        # Load best model
        best_model_path = os.path.join(self.config['save_dir'], 'best_model.pth')
        checkpoint = torch.load(best_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Final Evaluation"):
                inputs = inputs.to(self.device)
                logits = self.model(inputs)
                _, preds = logits.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        # Save predictions
        np.save(os.path.join(self.config['save_dir'], 'predictions.npy'), all_preds)
        np.save(os.path.join(self.config['save_dir'], 'targets.npy'), all_targets)

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.config['class_names'],
                    yticklabels=self.config['class_names'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.config['save_dir'], 'confusion_matrix.png'))
        plt.close()

        # Classification report
        report = classification_report(
            all_targets, all_preds,
            target_names=self.config['class_names'],
            digits=4
        )
        
        with open(os.path.join(self.config['save_dir'], 'classification_report.txt'), 'w') as f:
            f.write(report)

        print(f"Final evaluation completed. Results saved to: {self.config['save_dir']}")
        print("\nClassification Report:")
        print(report)
