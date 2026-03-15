import torch
import torch.nn as nn
from tqdm import tqdm
from .model_saver import ModelSaver, EarlyStopping


# モデル訓練クラス
class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, save_dir=None, save_checkpoints=False, early_stopping=False):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.losses = []
        self.accuracies = []
        
        # モデル保存
        self.save_dir = save_dir
        self.save_checkpoints = save_checkpoints
        
        if save_dir is not None:
            self.model_saver = ModelSaver(save_dir)
        else:
            self.model_saver = None
        
        # 早期終了
        if early_stopping:
            self.early_stopping = EarlyStopping(patience=20)
        else:
            self.early_stopping = None
    
    # 1エポック訓練
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for emg, label in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(emg)
            loss = self.criterion(outputs, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss
    
    # 訓練ループ
    def train(self, epochs, test_loader, evaluator, checkpoint_every=10):
        print(f"\nTraining Start ({epochs} epochs)")
        
        for epoch in range(epochs):
            # 訓練
            loss = self.train_epoch()
            self.losses.append(loss)
            
            # 評価
            accuracy = evaluator.calculate_accuracy(test_loader)
            self.accuracies.append(accuracy)
            
            # ベストモデル保存
            if self.model_saver is not None:
                self.model_saver.save_best_model(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    loss,
                    accuracy
                )
            
            # チェックポイント保存
            if self.save_checkpoints and (epoch + 1) % checkpoint_every == 0:
                if self.model_saver is not None:
                    self.model_saver.save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch + 1,
                        loss,
                        accuracy
                    )
            
            # 進捗表示
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Loss: {loss:.6f} | "
                      f"Acc: {accuracy:.2f}%")
            
            # 早期終了チェック
            if self.early_stopping is not None:
                if self.early_stopping(accuracy):
                    print(f"\nTraining stopped early at epoch {epoch+1}")
                    break
        
        # 最終モデル保存
        if self.model_saver is not None:
            self.model_saver.save_final_model(
                self.model,
                self.optimizer,
                epoch + 1,
                self.losses[-1],
                self.accuracies[-1]
            )
            
            # model_weight.pthとしても保存
            self.model_saver.save_model_weights(
                self.model,
                filename='model_weight.pth'
            )
        
        print(f"\nTraining completed!")
        print(f"   Final Loss: {self.losses[-1]:.6f}")
        print(f"   Final Accuracy: {self.accuracies[-1]:.2f}%")
        
        if self.model_saver is not None:
            best_info = self.model_saver.get_best_model_info()
            print(f"   Best Accuracy: {best_info['best_accuracy']:.2f}% "
                  f"(Epoch {best_info['best_epoch']})")
    
    # 訓練履歴を取得
    def get_history(self):
        return {
            'losses': self.losses,
            'accuracies': self.accuracies
        }
    
    # ベストモデルのパスを取得
    def get_best_model_path(self):
        if self.save_dir is not None:
            import os
            return os.path.join(self.save_dir, 'best_model.pth')
        return None