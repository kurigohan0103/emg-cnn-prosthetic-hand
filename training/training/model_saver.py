import torch
import os
import json
from datetime import datetime
import glob


# モデル保存管理クラス
class ModelSaver:
    def __init__(self, save_dir, keep_last_n=3):
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        
        # ディレクトリ作成
        os.makedirs(save_dir, exist_ok=True)
        
        # ベストモデルの記録
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
        print(f"ModelSaver initialized")
        print(f"   Save directory: {save_dir}")
    
    # チェックポイント保存
    def save_checkpoint(self, model, optimizer, epoch, loss, accuracy, filename=None):
        if filename is None:
            filename = f'checkpoint_epoch{epoch}.pth'
        
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filename}")
        
        # 古いチェックポイントを削除
        self._cleanup_old_checkpoints()
    
    # モデルの重みのみ保存
    def save_model_weights(self, model, filename='model_weight.pth'):
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(model.state_dict(), filepath)
        print(f"Model weights saved: {filename}")
        
        return filepath
    
    # ベストモデルを保存（精度が改善した場合のみ）
    def save_best_model(self, model, optimizer, epoch, loss, accuracy, filename='best_model.pth'):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            
            filepath = os.path.join(self.save_dir, filename)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, filepath)
            print(f"[Best] Model updated: Epoch {epoch}, Accuracy {accuracy:.2f}%")
            
            return True
        
        return False
    
    # 最終モデルを保存
    def save_final_model(self, model, optimizer, epoch, loss, accuracy, filename='final_model.pth'):
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat(),
            'final': True
        }
        
        torch.save(checkpoint, filepath)
        print(f"   Final model saved: {filename}")
        print(f"   Epoch: {epoch}, Accuracy: {accuracy:.2f}%")
    
    # モデル情報をJSON保存
    def save_model_info(self, model, config, results, filename='model_info.json'):
    
        filepath = os.path.join(self.save_dir, filename)
        
        # モデルのパラメータ数を計算
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'model_architecture': str(model),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'config': config,
            'results': {k: v for k, v in results.items() if k not in ['confusion_matrix', 'predictions', 'targets', 'classification_report', 'confusion_patterns']},
            'best_epoch': self.best_epoch,
            'best_accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(info, f, indent=2, default=str)
        
        print(f"Model info saved: {filename}")
    
    # 古いチェックポイントを削除
    def _cleanup_old_checkpoints(self):

        checkpoint_pattern = os.path.join(self.save_dir, 'checkpoint_epoch*.pth')
        checkpoints = glob.glob(checkpoint_pattern)
        
        if len(checkpoints) > self.keep_last_n:
            # 作成時刻でソート
            checkpoints.sort(key=os.path.getmtime)
            
            # 古いものを削除
            for old_checkpoint in checkpoints[:-self.keep_last_n]:
                os.remove(old_checkpoint)
                print(f"   Removed old checkpoint: {os.path.basename(old_checkpoint)}")
    
    # チェックポイントをロード
    def load_checkpoint(self, filepath, model, optimizer=None):
    
        checkpoint = torch.load(filepath)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Checkpoint loaded: {os.path.basename(filepath)}")
        print(f"   Epoch: {checkpoint['epoch']}")
        print(f"   Accuracy: {checkpoint['accuracy']:.2f}%")
        
        return checkpoint
    
    # モデルの重みのみロード
    def load_model_weights(self, filepath, model):
        model.load_state_dict(torch.load(filepath))
        print(f"Model weights loaded: {os.path.basename(filepath)}")
    
    # ベストモデルの情報を取得
    def get_best_model_info(self):
        return {
            'best_epoch': self.best_epoch,
            'best_accuracy': self.best_accuracy
        }


# 早期終了クラス
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_accuracy = 0.0
        self.early_stop = False
    
    # 早期終了判定
    def __call__(self, accuracy):
        if accuracy > self.best_accuracy + self.min_delta:
            # 改善した
            self.best_accuracy = accuracy
            self.counter = 0
        else:
            # 改善しなかった
            self.counter += 1
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\nEarly stopping triggered!")
                print(f"   No improvement for {self.patience} epochs")
                print(f"   Best accuracy: {self.best_accuracy:.2f}%")
        
        return self.early_stop