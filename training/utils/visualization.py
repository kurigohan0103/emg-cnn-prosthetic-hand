import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# 混同行列の可視化クラス
class ConfusionMatrixVisualizer:
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.task_names = [f'Task{i+1}' for i in range(num_classes)]
    
    # 混同行列をヒートマップで表示
    def plot_confusion_matrix(self, cm, title='Confusion Matrix', save_path=None):
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.task_names,
            yticklabels=self.task_names,
            cbar_kws={'label': 'Count'},
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # 正規化された混同行列をヒートマップで表示
    def plot_normalized_confusion_matrix(self, cm, title='Normalized Confusion Matrix', save_path=None):
        # 行ごとに正規化
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2%',
            cmap='YlOrRd',
            xticklabels=self.task_names,
            yticklabels=self.task_names,
            cbar_kws={'label': 'Percentage'},
            linewidths=0.5,
            linecolor='gray',
            vmin=0,
            vmax=1
        )
        
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # 両方の混同行列を表示
    def plot_both(self, cm, save_dir=None):
        # 通常版
        save_path_1 = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        self.plot_confusion_matrix(cm, save_path=save_path_1)
        
        # 正規化版
        save_path_2 = os.path.join(save_dir, 'confusion_matrix_normalized.png') if save_dir else None
        self.plot_normalized_confusion_matrix(cm, save_path=save_path_2)

# 訓練履歴の可視化クラス
class TrainingHistoryPlotter:
    # 損失の推移をプロット
    @staticmethod
    def plot_loss(losses, title='Training Loss', save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(losses, linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # 精度の推移をプロット
    @staticmethod
    def plot_accuracy(accuracies, title='Test Accuracy', save_path=None):
        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, linewidth=2, color='green')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # 損失と精度を並べて表示
    @staticmethod
    def plot_both(losses, accuracies, save_path=None):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 損失
        axes[0].plot(losses, linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # 精度
        axes[1].plot(accuracies, linewidth=2, color='green')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()