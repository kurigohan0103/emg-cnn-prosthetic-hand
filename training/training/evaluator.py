import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# モデル評価クラス
class Evaluator:
    def __init__(self, model, device, num_classes=6):
        self.model = model
        self.device = device
        self.num_classes = num_classes
    
    # 精度を計算
    def calculate_accuracy(self, test_loader):
        self.model.eval()
        correct = 0
        
        with torch.no_grad():
            for emg, label in test_loader:
                outputs = self.model(emg)
                pred = outputs.argmax(dim=1).cpu().numpy()
                target = label.argmax(dim=1).cpu().numpy()
                correct += np.sum(pred == target)
        
        accuracy = correct / len(test_loader.dataset) * 100
        return accuracy
    
    # 詳細評価
    def evaluate(self, test_loader, verbose=True):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for emg, label in test_loader:
                outputs = self.model(emg)
                pred = outputs.argmax(dim=1).cpu().numpy()
                target = label.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(pred)
                all_targets.extend(target)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 基本評価指標
        results = {
            'accuracy': accuracy_score(all_targets, all_preds) * 100,
            'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
            'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
            'f1': f1_score(all_targets, all_preds, average='weighted', zero_division=0) * 100,
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'predictions': all_preds,
            'targets': all_targets
        }
        
        # クラスごとの詳細評価
        report_dict = classification_report(
            all_targets,
            all_preds,
            target_names=[f'Task{i+1}' for i in range(self.num_classes)],
            digits=4,
            zero_division=0,
            output_dict=True
        )
        results['classification_report'] = report_dict
        
        # 混同パターン分析
        results['confusion_patterns'] = self._analyze_confusion_patterns(
            results['confusion_matrix']
        )
        
        if verbose:
            self.print_results(results)
        
        return results
    
    # 混同パターンを分析
    def _analyze_confusion_patterns(self, cm):
        mistakes = []
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j and cm[i, j] > 0:
                    mistakes.append({
                        'count': int(cm[i, j]),
                        'true_label': int(i),
                        'pred_label': int(j),
                        'percentage': float(cm[i, j] / cm[i].sum() * 100)
                    })
        
        # 回数でソート
        mistakes.sort(key=lambda x: x['count'], reverse=True)
        
        return mistakes
    
    # 評価結果を表示"
    def print_results(self, results):
        # 評価指標
        print("\nEvaluation Metrics")
        print(f"Accuracy:  {results['accuracy']:.2f}%")
        print(f"Precision: {results['precision']:.2f}%")
        print(f"Recall:    {results['recall']:.2f}%")
        print(f"F1-Score:  {results['f1']:.2f}%")
        
        # クラスごとの詳細評価
        print("\nClass-by-Class Detailed Evaluation")
        
        report_dict = results['classification_report']
        
        print(f"\n{'':12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")

        # 各クラス
        for i in range(self.num_classes):
            task_name = f'Task{i+1}'
            metrics = report_dict[task_name]
            
            precision = metrics['precision'] * 100
            recall = metrics['recall'] * 100
            f1 = metrics['f1-score'] * 100
            support = metrics['support']
            
            print(f"{task_name:12} {precision:11.2f}% {recall:11.2f}% "
                  f"{f1:11.2f}% {support:12.0f}")
        

        # 全体の平均
        accuracy = report_dict['accuracy'] * 100
        macro_avg = report_dict['macro avg']
        weighted_avg = report_dict['weighted avg']
        
        print(f"{'Accuracy':12} {accuracy:11.2f}%")
        print()
        print(f"{'Macro Avg':12} {macro_avg['precision']*100:11.2f}% "
              f"{macro_avg['recall']*100:11.2f}% {macro_avg['f1-score']*100:11.2f}%")
        print(f"{'Weighted Avg':12} {weighted_avg['precision']*100:11.2f}% "
              f"{weighted_avg['recall']*100:11.2f}% {weighted_avg['f1-score']*100:11.2f}%")
        
        # 混同行列
        print("\nConfusion Matrix")
        
        cm = results['confusion_matrix']
        task_names = [f'T{i+1}' for i in range(self.num_classes)]
        
        print("\n実際 \\ 予測", end="")
        for name in task_names:
            print(f"{name:>6}", end="")
        print()
        
        for i, row in enumerate(cm):
            print(f"    {task_names[i]}", end="")
            for val in row:
                print(f"{val:6d}", end="")
            print(f"  = {row.sum()}")
        print()
        
        # 混同パターン分析
        print("\nAnalysis of Confusion Patterns")
        
        mistakes = results['confusion_patterns']
        
        if len(mistakes) == 0:
            print("\nPerfect! No mistakes.")
        else:
            # サマリー
            total_mistakes = sum(m['count'] for m in mistakes)
            total_samples = len(results['targets'])
            correct_samples = (results['targets'] == results['predictions']).sum()
            
            print(f"\nMistake Summary:")
            print(f"  Total samples: {total_samples}")
            print(f"  Correct answers: {correct_samples}")
            print(f"  Total mistakes: {total_mistakes}")
            print(f"  Mistake types: {len(mistakes)} types")
            
            # 詳細リスト
            print(f"\nAll mistake patterns ({len(mistakes)} types):")
            print(f"{'Rank':>4} {'Pattern':20} {'Count':>8} {'Percentage':>12}")

            for rank, mistake in enumerate(mistakes, 1):
                pattern = f"Task{mistake['true_label']+1} → Task{mistake['pred_label']+1}"
                print(f"{rank:4d} {pattern:20} {mistake['count']:8d} "
                      f"{mistake['percentage']:11.1f}%")
            
            # 各クラスの間違い詳細
            print("\nMistakes by Class:")
            
            for i in range(self.num_classes):
                class_mistakes = [m for m in mistakes if m['true_label'] == i]
                
                total_class = cm[i].sum()
                correct_class = cm[i, i]
                
                if class_mistakes:
                    mistakes_count = sum(m['count'] for m in class_mistakes)
                    
                    print(f"\nTask{i+1} (Total: {total_class} samples):")
                    print(f"  ✓ Correct: {correct_class} ({correct_class/total_class*100:.1f}%)")
                    print(f"  ✗ Mistakes: {mistakes_count} ({mistakes_count/total_class*100:.1f}%)")
                    print(f"  Breakdown:")
                    
                    for m in sorted(class_mistakes, key=lambda x: x['count'], reverse=True):
                        print(f"    → Task{m['pred_label']+1}: {m['count']} ({m['percentage']:.1f}%)")
                else:
                    print(f"\nTask{i+1}: Perfect! (No mistakes)")
        
        print()