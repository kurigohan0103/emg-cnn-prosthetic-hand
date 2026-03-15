import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os


# 複数回実験を管理するクラス
class MultipleRunsExperiment:

    def __init__(self, config):
        self.config = config
        self.all_results = []
        self.best_run_idx = None
        self.best_accuracy = 0.0
    
    # 1回分の実験
    def run_single_experiment(self, seed, EMG, Label, SubjectIDs, emg_dataloader_class, data_split_func, model_class, trainer_class, evaluator_class):
      
        from utils.seed import set_all_seeds
        
        print(f"\nExperiment with seed={seed}")
        
        # シード固定
        set_all_seeds(seed)
        
        # デバイス
        device = self.config['device']
        
        # データセット作成
        full_dataset = emg_dataloader_class(
            EMG,
            Label,
            device,
            subject_ids=SubjectIDs,
            random_seed=seed,
            verbose=False
        )
        
        # データ分割
        train_dataset, test_dataset = data_split_func(
            dataset=full_dataset,
            subject_ids=SubjectIDs,
            train_ratio=self.config['train_ratio'],
            random_seed=seed,
            stratify_by_subject=self.config.get('stratify_by_subject', True),
            stratify_by_task=self.config.get('stratify_by_task', True),
            verbose=False
        )
        
        # DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"Data prepared: Train={len(train_dataset)}, Test={len(test_dataset)}")
        
        # モデル作成
        sample_emg, _ = next(iter(train_loader))
        sig_size = sample_emg.shape
        
        model = model_class(sig_size, self.config['num_classes']).to(device)
        
        # 訓練準備
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        evaluator = evaluator_class(model, device, self.config['num_classes'])
        trainer = trainer_class(model, train_loader, criterion, optimizer, device)
        
        # 訓練
        print(f"\nTraining started...")
        trainer.train(
            epochs=self.config['epochs'],
            test_loader=test_loader,
            evaluator=evaluator
        )
        
        # 評価
        print(f"\nEvaluation...")
        results = evaluator.evaluate(test_loader, verbose=False)
        
        # シード情報を追加
        results['seed'] = seed
        
        print(f"\nSeed={seed} completed")
        print(f"   Accuracy: {results['accuracy']:.2f}%")
        print(f"   F1-Score: {results['f1']:.2f}%")
        
        # ベストモデル追跡
        if results['accuracy'] > self.best_accuracy:
            self.best_accuracy = results['accuracy']
            self.best_run_idx = len(self.all_results)
            print(f"   [Best] New best model!")
        
        # メモリ解放
        del model, trainer, evaluator, train_loader, test_loader
        del train_dataset, test_dataset, full_dataset
        torch.cuda.empty_cache()
        
        return results
    
    # 複数回実験を実行
    def run_multiple_experiments(self, n_runs, seeds, EMG, Label, SubjectIDs, emg_dataloader_class, data_split_func, model_class, trainer_class, evaluator_class):
       
        print(f"\nMultiple Runs Experiment ({n_runs} runs), Seeds: {seeds[:n_runs]}")
        
        self.all_results = []
        
        for i, seed in enumerate(seeds[:n_runs], 1):
            print(f"\nRun {i}/{n_runs}")
            
            result = self.run_single_experiment(
                seed=seed,
                EMG=EMG,
                Label=Label,
                SubjectIDs=SubjectIDs,
                emg_dataloader_class=emg_dataloader_class,
                data_split_func=data_split_func,
                model_class=model_class,
                trainer_class=trainer_class,
                evaluator_class=evaluator_class
            )
            
            self.all_results.append(result)
        
        print(f"\nAll {n_runs} experiments completed!")
        
        # サマリー表示
        self.print_summary()
        
        return self.all_results
    
    # 結果のサマリーを表示
    def print_summary(self):
        print(f"\nSummary Statistics")
        
        # 各指標を集計
        accuracies = [r['accuracy'] for r in self.all_results]
        precisions = [r['precision'] for r in self.all_results]
        recalls = [r['recall'] for r in self.all_results]
        f1_scores = [r['f1'] for r in self.all_results]
        
        print(f"\n{'Metric':15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        
        metrics = {
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores
        }
        
        for name, values in metrics.items():
            mean = np.mean(values)
            std = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            print(f"{name:15} {mean:9.2f}% {std:9.2f}% {min_val:9.2f}% {max_val:9.2f}%")
        
        # 各実験の詳細
        print(f"\nDetailed Results")
        print(f"{'Run':>4} {'Seed':>6} {'Accuracy':>10} {'Precision':>11} {'Recall':>10} {'F1-Score':>10}")
        
        for i, result in enumerate(self.all_results, 1):
            print(f"{i:4d} {result['seed']:6d} "
                  f"{result['accuracy']:9.2f}% "
                  f"{result['precision']:10.2f}% "
                  f"{result['recall']:9.2f}% "
                  f"{result['f1']:9.2f}%")
        
        print()
    
    # 平均混同行列を計算
    def compute_average_confusion_matrix(self):
        all_cms = np.array([r['confusion_matrix'] for r in self.all_results])
        
        mean_cm = np.mean(all_cms, axis=0)
        std_cm = np.std(all_cms, axis=0)
        
        return mean_cm, std_cm
    
    # 結果を保存
    def save_results(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        # JSON保存用のデータ整形
        save_data = {
            'config': {k: str(v) if isinstance(v, torch.device) else v 
                      for k, v in self.config.items()},
            'n_runs': len(self.all_results),
            'summary': {
                'accuracy': {
                    'mean': float(np.mean([r['accuracy'] for r in self.all_results])),
                    'std': float(np.std([r['accuracy'] for r in self.all_results])),
                    'min': float(np.min([r['accuracy'] for r in self.all_results])),
                    'max': float(np.max([r['accuracy'] for r in self.all_results]))
                },
                'f1': {
                    'mean': float(np.mean([r['f1'] for r in self.all_results])),
                    'std': float(np.std([r['f1'] for r in self.all_results])),
                    'min': float(np.min([r['f1'] for r in self.all_results])),
                    'max': float(np.max([r['f1'] for r in self.all_results]))
                }
            },
            'individual_runs': [
                {
                    'seed': r['seed'],
                    'accuracy': float(r['accuracy']),
                    'precision': float(r['precision']),
                    'recall': float(r['recall']),
                    'f1': float(r['f1'])
                }
                for r in self.all_results
            ]
        }
        
        # JSON保存
        json_path = os.path.join(save_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nResults saved: {json_path}")
        
        # 平均混同行列を保存
        mean_cm, std_cm = self.compute_average_confusion_matrix()
        np.save(os.path.join(save_dir, 'mean_confusion_matrix.npy'), mean_cm)
        np.save(os.path.join(save_dir, 'std_confusion_matrix.npy'), std_cm)
        
        print(f"Confusion matrices saved")