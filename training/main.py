import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os

from config import *
from utils.seed import set_all_seeds
from utils.file_utils import load_multiple_subjects
from utils.visualization import ConfusionMatrixVisualizer, TrainingHistoryPlotter
from data.emg_dataset import EMGdataset
from data.dataloader import EMGDataLoader
from data.data_split import stratified_split_with_subject_info
from models.cnn import EMGNet
from training.trainer import Trainer
from training.evaluator import Evaluator
from experiments.multiple_runs import MultipleRunsExperiment


# 複数被験者での1回実験
def main_single_run_multi_subjects(subjects=None, save_model=True):
    if subjects is None:
        subjects = SUBJECTS
    
    print(f"\nSingle Run Experiment with {len(subjects)} subject(s): {subjects}")
    
    # シード固定
    set_all_seeds(SEED)
    
    # データ読み込み（複数被験者）
    EMG, Label, SubjectIDs = load_multiple_subjects(
        subjects=subjects,
        emg_dataset_class=EMGdataset,
        get_data_path_func=get_processed_data_path,
        get_timing_path_func=get_timing_path,
        trim_duration=TRIM_DURATION,
        trim_mode=TRIM_MODE,
        window_size=WINDOW_SIZE,
        auto_balance=False,
        balance_strategy='min',
        balanced_per_rep=True,
        filter_enabled=FILTER_ENABLED,
        lowcut=LOWCUT,
        highcut=HIGHCUT
    )
    
    # データセット作成（被験者ID付き）
    full_dataset = EMGDataLoader(
        EMG,
        Label,
        DEVICE,
        subject_ids=SubjectIDs,
        random_seed=SEED,
        verbose=True
    )
    
    # データ分割（被験者+タスクで層化）
    train_dataset, test_dataset = stratified_split_with_subject_info(
        dataset=full_dataset,
        subject_ids=SubjectIDs,
        train_ratio=TRAIN_RATIO,
        random_seed=SEED,
        stratify_by_subject=STRATIFY_BY_SUBJECT,
        stratify_by_task=STRATIFY_BY_TASK,
        verbose=True
    )
    
    # DataLoader作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"\nDataLoader created")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # モデル作成
    sample_emg, _ = next(iter(train_loader))
    sig_size = sample_emg.shape
    model = EMGNet(sig_size, NUM_CLASSES).to(DEVICE)
    
    print(f"\nModel created")
    print(f"   Signal size: {sig_size}")
    

    # 訓練準備
    # 保存ディレクトリ
    if save_model:
        subjects_str = '_'.join([str(s) for s in subjects])
        save_dir = os.path.join(MODELS_DIR, f'sub{subjects_str}_single_run')
    else:
        save_dir = None
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    evaluator = Evaluator(model, DEVICE, NUM_CLASSES)
    trainer = Trainer(
        model, train_loader, criterion, optimizer, DEVICE,
        save_dir=save_dir,
        save_checkpoints=False,
        early_stopping=EARLY_STOPPING
    )
    

    # 訓練
    trainer.train(
        epochs=EPOCHS,
        test_loader=test_loader,
        evaluator=evaluator,
        checkpoint_every=SAVE_CHECKPOINT_EVERY
    )
    
    # 評価
    results = evaluator.evaluate(test_loader, verbose=True)
    
    # モデル情報保存
    if save_model and save_dir:
        trainer.model_saver.save_model_info(
            model,
            config={
                'subjects': subjects,
                'seed': SEED,
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'train_ratio': TRAIN_RATIO,
                'window_size': WINDOW_SIZE,
                'num_classes': NUM_CLASSES
            },
            results=results
        )
    
    # 可視化
    visualizer = ConfusionMatrixVisualizer(NUM_CLASSES)
    visualizer.plot_both(results['confusion_matrix'])
    
    # 訓練履歴
    history = trainer.get_history()
    plotter = TrainingHistoryPlotter()
    plotter.plot_both(history['losses'], history['accuracies'])
    
    # 保存確認
    if save_model and save_dir:
        print(f"\nModel saved to: {save_dir}")
    
    return results


# 複数回実験
def main_multiple_runs(subjects=None, n_runs=10, save_best=True):
    if subjects is None:
        subjects = SUBJECTS
    
    print(f"\nMultiple Runs Experiment ({n_runs} runs), Subjects: {subjects}")
    
    # データ読み込み（全実験共通）
    EMG, Label, SubjectIDs = load_multiple_subjects(
        subjects=subjects,
        emg_dataset_class=EMGdataset,
        get_data_path_func=get_processed_data_path,
        get_timing_path_func=get_timing_path,
        trim_duration=TRIM_DURATION,
        trim_mode=TRIM_MODE,
        window_size=WINDOW_SIZE,
        auto_balance=False,
        balance_strategy='min',
        balanced_per_rep=True,
        filter_enabled=FILTER_ENABLED,
        lowcut=LOWCUT,
        highcut=HIGHCUT
    )
    
    # シード値
    seeds = [42, 123, 456, 789, 999, 1234, 5678, 7890, 8888, 9999]
    
    # 設定
    config = {
        'subjects': subjects,
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'train_ratio': TRAIN_RATIO,
        'num_classes': NUM_CLASSES,
        'device': DEVICE,
        'stratify_by_subject': STRATIFY_BY_SUBJECT,
        'stratify_by_task': STRATIFY_BY_TASK
    }
    
    # 複数回実験
    experiment = MultipleRunsExperiment(config)
    
    all_results = experiment.run_multiple_experiments(
        n_runs=n_runs,
        seeds=seeds,
        EMG=EMG,
        Label=Label,
        SubjectIDs=SubjectIDs,
        emg_dataloader_class=EMGDataLoader,
        data_split_func=stratified_split_with_subject_info,
        model_class=EMGNet,
        trainer_class=Trainer,
        evaluator_class=Evaluator
    )
    
    # 結果保存
    subjects_str = '_'.join([str(s) for s in subjects])
    save_dir = f'results/sub{subjects_str}_runs{n_runs}'
    experiment.save_results(save_dir)
    
    # 平均混同行列の可視化
    mean_cm, std_cm = experiment.compute_average_confusion_matrix()
    
    visualizer = ConfusionMatrixVisualizer(NUM_CLASSES)
    
    # 平均混同行列
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        mean_cm,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=visualizer.task_names,
        yticklabels=visualizer.task_names,
        cbar_kws={'label': 'Average Count'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(f'Average Confusion Matrix ({n_runs} runs)', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/average_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nAll results saved to: {save_dir}")
    
    return all_results


# 異なる被験者組み合わせでの性能比較
def compare_subject_combinations():
    print("\nComparing Different Subject Combinations")
    
    # 比較する組み合わせ
    combinations = [
        [1],           # Sub1のみ
        [2],           # Sub2のみ
        [1, 2],        # Sub1+2
    ]
    
    results_summary = []
    
    for subjects in combinations:
        print(f"\nTesting with subjects: {subjects}")
        
        try:
            results = main_single_run_multi_subjects(subjects=subjects, save_model=False)
            
            results_summary.append({
                'subjects': subjects,
                'n_subjects': len(subjects),
                'accuracy': results['accuracy'],
                'f1': results['f1']
            })
        except Exception as e:
            print(f"Error with subjects {subjects}: {e}")
            continue
    
    # サマリー表示
    print(f"\nComparison Summary")
    print(f"{'Subjects':20} {'N':>3} {'Accuracy':>10} {'F1-Score':>10}")
    
    for r in results_summary:
        subjects_str = ', '.join([f'Sub{s}' for s in r['subjects']])
        print(f"{subjects_str:20} {r['n_subjects']:3d} "
              f"{r['accuracy']:9.2f}% {r['f1']:9.2f}%")
    
    print()
    
    return results_summary


if __name__ == '__main__':

    # 単一被験者で1回実験
    # results = main_single_run_multi_subjects(subjects=[1])
    
    # 複数被験者で1回実験
    # results = main_single_run_multi_subjects(subjects=[1, 2])
    
    # config.pyの設定を使用（1回実験）
    results = main_single_run_multi_subjects()  # SUBJECTS = [2]がデフォルト
    
    # 10回実験
    # all_results = main_multiple_runs(subjects=[1, 2, 3], n_runs=10)
    
    # 複数組み合わせの比較
    # results_comparison = compare_subject_combinations()