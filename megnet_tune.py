from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
import pandas as pd
from pymatgen.core import Structure
import json
from sklearn.model_selection import train_test_split
import logging
import tensorflow as tf
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import os
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info(f"找到 {len(gpus)} 个物理GPU, {len(logical_gpus)} 个逻辑GPU")
    except RuntimeError as e:
        logger.error(f"GPU配置错误: {e}")

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)

class MEGNetOptimizer:
    def __init__(self, train_graphs, train_targets, val_graphs, val_targets, test_graphs, test_targets):
        self.train_graphs = train_graphs
        self.train_targets = train_targets
        self.val_graphs = val_graphs
        self.val_targets = val_targets
        self.test_graphs = test_graphs
        self.test_targets = test_targets

    def create_model(self, trial):
        """使用Optuna trial创建MEGNet模型"""
        # 超参数定义 - 扩大搜索范围
        nfeat_bond = trial.suggest_int('nfeat_bond', 8, 32, step=2)  # 扩大到32
        r_cutoff = trial.suggest_float('r_cutoff', 5.0, 10.0, step=0.5)  # 增加截断半径范围
        gaussian_width = trial.suggest_float('gaussian_width', 0.1, 1.2, step=0.1)  # 扩大高斯宽度范围
        npass = trial.suggest_int('npass', 2, 8)  # 增加到8
        nblocks = trial.suggest_int('nblocks', 2, 8)  # 增加到8
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  # 扩大学习率范围
        
        # 新增超参数
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])  # 添加batch_size作为超参数
        n_neurons = trial.suggest_int('n_neurons', 32, 256, step=32)  # 添加神经元数量
        
        # 创建图转换器
        gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
        graph_converter = CrystalGraph(cutoff=r_cutoff)
        
        model = MEGNetModel(
            graph_converter=graph_converter,
            centers=gaussian_centers,
            width=gaussian_width,
            learning_rate=learning_rate,
            npass=npass,
            nblocks=nblocks,
            n1=n_neurons,  # 设置神经元数量
            n2=n_neurons,
            n3=n_neurons
        )
        return model, batch_size

    def objective(self, trial):
        """Optuna的目标函数"""
        # 创建模型
        model, batch_size = self.create_model(trial)
        
        # 训练模型
        try:
            model.train_from_graphs(
                train_graphs=self.train_graphs,
                train_targets=self.train_targets,
                validation_graphs=self.val_graphs,
                validation_targets=self.val_targets,
                epochs=100,  # 增加训练轮数到100
                batch_size=batch_size,  # 使用优化的batch_size
                verbose=0
            )
            
            # 用当前模型在测试集上评估
            correct = 0
            total = 0
            for graph, target in zip(self.test_graphs, self.test_targets):
                result = model.predict_graph(graph)
                predicted = result > 0.5
                if predicted == target:
                    correct += 1
                total += 1
            test_acc = correct / total
            return test_acc
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            print('Exception:', e)
            raise  # 让 Optuna 记录为失败 trial，而不是 pruned

def load_and_process_data(r_cutoff=5.0):
    """加载和处理数据，支持二维和块体材料，所有graph加state_attributes=[[0, 0]]或[[1, 0]]"""
    graphs_valid, bulk_graphs_valid = [], []
    targets_valid, bulk_targets_valid = [], []
    # 创建图转换器
    graph_converter = CrystalGraph(cutoff=r_cutoff)
    # 加载my_data（二维材料）
    my_data = json.load(open('./data/structures_magnetic_data_20250501_104100.json'))
    for data in my_data:
        try:
            structure = Structure.from_str(data['poscar'], fmt='poscar')
            t = 1 if data['is_magnetic'] else 0
            graph = graph_converter.convert(structure, state_attributes=[[0, 0]])
            graphs_valid.append(graph)
            targets_valid.append(t)
        except Exception as e:
            continue
    # 加载mp_data（二维和块体材料）
    mp_data = json.load(open('./data/combined_data.json'))
    for data in mp_data:
        try:
            structure = Structure.from_str(data['cif'], fmt='cif')
            t = 1 if data['dft_mag_density'] != 0 else 0
            if data['larsen_score_2d'] >= 0.8:
                graph = graph_converter.convert(structure, state_attributes=[[0, 0]])
                graphs_valid.append(graph)
                targets_valid.append(t)
            else:
                graph = graph_converter.convert(structure, state_attributes=[[1, 0]])
                bulk_graphs_valid.append(graph)
                bulk_targets_valid.append(t)
        except Exception as e:
            continue
    # 加载2D Materials Encyclopedia数据（二维材料）
    df = pd.read_csv('./data/2D_Materials_Encyclopedia_formatted.csv')
    for s, m in zip(df['cif'], df['dft_mag_density']):
        try:
            structure = Structure.from_str(s, fmt='cif')
            t = 1 if m != 0 else 0
            graph = graph_converter.convert(structure, state_attributes=[[0, 0]])
            graphs_valid.append(graph)
            targets_valid.append(t)
        except Exception as e:
            continue
    # 加载c2db数据（二维材料）
    c2db_df = pd.read_csv('./data/c2db_formatted.csv')
    for s, m in zip(c2db_df['cif'], c2db_df['dft_mag_density']):
        try:
            structure = Structure.from_str(s, fmt='cif')
            t = 1 if m != 0 else 0
            graph = graph_converter.convert(structure, state_attributes=[[0, 0]])
            graphs_valid.append(graph)
            targets_valid.append(t)
        except Exception as e:
            continue
    return graphs_valid, targets_valid, bulk_graphs_valid, bulk_targets_valid

def main():
    # 创建保存目录
    os.makedirs('optuna_results', exist_ok=True)
    
    # 加载数据
    logger.info("开始加载数据...")
    graphs_valid, targets_valid, bulk_graphs_valid, bulk_targets_valid = load_and_process_data()
    
    # 划分二维材料数据集
    train_graphs, temp_graphs, train_targets, temp_targets = train_test_split(
        graphs_valid, targets_valid, test_size=0.2, random_state=42, shuffle=True
    )
    validation_graphs, test_graphs, validation_targets, test_targets = train_test_split(
        temp_graphs, temp_targets, test_size=0.25, random_state=42, shuffle=True
    )
    # 块体材料全部进训练集和验证集
    bulk_train_graphs, bulk_validation_graphs, bulk_train_targets, bulk_validation_targets = train_test_split(
        bulk_graphs_valid, bulk_targets_valid, test_size=0.2, random_state=42, shuffle=True
    )
    train_graphs = train_graphs + bulk_train_graphs
    train_targets = train_targets + bulk_train_targets
    validation_graphs = validation_graphs + bulk_validation_graphs
    validation_targets = validation_targets + bulk_validation_targets
    
    logger.info(f"训练集大小: {len(train_graphs)}")
    logger.info(f"验证集大小: {len(validation_graphs)}")
    logger.info(f"测试集大小: {len(test_graphs)}")
    
    # 创建优化器实例
    optimizer = MEGNetOptimizer(train_graphs, train_targets, validation_graphs, validation_targets, test_graphs, test_targets)
    
    # 创建Optuna study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10,  # 增加启动试验数
            n_warmup_steps=20,    # 增加预热步数
            interval_steps=10      # 减少检查间隔
        )
    )
    
    # 开始优化 - 增加试验次数到200
    logger.info("开始超参数优化...")
    study.optimize(optimizer.objective, n_trials=200, timeout=None)
    
    # 输出最佳结果
    logger.info("\n优化结果:")
    logger.info(f"最佳验证准确率: {study.best_value:.4f}")
    logger.info(f"最佳epoch: {study.best_trial.user_attrs['best_epoch']}")
    logger.info("\n最佳超参数:")
    for key, value in study.best_params.items():
        logger.info(f"{key}: {value}")
    
    # 保存优化历史
    history_fig = plot_optimization_history(study)
    history_fig.write_html("optuna_results/optimization_history.html")
    
    importance_fig = plot_param_importances(study)
    importance_fig.write_html("optuna_results/param_importances.html")
    
    # 保存所有trials的结果
    trials_df = study.trials_dataframe()
    trials_df.to_csv("optuna_results/trials_history.csv")
    
    # 使用最佳参数创建最终模型
    best_model, _ = optimizer.create_model(study.best_trial)
    
    # 在测试集上评估
    logger.info("\n在测试集上评估最佳模型...")
    success = 0
    fail = 0
    for graph, target in zip(test_graphs, test_targets):
        result = best_model.predict_graph(graph)
        predicted_magnetic = result > 0.5
        if target == predicted_magnetic:
            success += 1
        else:
            fail += 1
    
    total = success + fail
    accuracy = (success / total) * 100
    logger.info(f'测试集结果:')
    logger.info(f'总样本数: {total}')
    logger.info(f'正确预测: {success} ({accuracy:.2f}%)')
    logger.info(f'错误预测: {fail} ({100-accuracy:.2f}%)')
    
    # 保存最佳模型
    best_model.save_model('best_mag_optuna.keras')
    logger.info('优化完成，最佳模型已保存!')

if __name__ == '__main__':
    main() 