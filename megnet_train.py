from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
import numpy as np
import pandas as pd
from pymatgen.core import Structure
import json
from sklearn.model_selection import train_test_split
import logging
import tensorflow as tf

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
config={'nfeat_bond': 22, 'r_cutoff': 5.0, 'gaussian_width': 0.7000000000000001, 'npass': 4, 'nblocks': 5, 'learning_rate': 0.004694832362498996, 'batch_size': 128, 'n_neurons': 96}
# 模型参数
nfeat_bond = config['nfeat_bond']
r_cutoff = config['r_cutoff']
gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
gaussian_width = config['gaussian_width']
graph_converter = CrystalGraph(cutoff=r_cutoff)

# 创建模型
model = MEGNetModel(
    graph_converter=graph_converter,
    centers=gaussian_centers,
    width=gaussian_width,
    learning_rate=config['learning_rate'],
    npass=config['npass'],  # 减少 Set2Set 的迭代次数
    nblocks=config['nblocks'],  # 设置卷积块数量
    n1=config['n_neurons'],
    n2=config['n_neurons'],
    n3=config['n_neurons']
)

structure_list=[]
df=pd.read_csv('./data/2D_Materials_Encyclopedia_formatted.csv')
cif_list=df['cif'].to_list()
dft_mag_list=df['dft_mag_density'].to_list()
c2db_df=pd.read_csv('./data/c2db_formatted.csv')
c2db_cif_list=c2db_df['cif'].to_list()
c2db_dft_mag_list=c2db_df['dft_mag_density'].to_list()
my_data=json.load(open('./data/structures_magnetic_data_20250501_104100.json'))
mp_data=json.load(open('./data/combined_data.json'))
graphs_valid ,bulk_graphs_valid= [],[]
targets_valid ,bulk_targets_valid= [],[]
structures_invalid = []

# 定义磁性元素列表
magnetic_elements = ['Cr', 'Mn', 'Fe', 'Co', 'Ni']

def count_magnetic_atoms(structure):
    comp = structure.composition
    return sum(comp[el] for el in comp.elements if el.symbol in magnetic_elements)

def get_state(is_2d, structure):
    """返回state_attributes: [0/1, n_magnetic_atoms]"""
    n_magnetic_atoms = count_magnetic_atoms(structure)
    return [[0, n_magnetic_atoms]] if is_2d else [[1, n_magnetic_atoms]]

for data in my_data:
    structure=Structure.from_str(data['poscar'],fmt='poscar')
    structure_list.append(structure)
    if data['is_magnetic']:
        t=1
    else:
        t=0
    state = get_state(True, structure)
    try:
        graph = model.graph_converter.convert(structure=structure,state_attributes=state)
        graphs_valid.append(graph)
        targets_valid.append(t)
    except:
        structures_invalid.append(structure)
for data in mp_data:
    structure=Structure.from_str(data['cif'],fmt='cif')
    structure_list.append(structure)
    if data['dft_mag_density']!=0:
        t=1
    else:
        t=0
    if data['larsen_score_2d']>=0.8:
        state = get_state(True, structure)
        try:
            graph = model.graph_converter.convert(structure=structure,state_attributes=state)
            graphs_valid.append(graph)
            targets_valid.append(t)
        except:
            structures_invalid.append(structure)
    else:
        state = get_state(False, structure)
        try:
            graph = model.graph_converter.convert(structure=structure,state_attributes=state)
            bulk_graphs_valid.append(graph)
            bulk_targets_valid.append(t)
        except:
            structures_invalid.append(structure)
for s, p in zip(cif_list, dft_mag_list):
    try:
        structure=Structure.from_str(s,fmt='cif')
        if p!=0:
            t=1
        else:
            t=0
        state = get_state(True, structure)
        graph = model.graph_converter.convert(structure=structure,state_attributes=state)
        graphs_valid.append(graph)
        targets_valid.append(t)
    except:
        structures_invalid.append(s)
for s, p in zip(c2db_cif_list, c2db_dft_mag_list):
    try:
        structure=Structure.from_str(s,fmt='cif')
        if p!=0:
            t=1
        else:
            t=0
        state = get_state(True, structure)
        graph = model.graph_converter.convert(structure=structure,state_attributes=state)
        graphs_valid.append(graph)
        targets_valid.append(t)
    except:
        structures_invalid.append(s)

# 使用 train_test_split 划分数据
# 首先将数据分为训练集(80%)和临时集(20%)
train_graphs, temp_graphs, train_targets, temp_targets = train_test_split(
    graphs_valid,
    targets_valid,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 将临时集再分为验证集(15%)和测试集(5%)
validation_graphs, test_graphs, validation_targets, test_targets = train_test_split(
    temp_graphs,
    temp_targets,
    test_size=0.25,  # 0.05/0.2 = 0.25
    random_state=42,
    shuffle=True
)
bulk_train_graphs, bulk_validation_graphs, bulk_train_targets, bulk_validation_targets = train_test_split(
    bulk_graphs_valid,
    bulk_targets_valid,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
train_graphs = train_graphs + bulk_train_graphs
train_targets = train_targets + bulk_train_targets
validation_graphs = validation_graphs + bulk_validation_graphs
validation_targets = validation_targets + bulk_validation_targets
# 打印数据集大小
logger.info(f"训练集大小: {len(train_graphs)}")
logger.info(f"验证集大小: {len(validation_graphs)}")

# 创建回调函数
callbacks = [
    # 模型检查点
    tf.keras.callbacks.ModelCheckpoint(
        filepath='callback/model_epoch_{epoch:03d}_loss_{loss:.4f}.hdf5',
        monitor='loss',
        mode='min',
        save_best_only=True,
        verbose=1
    ),
    # 早停
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=100,
        restore_best_weights=True,
        verbose=1
    ),
    # 学习率调度器
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.7,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
]

epoch_list = [100]
results = []

for epochs in epoch_list:
    print(f"\n开始训练 {epochs} epochs ...")
    model.train_from_graphs(
        train_graphs=train_graphs,
        train_targets=train_targets,
        validation_graphs=validation_graphs,
        validation_targets=validation_targets,
        batch_size=config['batch_size'],
        epochs=epochs,
        verbose=1
    )

    # 测试集评估
    success = 0
    fail = 0
    for graph, target in zip(test_graphs, test_targets):
        result = model.predict_graph(graph)
        predicted_magnetic = result > 0.5
        if target == predicted_magnetic:
            success += 1
        else:
            fail += 1
    total = success + fail
    accuracy = (success / total) * 100
    print(f"epochs={epochs} 测试集准确率: {accuracy:.2f}%，正确预测: {success}，错误预测: {fail}")
    results.append((epochs, accuracy))

# 保存最终模型
model.save_model('magnew.keras')
logger.info('训练完成!')
