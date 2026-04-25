import warnings
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from sklearn.preprocessing import LabelEncoder

def extract_pymatgen_features(pymatgen_str):
    """
    从 pymatgen_dict 提取所有原始标量特征 + 电负性统计特征
    """
    features = {
        # 晶格参数（5 个，剔除 alpha/beta）
        'lattice_a': None,
        'lattice_b': None,
        'lattice_c': None,
        'lattice_gamma': None,
        'lattice_volume': None,
        # 电负性统计（4 个，剔除 mean_X）
        'min_X': None,
        'max_X': None,
        'X_range': None,
        'X_std': None,
    }

    if pd.isna(pymatgen_str):
        return features

    try:
        import json
        d = json.loads(pymatgen_str)
        struct = Structure.from_dict(d)

        # 晶格参数（原始值，直接取）
        features['lattice_a'] = struct.lattice.a
        features['lattice_b'] = struct.lattice.b
        features['lattice_c'] = struct.lattice.c
        features['lattice_gamma'] = struct.lattice.gamma
        features['lattice_volume'] = struct.lattice.volume

        # 电负性统计（按占位分数加权）
        X_list = []  # (电负性, 占位分数)
        for site in struct:
            for sp, occu in site.species.items():
                if hasattr(sp, 'X') and sp.X is not None:
                    X_list.append((sp.X, occu))

        if X_list:
            total_occu = sum(occu for _, occu in X_list)
            weights = [occu / total_occu for _, occu in X_list]
            electronegativities = [x for x, _ in X_list]

            features['min_X'] = min(electronegativities)
            features['max_X'] = max(electronegativities)
            features['X_range'] = features['max_X'] - features['min_X']
            mean_X = sum(w * x for w, (_, x) in zip(weights, X_list))
            if len(X_list) > 1:
                features['X_std'] = (sum(w * (x - mean_X)**2
                                      for w, (_, x) in zip(weights, X_list))) ** 0.5
            else:
                features['X_std'] = 0.0

    except Exception:
        pass

    return features

# ============================================================
# matminer 一键特征提取（优化版：分批 + 只提取 Top-5 特征）
# ============================================================
# Top-5 Magpie 特征（根据 XGBoost 特征重要性）
TOP5_MAGPIE_FEATURES = [
    "mean NpValence",
    "maximum SpaceGroupNumber", 
    "range Column",
    "maximum GSmagmom",
    "avg_dev GSmagmom"
]

def extract_matminer_features(df, batch_size=500, use_top5_only=True):
    """
    使用 matminer 自动提取特征（优化版）：
    - 分批处理，避免 MemoryError
    - 默认只提取 Top-5 Magpie 特征 + spacegroup_number，大幅降低内存和 CPU 占用
    
    参数:
        df: 输入 DataFrame
        batch_size: 每批处理的样本数，默认 500
        use_top5_only: 是否只提取 Top-5 特征（默认 True）
    """
    from matminer.featurizers.composition import ElementProperty

    print(f"使用 matminer 提取特征（Top-5 Magpie + spacegroup_number）...")

    # 分批解析 Structure 对象（边解析边处理，不全部缓存）
    import json
    n_samples = len(df)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    print(f"  总样本数: {n_samples} | 分批大小: {batch_size} | 批次数: {n_batches}")

    # 初始化 featurizer：3 种属性 × 4 种统计量 = 12 个特征
    elem_prop_feat = ElementProperty(
        data_source="magpie",
        features=["NpValence", "Column", "GSmagmom"],
        stats=["mean", "maximum", "range", "avg_dev"]
    )

    all_magpie_results = []
    all_sg_results = []
    all_indices = []

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        
        # 解析当前批次的 Structure
        batch_structs = []
        batch_valid_indices = []
        
        for i in range(start_idx, end_idx):
            s = df['pymatgen_dict'].iloc[i]
            if pd.isna(s):
                continue
            try:
                d = json.loads(s)
                struct = Structure.from_dict(d)
                batch_structs.append(struct)
                batch_valid_indices.append(i)
            except:
                continue
        
        if not batch_structs:
            continue

        # 提取当前批次的特征
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compositions = [s.composition for s in batch_structs]
            
            # 1. Magpie 组成特征（12 个）
            elem_prop_feat.fit(compositions)
            batch_magpie = elem_prop_feat.featurize_many(compositions, ignore_errors=True)
            
            # 2. 空间群号（直接从 Structure 对象取，只保留 1 个特征）
            batch_sg = np.array([[s.spacegroup.number if hasattr(s, 'spacegroup') and s.spacegroup else np.nan 
                                  for s in batch_structs]]).T

        all_magpie_results.append(batch_magpie)
        all_sg_results.append(batch_sg)
        all_indices.extend(batch_valid_indices)
        
        print(f"  批次 {batch_idx + 1}/{n_batches}: 处理了 {len(batch_structs)} 个样本")

    # 合并所有批次结果
    magpie_arr = np.vstack(all_magpie_results) if all_magpie_results else np.array([]).reshape(0, 12)
    sg_arr = np.vstack(all_sg_results) if all_sg_results else np.array([]).reshape(0, 1)
    feature_array = np.hstack([magpie_arr, sg_arr])

    magpie_labels = elem_prop_feat.feature_labels()
    feature_labels = magpie_labels + ["spacegroup_number"]
    print(f"  提取完成: {len(feature_labels)} 个特征（Magpie: {len(magpie_labels)}, spacegroup: 1）")

    # 构建 DataFrame
    matminer_df = pd.DataFrame(feature_array, columns=feature_labels, index=all_indices)
    matminer_df = matminer_df.reindex(range(len(df)))

    # 数值化 + 中位数填充
    for col in matminer_df.columns:
        matminer_df[col] = pd.to_numeric(matminer_df[col], errors='coerce')
        matminer_df[col] = matminer_df[col].fillna(matminer_df[col].median())

    return matminer_df


def build_features(df, pymatgen_df=None, matminer_df=None,
                   use_leaky=False, use_pymatgen=True, use_matminer=False):
    """
    特征工程，生成特征列 + 特征列表

    参数:
        df: 原始 DataFrame
        pymatgen_df: extract_pymatgen_features() 生成的特征 DataFrame
        matminer_df: extract_matminer_features() 生成的特征 DataFrame
        use_leaky: 是否包含泄露特征
        use_pymatgen: 是否使用 pymatgen 手动提取特征
        use_matminer: 是否使用 matminer 自动提取特征
    """
    # 若未传入 pymatgen_df，且需要使用，则自动提取
    if use_pymatgen and pymatgen_df is None:
        pymatgen_df = _get_pymatgen_df(df)

    # 若未传入 matminer_df，且需要使用，则自动提取
    if use_matminer and matminer_df is None:
        matminer_df = extract_matminer_features(df)

    # ========== 5 个非泄露原始特征 ==========
    base_num_cols = ['Density (g/cm³)', 'Volume', 'nsites', 'Number of Elements', 'Electronegativity']
    for col in base_num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # ========== 泄露特征（仅 use_leaky 时加入） ==========
    df['ordering'] = pd.to_numeric(df['ordering'], errors='coerce')
    le_ord = LabelEncoder()
    df['ordering_enc'] = le_ord.fit_transform(df['ordering'].fillna('unknown').astype(str))
    df['num_mag_sites'] = pd.to_numeric(df['Number of Magnetic Sites'], errors='coerce')
    df['total_mag'] = pd.to_numeric(df['Total Magnetization'], errors='coerce')

    if use_leaky:
        for col in ['num_mag_sites', 'total_mag']:
            df[col] = df[col].fillna(df[col].median())

    # 仅用原始特征时，跳过 pymatgen 处理
    if not use_pymatgen and not use_matminer:
        feature_cols = base_num_cols
        X = df[feature_cols].values
        y = df['label'].values
        print(f"当前模式: 仅原始特征（不含泄露）")
        print(f"特征数: {len(feature_cols)} | 特征列表: {feature_cols}")
        print(f"特征矩阵: {X.shape} | 标签分布: {np.bincount(y)}")
        return X, y, feature_cols

    # ========== pymatgen 手动特征处理 ==========
    feature_cols = list(base_num_cols)

    if use_pymatgen:
        pymatgen_feature_cols = [
            'lattice_a', 'lattice_b', 'lattice_c',
            'lattice_gamma', 'lattice_volume',
            'min_X', 'max_X', 'X_range', 'X_std',
        ]
        for col in pymatgen_feature_cols:
            pymatgen_df[col] = pd.to_numeric(pymatgen_df[col], errors='coerce')
            pymatgen_df[col] = pymatgen_df[col].fillna(pymatgen_df[col].median())
        feature_cols += pymatgen_feature_cols
    else:
        pymatgen_feature_cols = []

    # ========== matminer 自动特征 ==========
    if use_matminer:
        matminer_feature_cols = list(matminer_df.columns)
        feature_cols += matminer_feature_cols
    else:
        matminer_feature_cols = []

    if use_leaky:
        leaky_cols = ['ordering_enc', 'num_mag_sites', 'total_mag']
        feature_cols = feature_cols + leaky_cols

    # 从 df / pymatgen_df / matminer_df 合并选取特征
    parts = [df[base_num_cols].reset_index(drop=True)]

    if use_pymatgen and pymatgen_feature_cols:
        parts.append(pymatgen_df[pymatgen_feature_cols].reset_index(drop=True))

    if use_matminer and matminer_feature_cols:
        parts.append(matminer_df[matminer_feature_cols].reset_index(drop=True))

    combined = pd.concat(parts, axis=1)

    X = combined[feature_cols].values
    y = df['label'].values

    mode_desc = []
    if use_pymatgen:
        mode_desc.append(f"pymatgen({len(pymatgen_feature_cols)})")
    if use_matminer:
        mode_desc.append(f"matminer({len(matminer_feature_cols)})")
    mode_str = " + ".join(mode_desc) if mode_desc else "仅原始"

    print(f"当前模式: {mode_str} | {'含泄露' if use_leaky else '不含泄露'}")
    print(f"特征数: {len(feature_cols)} | 特征矩阵: {X.shape} | 标签分布: {np.bincount(y)}")

    return X, y, feature_cols


def _get_pymatgen_df(df):
    """预提取全量 pymatgen 特征，避免重复解析"""
    print("提取 pymatgen 深度特征...")
    pymatgen_features = df['pymatgen_dict'].apply(extract_pymatgen_features)
    pymatgen_df = pd.DataFrame(pymatgen_features.tolist())
    print(f"  → 完成，共 {pymatgen_df.shape[1]} 个结构特征")
    return pymatgen_df
