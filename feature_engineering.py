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
# matminer 一键特征提取
# ============================================================
def extract_matminer_features(df):
    """
    使用 matminer 自动提取 140 个特征：
    - DensityFeatures: 3 个（密度、每原子体积、堆积分数）
    - GlobalSymmetryFeatures: 5 个（空间群号、晶系、是否中心对称、对称操作数）
    - ElementProperty(magpie): 132 个（22 种元素属性 × 6 种统计量）
    """
    from matminer.featurizers.structure import GlobalSymmetryFeatures, DensityFeatures
    from matminer.featurizers.composition import ElementProperty

    print("使用 matminer 自动提取特征（140 个）...")

    # 解析所有 Structure 对象
    import json
    structures = []
    for s in df['pymatgen_dict']:
        if pd.isna(s):
            structures.append(None)
        else:
            try:
                d = json.loads(s)
                structures.append(Structure.from_dict(d))
            except:
                structures.append(None)

    # 构建 featurizer
    # 注意：ElementProperty.from_preset("magpie") 需要 Composition 对象，
    # pymatgen 新版本中 Structure 没有 element_composition 属性，
    # 因此单独对 composition 做特征提取
    density_feat = DensityFeatures()
    symmetry_feat = GlobalSymmetryFeatures()
    elem_prop_feat = ElementProperty.from_preset("magpie")

    # 对每个结构提取特征
    valid_indices = [i for i, s in enumerate(structures) if s is not None]
    valid_structs = [structures[i] for i in valid_indices]

    print(f"  有效结构数: {len(valid_structs)} / {len(df)}")
    print(f"  正在提取特征，请稍候...")

    # 分批提取：Structure 特征 + Composition 特征
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # 1. 结构特征（DensityFeatures + GlobalSymmetryFeatures）
        density_feat.fit(valid_structs)
        symmetry_feat.fit(valid_structs)
        density_arr  = density_feat.featurize_many(valid_structs, ignore_errors=True)
        symmetry_arr = symmetry_feat.featurize_many(valid_structs, ignore_errors=True)

        # 2. 组成特征（ElementProperty 需要 Composition）
        compositions = [s.composition for s in valid_structs]
        elem_prop_feat.fit(compositions)
        elem_arr = elem_prop_feat.featurize_many(compositions, ignore_errors=True)

    # 合并三个 featurizer 的结果
    feature_array = np.hstack([density_arr, symmetry_arr, elem_arr])

    # 获取特征名（三个 featurizer 拼接）
    feature_labels = (
        density_feat.feature_labels()
        + symmetry_feat.feature_labels()
        + elem_prop_feat.feature_labels()
    )
    print(f"  提取完成: {len(feature_labels)} 个 matminer 特征")

    # 构建 DataFrame
    matminer_df = pd.DataFrame(feature_array, columns=feature_labels, index=valid_indices)

    # 对无效行填充 NaN
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
