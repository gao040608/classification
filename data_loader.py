import pandas as pd

DATA_PATH = r"C:\Users\Administrator\Desktop\classification\data_structures_summary(2).xlsx"


def load_data():
    """
    加载 Excel，过滤有效标签，返回 df（含 label 列）
    """
    df = pd.read_excel(DATA_PATH)

    # 过滤有效标签
    df = df[df['Is Magnetic'].notna() & (df['Is Magnetic'] != '-') & (df['Is Magnetic'] != '')].copy()
    df['label'] = df['Is Magnetic'].map({
        'TRUE': 1, 'FALSE': 0,
        'True': 1, 'False': 0,
        True: 1, False: 0
    })
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # 过滤 pymatgen_dict 为空的行（特征提取需要）
    df = df[df['pymatgen_dict'].notna() & (df['pymatgen_dict'] != '')].copy()

    print(f"加载样本数: {len(df)}")
    print(f"标签分布: 0={sum(df['label']==0)}, 1={sum(df['label']==1)}")
    return df
