import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
import xgboost as xgb

from feature_engineering import build_features, _get_pymatgen_df

RANDOM_STATE = 42
N_FOLDS = 5


def get_best_model():
    """返回表现最好的单一模型（XGBoost）"""
    return xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        colsample_bytree=0.8, reg_alpha=0.1,
        random_state=RANDOM_STATE, eval_metric='logloss'
    )


def run_xgboost(df, use_leaky=False, use_pymatgen=True, use_matminer=False):
    """
    XGBoost 表格模型 - 全数据 5 折交叉验证
    """
    print("\n" + "=" * 60)
    print("第一步：XGBoost 表格模型（全数据 5 折 CV）")
    print("=" * 60)

    # 预提取 pymatgen 深度特征（仅在需要时）
    pymatgen_df = _get_pymatgen_df(df) if use_pymatgen else None
    matminer_df = None
    if use_matminer:
        from feature_engineering import extract_matminer_features
        matminer_df = extract_matminer_features(df)

    X, y, feature_cols = build_features(df, pymatgen_df, matminer_df,
                                         use_leaky=use_leaky,
                                         use_pymatgen=use_pymatgen,
                                         use_matminer=use_matminer)
    print(f"样本数: {len(X)} | 特征数: {X.shape[1]}")
    print(f"标签分布: 0={sum(y==0)}, 1={sum(y==1)}")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    all_probs, all_labels = [], []
    fold_train_metrics = []
    fold_val_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = get_best_model()
        model.fit(X_train, y_train)

        # 训练集指标
        train_preds = model.predict(X_train)
        train_probs = model.predict_proba(X_train)[:, 1]
        t_acc = accuracy_score(y_train, train_preds)
        t_prec = precision_score(y_train, train_preds)
        t_rec = recall_score(y_train, train_preds)
        t_f1 = f1_score(y_train, train_preds)
        t_auc = roc_auc_score(y_train, train_probs)

        # 验证集指标
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)
        v_acc = accuracy_score(y_val, val_preds)
        v_prec = precision_score(y_val, val_preds)
        v_rec = recall_score(y_val, val_preds)
        v_f1 = f1_score(y_val, val_preds)
        v_auc = roc_auc_score(y_val, val_probs)

        fold_train_metrics.append({'acc': t_acc, 'prec': t_prec, 'rec': t_rec, 'f1': t_f1, 'auc': t_auc})
        fold_val_metrics.append({'acc': v_acc, 'prec': v_prec, 'rec': v_rec, 'f1': v_f1, 'auc': v_auc})

        all_probs.extend(val_probs)
        all_labels.extend(y_val)

        print(f"Fold {fold_idx+1}/{N_FOLDS}: "
              f"Train Acc={t_acc:.4f} F1={t_f1:.4f} | "
              f"Val Acc={v_acc:.4f} F1={v_f1:.4f} AUC={v_auc:.4f}")

    # 五折平均指标
    avg_train_acc = np.mean([m['acc'] for m in fold_train_metrics])
    avg_train_prec = np.mean([m['prec'] for m in fold_train_metrics])
    avg_train_rec = np.mean([m['rec'] for m in fold_train_metrics])
    avg_train_f1 = np.mean([m['f1'] for m in fold_train_metrics])
    avg_train_auc = np.mean([m['auc'] for m in fold_train_metrics])
    avg_val_acc = np.mean([m['acc'] for m in fold_val_metrics])
    avg_val_prec = np.mean([m['prec'] for m in fold_val_metrics])
    avg_val_rec = np.mean([m['rec'] for m in fold_val_metrics])
    avg_val_f1 = np.mean([m['f1'] for m in fold_val_metrics])
    avg_val_auc = np.mean([m['auc'] for m in fold_val_metrics])

    print(f"\n===== XGBoost 5 折 CV 平均结果 =====")
    print(f"【训练集】 Acc={avg_train_acc:.4f}  Prec={avg_train_prec:.4f}  Rec={avg_train_rec:.4f}  F1={avg_train_f1:.4f}  AUC={avg_train_auc:.4f}")
    print(f"【验证集】 Acc={avg_val_acc:.4f}  Prec={avg_val_prec:.4f}  Rec={avg_val_rec:.4f}  F1={avg_val_f1:.4f}  AUC={avg_val_auc:.4f}")


    # 特征重要性排名（用最后一折验证集的模型）
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n===== XGBoost 特征重要性排名（Top {min(20, len(feature_cols))}）=====")
    for rank, idx in enumerate(sorted_idx[:20], 1):
        bar = '█' * int(importances[idx] * 50)
        print(f"  {rank:2d}. {feature_cols[idx]:<30s} {importances[idx]:.4f} {bar}")

    return {'probs': all_probs, 'labels': all_labels}
