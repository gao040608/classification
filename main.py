"""
磁性材料分类 - 主入口
cd desktop/classification
用法:
    python main.py                    # 默认 XGBoost（8维非泄露特征）
    python main.py --use_leaky true   # 开启泄露特征
    python main.py --no_pymatgen       # 仅使用原始9个特征
    python main.py --use_matminer      # 使用matminer自动提取140个特征
"""
import argparse
from data_loader import load_data
from models import run_xgboost


def main():
    parser = argparse.ArgumentParser(description="磁性材料分类")
    parser.add_argument('--use_leaky', type=str, default='false',
                        help='是否包含泄露特征: true 或 false')
    parser.add_argument('--no_pymatgen', action='store_true',
                        help='仅使用原始9个基础特征，不提取pymatgen深度特征')
    parser.add_argument('--use_matminer', action='store_true',
                        help='使用matminer自动提取140个特征')

    args = parser.parse_args()
    use_leaky = args.use_leaky.lower() in ('true', '1', 'yes')
    use_pymatgen = not args.no_pymatgen
    use_matminer = args.use_matminer

    print(f"泄露特征: {use_leaky} | pymatgen特征: {use_pymatgen} | matminer特征: {use_matminer}")

    df = load_data()
    run_xgboost(df, use_leaky=use_leaky, use_pymatgen=use_pymatgen, use_matminer=use_matminer)

    print("\n" + "=" * 60)
    print("XGBoost 训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
