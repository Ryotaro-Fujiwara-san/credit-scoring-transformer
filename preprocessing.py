import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def aggregate_bureau_data(bureau_df):
    """
    子テーブル(bureau)をSK_ID_CURRごとに集約する
    """
    # 集約ルールの定義
    agg_dict = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'CREDIT_ACTIVE': ['count'] # 借入件数など
    }
    
    # グルーピングと集約
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg(agg_dict)
    
    # カラム名のフラット化 (例: DAYS_CREDIT_min)
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns.values]
    bureau_agg.reset_index(inplace=True)
    
    return bureau_agg

def preprocess_main_data(df):
    """
    数値データの正規化とカテゴリデータのエンコーディング
    """
    # 欠損値処理（簡易版：中央値埋め。実務ではより高度な手法を検討）
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # カテゴリカル変数のLabel Encoding
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 数値変数の正規化 (Transformerには必須)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [c for c in num_cols if c not in ['TARGET', 'SK_ID_CURR']]
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df, num_cols, cat_cols