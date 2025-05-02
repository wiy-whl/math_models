import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from datetime import datetime, timedelta

# 列名映射
COLUMN_MAPPING = {
    '用户ID (User ID)': '用户ID',
    '用户行为 (User behaviour)': '用户行为',
    '博主ID (Blogger ID)': '博主ID',
    '时间 (Time)': '时间'
}

def convert_date(date_str):
    """将2021年的日期转换为2024年的日期"""
    date = pd.to_datetime(date_str)
    if date.year == 2021:
        return date + pd.DateOffset(years=3)
    return date

# 读取数据
def load_data():
    try:
        print("开始加载数据...")
        # 读取附件1和附件2
        df1 = pd.read_csv('附件1 (Attachment 1).csv')
        df2 = pd.read_csv('附件2 (Attachment 2).csv')
        
        # 重命名列
        df1 = df1.rename(columns=COLUMN_MAPPING)
        df2 = df2.rename(columns=COLUMN_MAPPING)
        
        # 转换时间列
        df1['时间'] = pd.to_datetime(df1['时间']).apply(convert_date)
        df2['时间'] = pd.to_datetime(df2['时间']).apply(convert_date)
        
        # 转换用户行为为数值类型
        df1['用户行为'] = pd.to_numeric(df1['用户行为'])
        df2['用户行为'] = pd.to_numeric(df2['用户行为'])
        
        print("数据加载完成")
        print(f"附件1数据量: {len(df1)}")
        print(f"附件2数据量: {len(df2)}")
        return df1, df2
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

# 构建用户特征
def create_user_features(df):
    try:
        print("\n构建用户特征...")
        # 计算用户历史活跃天数（7.11-7.20）
        start_date = datetime(2024, 7, 11)
        end_date = datetime(2024, 7, 20)
        print("筛选时间范围内的数据...")
        df_period = df[(df['时间'] >= start_date) & (df['时间'] <= end_date)]
        
        print("计算用户活跃天数...")
        user_active_days = df_period.groupby('用户ID')['时间'].agg('nunique').reset_index(name='历史活跃天数')
        
        print("计算用户行为统计...")
        # 使用向量化操作替代lambda函数
        behavior_stats = pd.DataFrame()
        behavior_stats['用户ID'] = df_period['用户ID'].unique()
        for behavior, name in [(1, '观看次数'), (2, '点赞次数'), (3, '评论次数'), (4, '关注次数')]:
            behavior_counts = df_period[df_period['用户行为'] == behavior]['用户ID'].value_counts()
            behavior_stats[name] = behavior_stats['用户ID'].map(behavior_counts).fillna(0)
        
        print("计算用户转化率...")
        behavior_stats['点赞后关注率'] = behavior_stats['关注次数'] / behavior_stats['点赞次数'].replace(0, 1)
        behavior_stats['评论后关注率'] = behavior_stats['关注次数'] / behavior_stats['评论次数'].replace(0, 1)
        
        print("合并特征...")
        user_features = user_active_days.merge(behavior_stats, on='用户ID', how='left')
        user_features = user_features.fillna(0)
        
        print("用户特征构建完成")
        return user_features
    except Exception as e:
        print(f"构建用户特征时出错: {str(e)}")
        raise

# 构建博主特征
def create_blogger_features(df):
    try:
        print("\n构建博主特征...")
        # 计算截至7月21日的博主数据
        end_date = datetime(2024, 7, 21)
        print("筛选时间范围内的数据...")
        df_period = df[df['时间'] <= end_date]
        
        print("计算博主行为统计...")
        # 使用向量化操作替代lambda函数
        blogger_stats = pd.DataFrame()
        blogger_stats['博主ID'] = df_period['博主ID'].unique()
        for behavior, name in [(1, '观看次数'), (2, '点赞次数'), (3, '评论次数'), (4, '关注次数')]:
            behavior_counts = df_period[df_period['用户行为'] == behavior]['博主ID'].value_counts()
            blogger_stats[name] = blogger_stats['博主ID'].map(behavior_counts).fillna(0)
        
        print("计算博主质量指标...")
        blogger_stats['互动率'] = (blogger_stats['点赞次数'] + blogger_stats['评论次数']) / blogger_stats['观看次数'].replace(0, 1)
        blogger_stats['粉丝增长率'] = blogger_stats['关注次数'] / blogger_stats['观看次数'].replace(0, 1)
        
        print("博主特征构建完成")
        return blogger_stats
    except Exception as e:
        print(f"构建博主特征时出错: {str(e)}")
        raise

# 构建用户-博主交互特征
def create_interaction_features(df1, df2, user_features, blogger_features):
    try:
        print("\n构建用户-博主交互特征...")
        # 获取7月22日的互动数据
        print("筛选7月22日数据...")
        interactions_722 = df2[df2['时间'].dt.date == datetime(2024, 7, 22).date()]
        
        print("创建用户-博主对...")
        user_blogger_pairs = interactions_722[['用户ID', '博主ID']].drop_duplicates()
        
        print("重命名特征列...")
        # 重命名用户特征列
        user_features_renamed = user_features.copy()
        user_features_renamed.columns = ['用户ID' if col == '用户ID' else f'用户_{col}' 
                                       for col in user_features.columns]
        
        # 重命名博主特征列
        blogger_features_renamed = blogger_features.copy()
        blogger_features_renamed.columns = ['博主ID' if col == '博主ID' else f'博主_{col}' 
                                         for col in blogger_features.columns]
        
        print("合并用户特征...")
        features = user_blogger_pairs.merge(user_features_renamed, on='用户ID', how='left')
        
        print("合并博主特征...")
        features = features.merge(blogger_features_renamed, on='博主ID', how='left')
        
        print("计算当日互动统计...")
        # 使用向量化操作计算当日统计
        daily_stats = pd.DataFrame()
        daily_stats['用户ID'] = user_blogger_pairs['用户ID']
        daily_stats['博主ID'] = user_blogger_pairs['博主ID']
        
        for behavior, name in [(1, '当日观看次数'), (2, '当日点赞次数'), (3, '当日评论次数')]:
            behavior_data = interactions_722[interactions_722['用户行为'] == behavior]
            behavior_counts = behavior_data.groupby(['用户ID', '博主ID']).size().reset_index(name=name)
            daily_stats = daily_stats.merge(behavior_counts, on=['用户ID', '博主ID'], how='left')
        
        print("合并当日互动数据...")
        features = features.merge(daily_stats, on=['用户ID', '博主ID'], how='left')
        features = features.fillna(0)
        
        print("交互特征构建完成")
        return features
    except Exception as e:
        print(f"构建交互特征时出错: {str(e)}")
        raise

# 训练模型并预测
def train_and_predict(features, df1, target_users):
    try:
        print("\n开始模型训练和预测...")
        # 准备特征
        feature_cols = ['用户_历史活跃天数', 
                       '用户_观看次数', '用户_点赞次数', '用户_评论次数', '用户_关注次数',
                       '用户_点赞后关注率', '用户_评论后关注率',
                       '博主_观看次数', '博主_点赞次数', '博主_评论次数', '博主_关注次数',
                       '博主_互动率', '博主_粉丝增长率',
                       '当日观看次数', '当日点赞次数', '当日评论次数']
        
        print("检查特征列...")
        missing_cols = [col for col in feature_cols if col not in features.columns]
        if missing_cols:
            print("警告：以下特征列不存在：", missing_cols)
            print("可用的列：", features.columns.tolist())
            raise ValueError(f"缺少必要的特征列：{missing_cols}")
        
        print("准备训练数据...")
        X = features[feature_cols]
        
        print("构建标签...")
        # 使用向量化操作构建标签
        historical_follows = df1[(df1['用户行为'] == 4) & 
                               (df1['时间'] < datetime(2024, 7, 22))][['用户ID', '博主ID']].drop_duplicates()
        historical_follows['是否关注'] = 1
        features = features.merge(historical_follows[['用户ID', '博主ID', '是否关注']], 
                                on=['用户ID', '博主ID'], how='left')
        features['是否关注'] = features['是否关注'].fillna(0)
        y = features['是否关注']
        
        print("处理类别不平衡...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print("训练XGBoost模型...")
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=len(y[y==0])/len(y[y==1]),  # 处理类别不平衡
            tree_method='hist',  # 使用直方图优化
            n_jobs=-1  # 使用所有CPU核心
        )
        
        model.fit(X_resampled, y_resampled)
        
        print("开始预测...")
        results = {}
        for user_id in target_users:
            print(f"处理用户 {user_id}...")
            user_features = features[features['用户ID'] == user_id]
            
            if len(user_features) == 0:
                print(f"警告：用户 {user_id} 在7月22日没有互动数据")
                continue
            
            probs = model.predict_proba(user_features[feature_cols])[:, 1]
            top_bloggers = user_features.iloc[probs.argsort()[-3:][::-1]]['博主ID'].tolist()
            results[user_id] = top_bloggers
        
        return results
    except Exception as e:
        print(f"模型训练和预测时出错: {str(e)}")
        raise

def main():
    try:
        # 加载数据
        df1, df2 = load_data()
        
        # 构建特征
        user_features = create_user_features(df1)
        blogger_features = create_blogger_features(df1)
        interaction_features = create_interaction_features(df1, df2, user_features, blogger_features)
        
        # 目标用户列表
        target_users = ['U7', 'U6749', 'U5769', 'U14990', 'U52010']
        
        # 训练模型并预测
        predictions = train_and_predict(interaction_features, df1, target_users)
        
        # 输出结果
        print("\n问题2最终结果：")
        print("用户ID\t预测将要关注的博主ID")
        for user_id in target_users:
            if user_id in predictions:
                bloggers = predictions[user_id]
                print(f"{user_id}\t{', '.join(bloggers)}")
            else:
                print(f"{user_id}\t无预测结果")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 