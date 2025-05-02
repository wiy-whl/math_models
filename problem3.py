import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
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

def load_data():
    """加载数据并进行预处理"""
    try:
        print("开始加载数据...")
        df = pd.read_csv('附件1 (Attachment 1).csv')
        df = df.rename(columns=COLUMN_MAPPING)
        df['时间'] = pd.to_datetime(df['时间']).apply(convert_date)
        df['用户行为'] = pd.to_numeric(df['用户行为'])
        print("数据加载完成")
        print(f"数据量: {len(df)}")
        return df
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def create_online_features(df, target_date):
    """创建用户在线状态预测特征"""
    try:
        print("\n构建在线状态预测特征...")
        # 计算历史活跃天数（目标日期前10天）
        start_date = target_date - timedelta(days=10)
        df_period = df[(df['时间'] >= start_date) & (df['时间'] < target_date)]
        
        # 用户活跃天数
        active_days = df_period.groupby('用户ID')['时间'].agg('nunique').reset_index(name='历史活跃天数')
        
        # 最近一次登录时间间隔
        last_login = df_period.groupby('用户ID')['时间'].max().reset_index(name='最近登录时间')
        last_login['登录间隔'] = (target_date - last_login['最近登录时间']).dt.total_seconds() / 3600
        
        # 时段活跃度（按小时统计）
        df_period['小时'] = df_period['时间'].dt.hour
        hourly_activity = df_period.groupby(['用户ID', '小时']).size().unstack(fill_value=0)
        hourly_activity.columns = [f'时段_{h}点活跃度' for h in hourly_activity.columns]
        
        # 合并特征
        features = active_days.merge(last_login[['用户ID', '登录间隔']], on='用户ID', how='left')
        features = features.merge(hourly_activity, on='用户ID', how='left')
        features = features.fillna(0)
        
        print("在线状态预测特征构建完成")
        return features
    except Exception as e:
        print(f"构建在线状态预测特征时出错: {str(e)}")
        raise

def create_interaction_features(df, target_date):
    """创建用户-博主互动预测特征"""
    try:
        print("\n构建互动预测特征...")
        # 计算历史互动数据（目标日期前10天）
        start_date = target_date - timedelta(days=10)
        df_period = df[(df['时间'] >= start_date) & (df['时间'] < target_date)]
        
        # 用户-博主对的历史互动统计
        interaction_stats = df_period.groupby(['用户ID', '博主ID']).agg({
            '用户行为': [
                ('观看次数', lambda x: (x == 1).sum()),
                ('点赞次数', lambda x: (x == 2).sum()),
                ('评论次数', lambda x: (x == 3).sum()),
                ('关注次数', lambda x: (x == 4).sum())
            ]
        }).reset_index()
        
        # 处理多级索引列名
        interaction_stats.columns = ['用户ID', '博主ID', '观看次数', '点赞次数', '评论次数', '关注次数']
        
        # 计算互动总数
        interaction_stats['历史互动总数'] = interaction_stats['点赞次数'] + interaction_stats['评论次数'] + interaction_stats['关注次数']
        
        # 计算用户对博主的偏好度
        interaction_stats['互动偏好度'] = interaction_stats['历史互动总数'] / interaction_stats['观看次数'].replace(0, 1)
        
        print("互动预测特征构建完成")
        return interaction_stats
    except Exception as e:
        print(f"构建互动预测特征时出错: {str(e)}")
        raise

def predict_online_status(features, target_users):
    """预测用户在线状态"""
    try:
        print("\n开始预测用户在线状态...")
        # 准备特征
        feature_cols = [col for col in features.columns if col != '用户ID']
        X = features[feature_cols]
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练随机森林模型
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        # 使用历史数据作为训练集
        # 这里假设历史数据中的用户都是活跃的
        y = np.ones(len(X))
        model.fit(X_scaled, y)
        
        # 预测目标用户
        target_features = features[features['用户ID'].isin(target_users)]
        if len(target_features) == 0:
            return {}
        
        X_target = target_features[feature_cols]
        X_target_scaled = scaler.transform(X_target)
        predictions = model.predict(X_target_scaled)
        
        # 返回预测结果
        results = dict(zip(target_features['用户ID'], predictions))
        print("在线状态预测完成")
        return results
    except Exception as e:
        print(f"预测在线状态时出错: {str(e)}")
        raise

def predict_interactions(features, online_users):
    """预测用户-博主互动数"""
    try:
        print("\n开始预测用户-博主互动数...")
        # 准备特征
        feature_cols = ['观看次数', '点赞次数', '评论次数', '关注次数', '历史互动总数', '互动偏好度']
        X = features[feature_cols]
        y = features['历史互动总数']
        
        # 训练XGBoost模型
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X, y)
        
        # 预测目标用户的互动数
        results = {}
        for user_id in online_users:
            user_features = features[features['用户ID'] == user_id]
            if len(user_features) == 0:
                continue
            
            # 预测互动数
            predictions = model.predict(user_features[feature_cols])
            
            # 获取互动数最高的3个博主
            top_indices = predictions.argsort()[-3:][::-1]
            top_bloggers = user_features.iloc[top_indices]['博主ID'].tolist()
            results[user_id] = top_bloggers
        
        print("互动数预测完成")
        return results
    except Exception as e:
        print(f"预测互动数时出错: {str(e)}")
        raise

def main():
    try:
        # 加载数据
        df = load_data()
        
        # 设置目标日期
        target_date = datetime(2024, 7, 21)
        
        # 目标用户列表
        target_users = ['U9', 'U22405', 'U16', 'U48420']
        
        # 构建特征
        online_features = create_online_features(df, target_date)
        interaction_features = create_interaction_features(df, target_date)
        
        # 预测在线状态
        online_predictions = predict_online_status(online_features, target_users)
        
        # 获取在线用户
        online_users = [user for user, is_online in online_predictions.items() if is_online]
        
        # 预测互动数
        interaction_predictions = predict_interactions(interaction_features, online_users)
        
        # 输出结果
        print("\n问题3最终结果：")
        print("用户ID\t博主ID 1\t博主ID 2\t博主ID 3")
        for user_id in target_users:
            if user_id in online_predictions and online_predictions[user_id]:
                bloggers = interaction_predictions.get(user_id, [''] * 3)
                print(f"{user_id}\t{bloggers[0]}\t{bloggers[1]}\t{bloggers[2]}")
            else:
                print(f"{user_id}\t\t\t")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 