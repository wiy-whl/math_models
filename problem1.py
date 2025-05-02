import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        # 读取附件1
        df = pd.read_csv('附件1 (Attachment 1).csv')
        # 重命名列
        df = df.rename(columns=COLUMN_MAPPING)
        # 转换时间列为datetime类型并调整年份
        df['时间'] = pd.to_datetime(df['时间']).apply(convert_date)
        # 转换用户行为为数值类型
        df['用户行为'] = pd.to_numeric(df['用户行为'])
        
        print("数据加载完成")
        print(f"总数据量: {len(df)}")
        print("\n数据时间范围:")
        print(f"最早时间: {df['时间'].min()}")
        print(f"最晚时间: {df['时间'].max()}")
        return df
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

# 数据预处理
def preprocess_data(df):
    try:
        print("\n开始数据预处理...")
        
        # 提取每种行为的数据
        follows = df[df['用户行为'] == 4]
        views = df[df['用户行为'] == 1]
        likes = df[df['用户行为'] == 2]
        comments = df[df['用户行为'] == 3]
        
        print("按行为类型分组完成")
        
        # 按博主ID和日期聚合，统计每日新增关注数
        daily_follows = follows.groupby(['博主ID', follows['时间'].dt.date]).size().reset_index(name='新增关注数')
        
        # 计算每个博主每天的其他指标
        daily_views = views.groupby(['博主ID', views['时间'].dt.date]).size().reset_index(name='观看数')
        daily_likes = likes.groupby(['博主ID', likes['时间'].dt.date]).size().reset_index(name='点赞数')
        daily_comments = comments.groupby(['博主ID', comments['时间'].dt.date]).size().reset_index(name='评论数')
        
        print("计算每日统计指标完成")
        
        # 合并所有指标
        result = daily_follows.merge(daily_views, on=['博主ID', '时间'], how='left')
        result = result.merge(daily_likes, on=['博主ID', '时间'], how='left')
        result = result.merge(daily_comments, on=['博主ID', '时间'], how='left')
        
        # 填充缺失值
        result = result.fillna(0)
        
        # 计算互动率
        result['互动率'] = (result['点赞数'] + result['评论数']) / result['观看数'].replace(0, 1)
        
        # 将时间列转换为datetime
        result['时间'] = pd.to_datetime(result['时间'])
        
        print("特征计算完成")
        print(f"处理后数据量: {len(result)}")
        print("\n数据时间范围:")
        print(f"最早时间: {result['时间'].min()}")
        print(f"最晚时间: {result['时间'].max()}")
        
        return result
    except Exception as e:
        print(f"数据预处理时出错: {str(e)}")
        raise

# 特征工程
def create_features(df):
    try:
        print("\n开始特征工程...")
        
        # 按博主ID分组并排序
        df = df.sort_values(['博主ID', '时间'])
        
        # 计算滑动窗口特征
        df['3天均值'] = df.groupby('博主ID')['新增关注数'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        df['5天均值'] = df.groupby('博主ID')['新增关注数'].rolling(window=5, min_periods=1).mean().reset_index(0, drop=True)
        
        # 计算一阶差分
        df['一阶差分'] = df.groupby('博主ID')['新增关注数'].diff()
        
        # 计算增长率
        df['增长率'] = df['一阶差分'] / df.groupby('博主ID')['新增关注数'].shift(1)
        
        # 标记周末
        df['是否周末'] = df['时间'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # 填充缺失值
        df = df.fillna(0)
        
        print("特征工程完成")
        return df
    except Exception as e:
        print(f"特征工程时出错: {str(e)}")
        raise

# 模型训练和预测
def train_and_predict(df):
    try:
        print("\n开始模型训练和预测...")
        
        # 准备训练数据
        target_date = pd.Timestamp('2024-07-21')
        train_data = df[df['时间'].dt.date < target_date.date()]
        test_data = df[df['时间'].dt.date == target_date.date()]
        
        print(f"训练数据量: {len(train_data)}")
        print(f"测试数据量: {len(test_data)}")
        
        if len(test_data) == 0:
            print("警告：没有找到2024-07-21的测试数据，将使用最近日期的数据进行预测")
            # 获取最后一天的数据作为测试集
            last_date = df['时间'].dt.date.max()
            test_data = df[df['时间'].dt.date == last_date]
            train_data = df[df['时间'].dt.date < last_date]
            print(f"使用 {last_date} 的数据作为测试集")
            print(f"更新后训练数据量: {len(train_data)}")
            print(f"更新后测试数据量: {len(test_data)}")
        
        # 特征列表
        features = ['观看数', '点赞数', '评论数', '互动率', '3天均值', '5天均值', 
                    '一阶差分', '增长率', '是否周末']
        
        # 准备XGBoost模型
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        
        # 训练模型
        model.fit(train_data[features], train_data['新增关注数'])
        
        # 预测
        predictions = model.predict(test_data[features])
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            '博主ID': test_data['博主ID'],
            '预测新增关注数': predictions
        })
        
        # 按预测值排序并获取前5名
        top_5 = results.nlargest(5, '预测新增关注数')
        
        return top_5
    except Exception as e:
        print(f"模型训练和预测时出错: {str(e)}")
        raise

def main():
    try:
        # 加载数据
        df = load_data()
        
        # 数据预处理
        processed_data = preprocess_data(df)
        
        # 特征工程
        data_with_features = create_features(processed_data)
        
        # 训练模型并预测
        top_5_predictions = train_and_predict(data_with_features)
        
        # 输出结果
        print("\n问题1最终结果：")
        print("排名\t博主ID\t预测新增关注数")
        for i, (_, row) in enumerate(top_5_predictions.iterrows(), 1):
            print(f"{i}\t{row['博主ID']}\t{int(row['预测新增关注数'])}")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main() 