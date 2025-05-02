import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import VotingRegressor

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
    """特征工程"""
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
        df['增长率'] = df['一阶差分'] / df.groupby('博主ID')['新增关注数'].shift(1).replace(0, 1)
        
        # 标记周末
        df['是否周末'] = df['时间'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # 填充缺失值
        df = df.fillna(0)
        
        print("特征工程完成")
        return df
    except Exception as e:
        print(f"特征工程时出错: {str(e)}")
        raise

def train_arima_model(series, order=(1,1,1)):
    """训练ARIMA模型"""
    try:
        model = ARIMA(series, order=order)
        results = model.fit()
        return results
    except Exception as e:
        print(f"ARIMA模型训练出错: {str(e)}")
        return None

def predict_with_arima(model, steps=1):
    """使用ARIMA模型进行预测"""
    try:
        forecast = model.forecast(steps=steps)
        return forecast[0]
    except Exception as e:
        print(f"ARIMA预测出错: {str(e)}")
        return None

def train_and_predict(df):
    """模型训练和预测"""
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
            last_date = df['时间'].dt.date.max()
            test_data = df[df['时间'].dt.date == last_date]
            train_data = df[df['时间'].dt.date < last_date]
            print(f"使用 {last_date} 的数据作为测试集")
            print(f"更新后训练数据量: {len(train_data)}")
            print(f"更新后测试数据量: {len(test_data)}")
        
        # 特征列表
        features = ['观看数', '点赞数', '评论数', '互动率', '3天均值', '5天均值', 
                   '一阶差分', '增长率', '是否周末']
        
        # 确保所有特征都存在
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"警告：以下特征不存在，将被移除：{missing_features}")
            features = [f for f in features if f in df.columns]
        
        # 准备XGBoost模型
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # 训练XGBoost模型
        print("开始训练XGBoost模型...")
        xgb_model.fit(train_data[features], train_data['新增关注数'])
        
        # 准备ARIMA模型
        print("开始训练ARIMA模型...")
        arima_predictions = []
        for blogger_id in test_data['博主ID'].unique():
            blogger_train = train_data[train_data['博主ID'] == blogger_id].sort_values('时间')
            blogger_test = test_data[test_data['博主ID'] == blogger_id]
            
            if len(blogger_train) > 0:
                # 使用固定的ARIMA参数
                arima_model = train_arima_model(blogger_train['新增关注数'].values, order=(1,1,1))
                if arima_model is not None:
                    arima_pred = predict_with_arima(arima_model)
                    arima_predictions.append({
                        '博主ID': blogger_id,
                        'ARIMA预测': arima_pred if arima_pred is not None else 0
                    })
        
        # 合并ARIMA预测结果
        arima_df = pd.DataFrame(arima_predictions)
        test_data = test_data.merge(arima_df, on='博主ID', how='left')
        test_data['ARIMA预测'] = test_data['ARIMA预测'].fillna(0)
        
        # XGBoost预测
        print("开始XGBoost预测...")
        xgb_pred = xgb_model.predict(test_data[features])
        test_data['XGBoost预测'] = xgb_pred
        
        # 集成预测（加权平均）
        test_data['预测新增关注数'] = 0.6 * test_data['XGBoost预测'] + 0.4 * test_data['ARIMA预测']
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            '博主ID': test_data['博主ID'],
            '预测新增关注数': test_data['预测新增关注数']
        })
        
        # 按预测值排序并获取前5名
        top_5 = results.nlargest(5, '预测新增关注数')
        
        return top_5
    except Exception as e:
        print(f"模型训练和预测时出错: {str(e)}")
        raise

def plot_time_series_analysis(df, blogger_ids=None):
    """绘制时间序列分析图表"""
    try:
        print("\n开始绘制时间序列分析图表...")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 如果没有指定博主ID，选择关注数最多的前5个博主
        if blogger_ids is None:
            blogger_ids = df.groupby('博主ID')['新增关注数'].sum().nlargest(5).index.tolist()
        
        # 1. 绘制每个博主的每日新增关注数折线图
        plt.figure(figsize=(15, 8))
        for blogger_id in blogger_ids:
            blogger_data = df[df['博主ID'] == blogger_id].sort_values('时间')
            plt.plot(blogger_data['时间'], blogger_data['新增关注数'], 
                    label=f'博主{blogger_id}', marker='o')
        
        plt.title('博主每日新增关注数变化趋势')
        plt.xlabel('日期')
        plt.ylabel('新增关注数')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('daily_follows_trend.png')
        plt.close()
        
        # 2. 绘制分面图
        df_subset = df[df['博主ID'].isin(blogger_ids)].sort_values(['博主ID', '时间'])
        g = sns.FacetGrid(df_subset, 
                         col='博主ID', 
                         col_wrap=3, 
                         height=4, 
                         aspect=1.5)
        g.map_dataframe(sns.lineplot, x='时间', y='新增关注数', marker='o')
        g.set_axis_labels('日期', '新增关注数')
        g.set_titles('博主 {col_name}')
        for ax in g.axes:
            ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig('faceted_follows_trend.png')
        plt.close()
        
        # 3. 为每个博主绘制ACF和PACF图
        for blogger_id in blogger_ids:
            blogger_data = df[df['博主ID'] == blogger_id].sort_values('时间')
            follows_series = blogger_data['新增关注数'].values
            n_samples = len(follows_series)
            max_lags = min(4, n_samples // 2)  # 设置最大滞后阶数为样本量的一半或4，取较小值
            
            try:
                # 绘制ACF图
                plt.figure(figsize=(12, 4))
                plot_acf(follows_series, lags=max_lags, title=f'博主{blogger_id}的ACF图')
                plt.tight_layout()
                plt.savefig(f'acf_blogger_{blogger_id}.png')
                plt.close()
                
                # 绘制PACF图
                plt.figure(figsize=(12, 4))
                plot_pacf(follows_series, lags=max_lags, title=f'博主{blogger_id}的PACF图')
                plt.tight_layout()
                plt.savefig(f'pacf_blogger_{blogger_id}.png')
                plt.close()
            except Exception as e:
                print(f"为博主{blogger_id}绘制ACF/PACF图时出错: {str(e)}")
                continue
        
        print("时间序列分析图表绘制完成")
        print("已生成以下文件：")
        print("1. daily_follows_trend.png - 每日新增关注数变化趋势")
        print("2. faceted_follows_trend.png - 分面图")
        print("3. acf_blogger_*.png - 各博主的ACF图")
        print("4. pacf_blogger_*.png - 各博主的PACF图")
        
    except Exception as e:
        print(f"绘制时间序列分析图表时出错: {str(e)}")
        raise

def main():
    try:
        # 加载数据
        df = load_data()
        
        # 数据预处理
        processed_data = preprocess_data(df)
        
        # 特征工程
        data_with_features = create_features(processed_data)
        
        # 绘制时间序列分析图表
        plot_time_series_analysis(processed_data)
        
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