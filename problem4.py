import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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
        # 使用chunksize分块读取数据
        chunks = []
        for chunk in pd.read_csv('附件1 (Attachment 1).csv', chunksize=100000):
            chunk = chunk.rename(columns=COLUMN_MAPPING)
            chunk['时间'] = pd.to_datetime(chunk['时间']).apply(convert_date)
            chunk['用户行为'] = pd.to_numeric(chunk['用户行为'])
            chunks.append(chunk)
        
        df = pd.concat(chunks, ignore_index=True)
        print("数据加载完成")
        print(f"数据量: {len(df)}")
        return df
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

class LSTMModel(nn.Module):
    """LSTM模型用于预测用户在线状态"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

class TimeSeriesDataset(Dataset):
    """时间序列数据集"""
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_time_slot_features(df, target_date, target_users):
    """创建时段特征"""
    try:
        print("\n构建时段特征...")
        # 计算历史数据（目标日期前10天）
        start_date = target_date - timedelta(days=10)
        print("筛选时间范围内的数据...")
        
        # 只处理目标用户的数据
        df_period = df[
            (df['时间'] >= start_date) & 
            (df['时间'] < target_date) & 
            (df['用户ID'].isin(target_users))
        ]
        
        # 添加时间特征
        print("添加时间特征...")
        df_period['小时'] = df_period['时间'].dt.hour
        df_period['星期'] = df_period['时间'].dt.dayofweek
        df_period['是否周末'] = df_period['星期'].isin([5, 6]).astype(int)
        df_period['月份'] = df_period['时间'].dt.month
        df_period['日期'] = df_period['时间'].dt.day
        df_period['是否月初'] = df_period['日期'].isin(range(1, 6)).astype(int)
        df_period['是否月末'] = df_period['日期'].isin(range(25, 32)).astype(int)
        df_period['是否工作日'] = ~df_period['星期'].isin([5, 6]).astype(int)
        df_period['时段'] = pd.cut(df_period['小时'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['凌晨', '上午', '下午', '晚上'])
        
        # 创建用户-时段活跃度矩阵
        print("计算用户时段活跃度...")
        user_time_slots = pd.pivot_table(
            df_period,
            values='用户行为',
            index='小时',
            columns='用户ID',
            aggfunc='count',
            fill_value=0
        )
        
        # 确保所有时段都存在
        user_time_slots = user_time_slots.reindex(range(24), fill_value=0)
        
        # 计算用户活跃度趋势
        print("计算用户活跃度趋势...")
        user_trends = {}
        for user_id in target_users:
            if user_id in user_time_slots.columns:
                # 计算最近3天的活跃度变化趋势
                user_data = df_period[df_period['用户ID'] == user_id]
                daily_activity = user_data.groupby(user_data['时间'].dt.date).size()
                if len(daily_activity) >= 3:
                    trend = (daily_activity.iloc[-1] - daily_activity.iloc[-3]) / daily_activity.iloc[-3]
                    user_trends[user_id] = trend
                else:
                    user_trends[user_id] = 0
        
        # 计算用户时间偏好
        print("计算用户时间偏好...")
        user_time_preferences = {}
        for user_id in target_users:
            user_data = df_period[df_period['用户ID'] == user_id]
            if len(user_data) > 0:
                # 计算用户在不同时段的活跃度
                time_preferences = {
                    '时段偏好': user_data['时段'].value_counts().nlargest(2).index.tolist(),
                    '工作日活跃度': user_data[user_data['是否工作日'] == 1]['用户行为'].count(),
                    '周末活跃度': user_data[user_data['是否周末'] == 1]['用户行为'].count(),
                    '月初活跃度': user_data[user_data['是否月初'] == 1]['用户行为'].count(),
                    '月末活跃度': user_data[user_data['是否月末'] == 1]['用户行为'].count()
                }
                user_time_preferences[user_id] = time_preferences
        
        # 获取相关博主
        relevant_bloggers = df_period['博主ID'].unique()
        
        # 创建博主-时段活跃度矩阵
        print("计算博主时段活跃度...")
        blogger_time_slots = pd.pivot_table(
            df_period,
            values='用户行为',
            index='小时',
            columns='博主ID',
            aggfunc='count',
            fill_value=0
        )
        
        # 确保所有时段都存在
        blogger_time_slots = blogger_time_slots.reindex(range(24), fill_value=0)
        
        # 计算博主活跃度趋势
        print("计算博主活跃度趋势...")
        blogger_trends = {}
        for blogger_id in relevant_bloggers:
            blogger_data = df_period[df_period['博主ID'] == blogger_id]
            daily_activity = blogger_data.groupby(blogger_data['时间'].dt.date).size()
            if len(daily_activity) >= 3:
                trend = (daily_activity.iloc[-1] - daily_activity.iloc[-3]) / daily_activity.iloc[-3]
                blogger_trends[blogger_id] = trend
            else:
                blogger_trends[blogger_id] = 0
        
        # 计算博主时间偏好
        print("计算博主时间偏好...")
        blogger_time_preferences = {}
        for blogger_id in relevant_bloggers:
            blogger_data = df_period[df_period['博主ID'] == blogger_id]
            if len(blogger_data) > 0:
                # 计算博主在不同时段的活跃度
                time_preferences = {
                    '时段偏好': blogger_data['时段'].value_counts().nlargest(2).index.tolist(),
                    '工作日活跃度': blogger_data[blogger_data['是否工作日'] == 1]['用户行为'].count(),
                    '周末活跃度': blogger_data[blogger_data['是否周末'] == 1]['用户行为'].count(),
                    '月初活跃度': blogger_data[blogger_data['是否月初'] == 1]['用户行为'].count(),
                    '月末活跃度': blogger_data[blogger_data['是否月末'] == 1]['用户行为'].count()
                }
                blogger_time_preferences[blogger_id] = time_preferences
        
        print("时段特征构建完成")
        return user_time_slots, blogger_time_slots, user_trends, blogger_trends, user_time_preferences, blogger_time_preferences
    except Exception as e:
        print(f"构建时段特征时出错: {str(e)}")
        raise

def create_interaction_features(df, target_date, target_users):
    """创建用户-博主互动特征"""
    try:
        print("\n构建互动特征...")
        # 计算历史互动数据（目标日期前10天）
        start_date = target_date - timedelta(days=10)
        print("筛选时间范围内的数据...")
        
        # 只处理目标用户的数据
        df_period = df[
            (df['时间'] >= start_date) & 
            (df['时间'] < target_date) & 
            (df['用户ID'].isin(target_users))
        ]
        
        # 添加时间特征
        print("添加时间特征...")
        df_period['小时'] = df_period['时间'].dt.hour
        df_period['星期'] = df_period['时间'].dt.dayofweek
        df_period['是否周末'] = df_period['星期'].isin([5, 6]).astype(int)
        
        # 用户-博主-时段互动统计
        print("计算互动统计...")
        # 分别计算各类行为
        views = df_period[df_period['用户行为'] == 1].groupby(['用户ID', '博主ID', '小时']).size()
        likes = df_period[df_period['用户行为'] == 2].groupby(['用户ID', '博主ID', '小时']).size()
        comments = df_period[df_period['用户行为'] == 3].groupby(['用户ID', '博主ID', '小时']).size()
        follows = df_period[df_period['用户行为'] == 4].groupby(['用户ID', '博主ID', '小时']).size()
        
        # 合并统计结果
        interaction_stats = pd.DataFrame({
            '观看次数': views,
            '点赞次数': likes,
            '评论次数': comments,
            '关注次数': follows
        }).fillna(0).reset_index()
        
        # 计算互动总数和互动质量
        print("计算互动质量指标...")
        interaction_stats['互动总数'] = interaction_stats['点赞次数'] + interaction_stats['评论次数'] + interaction_stats['关注次数']
        interaction_stats['互动质量'] = (
            interaction_stats['点赞次数'] * 1 + 
            interaction_stats['评论次数'] * 2 + 
            interaction_stats['关注次数'] * 3
        ) / interaction_stats['互动总数'].replace(0, 1)
        
        # 计算用户偏好
        print("计算用户偏好...")
        user_preferences = {}
        for user_id in target_users:
            user_data = interaction_stats[interaction_stats['用户ID'] == user_id]
            if len(user_data) > 0:
                # 计算用户最常互动的时段
                preferred_hours = user_data.groupby('小时')['互动总数'].sum().nlargest(3).index.tolist()
                # 计算用户最常互动的博主类型
                preferred_bloggers = user_data.groupby('博主ID')['互动质量'].mean().nlargest(3).index.tolist()
                user_preferences[user_id] = {
                    'preferred_hours': preferred_hours,
                    'preferred_bloggers': preferred_bloggers
                }
        
        print("互动特征构建完成")
        return interaction_stats, user_preferences
    except Exception as e:
        print(f"构建互动特征时出错: {str(e)}")
        raise

def predict_online_status(user_time_slots, target_users):
    """预测用户在线状态"""
    try:
        print("\n开始预测用户在线状态...")
        # 准备数据
        print("准备训练数据...")
        sequences = []
        valid_users = []
        
        for user_id in target_users:
            if user_id in user_time_slots.columns:
                # 获取用户过去10天的时段活跃度序列
                user_sequence = user_time_slots[user_id].values.reshape(1, -1)
                sequences.append(user_sequence)
                valid_users.append(user_id)
        
        if not sequences:
            return {}
        
        # 转换为PyTorch张量
        print("转换为张量...")
        X = torch.FloatTensor(np.array(sequences))
        
        # 创建和训练LSTM模型
        print("创建LSTM模型...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        input_size = 24  # 24个时段
        hidden_size = 64
        num_layers = 2
        output_size = 24  # 预测24个时段的在线概率
        
        model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 将数据移到GPU（如果可用）
        X = X.to(device)
        
        # 训练模型
        print("训练模型...")
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, torch.ones_like(outputs))
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")
        
        # 预测
        print("进行预测...")
        model.eval()
        with torch.no_grad():
            predictions = model(X)
            predictions = predictions.cpu().numpy()
        
        # 处理预测结果
        print("处理预测结果...")
        results = {}
        for i, user_id in enumerate(valid_users):
            # 获取预测概率大于0.5的时段
            online_slots = np.where(predictions[i] > 0.5)[0]
            results[user_id] = online_slots.tolist()
        
        print("在线状态预测完成")
        return results
    except Exception as e:
        print(f"预测在线状态时出错: {str(e)}")
        raise

def predict_interactions(interaction_stats, user_preferences, online_users, online_slots, blogger_trends, user_time_preferences, blogger_time_preferences, target_date):
    """预测用户-博主互动数"""
    try:
        print("\n开始预测用户-博主互动数...")
        results = {}
        
        # 预处理互动统计数据
        print("预处理互动统计数据...")
        interaction_stats_dict = {}
        for user_id in online_users:
            user_data = interaction_stats[interaction_stats['用户ID'] == user_id]
            interaction_stats_dict[user_id] = {
                hour: group for hour, group in user_data.groupby('小时')
            }
        
        for user_id in online_users:
            if user_id not in online_slots:
                continue
            
            print(f"处理用户 {user_id}...")
            user_results = {}
            user_stats = interaction_stats_dict.get(user_id, {})
            user_pref = user_preferences.get(user_id, {})
            user_time_pref = user_time_preferences.get(user_id, {})
            
            for slot in online_slots[user_id]:
                # 获取该用户在该时段的历史互动数据
                slot_data = user_stats.get(slot, pd.DataFrame())
                
                if len(slot_data) == 0:
                    continue
                
                # 计算综合得分
                slot_data['趋势得分'] = slot_data['博主ID'].map(blogger_trends).fillna(0)
                slot_data['偏好得分'] = slot_data['博主ID'].isin(user_pref.get('preferred_bloggers', [])).astype(float)
                
                # 计算时间匹配得分
                time_match_scores = []
                for blogger_id in slot_data['博主ID']:
                    blogger_time_pref = blogger_time_preferences.get(blogger_id, {})
                    time_match_score = 0
                    
                    # 1. 时段匹配（40%）
                    slot_period = '凌晨' if slot < 6 else '上午' if slot < 12 else '下午' if slot < 18 else '晚上'
                    if slot_period in user_time_pref.get('时段偏好', []):
                        time_match_score += 0.4
                    
                    # 2. 工作日/周末匹配（30%）
                    is_weekend = target_date.weekday() >= 5
                    if is_weekend:
                        if user_time_pref.get('周末活跃度', 0) > user_time_pref.get('工作日活跃度', 0):
                            time_match_score += 0.3
                    else:
                        if user_time_pref.get('工作日活跃度', 0) > user_time_pref.get('周末活跃度', 0):
                            time_match_score += 0.3
                    
                    # 3. 月初/月末匹配（30%）
                    is_month_start = target_date.day <= 5
                    if is_month_start:
                        if user_time_pref.get('月初活跃度', 0) > user_time_pref.get('月末活跃度', 0):
                            time_match_score += 0.3
                    else:
                        if user_time_pref.get('月末活跃度', 0) > user_time_pref.get('月初活跃度', 0):
                            time_match_score += 0.3
                    
                    time_match_scores.append(time_match_score)
                
                slot_data['时间匹配得分'] = time_match_scores
                
                # 计算最终得分
                slot_data['最终得分'] = (
                    slot_data['互动质量'] * 0.35 +    # 互动质量
                    slot_data['趋势得分'] * 0.25 +    # 趋势得分
                    slot_data['偏好得分'] * 0.20 +    # 偏好得分
                    slot_data['时间匹配得分'] * 0.20  # 时间匹配得分
                )
                
                # 按最终得分排序，获取前3名博主
                top_bloggers = slot_data.nlargest(3, '最终得分')[['博主ID', '最终得分']]
                
                if len(top_bloggers) > 0:
                    user_results[slot] = top_bloggers['博主ID'].tolist()
            
            if user_results:
                results[user_id] = user_results
        
        print("互动数预测完成")
        return results
    except Exception as e:
        print(f"预测互动数时出错: {str(e)}")
        raise

def format_time_slot(hour):
    """格式化时段"""
    return f"{hour:02d}:00-{(hour+1):02d}:00"

def main():
    try:
        # 加载数据
        df = load_data()
        
        # 设置目标日期
        target_date = datetime(2024, 7, 23)
        
        # 目标用户列表
        target_users = ['U10', 'U1951', 'U1833', 'U26447']
        
        # 构建特征
        print("\n开始构建特征...")
        user_time_slots, blogger_time_slots, user_trends, blogger_trends, user_time_preferences, blogger_time_preferences = create_time_slot_features(df, target_date, target_users)
        interaction_stats, user_preferences = create_interaction_features(df, target_date, target_users)
        
        # 预测在线状态
        print("\n开始预测在线状态...")
        online_slots = predict_online_status(user_time_slots, target_users)
        
        # 预测互动数
        print("\n开始预测互动数...")
        interaction_predictions = predict_interactions(
            interaction_stats, 
            user_preferences,
            target_users, 
            online_slots,
            blogger_trends,
            user_time_preferences,
            blogger_time_preferences,
            target_date
        )
        
        # 输出结果
        print("\n问题4最终结果：")
        print("用户ID\t博主ID 1\t时段1\t博主ID 2\t时段2\t博主ID 3\t时段3")
        for user_id in target_users:
            if user_id in interaction_predictions:
                user_results = interaction_predictions[user_id]
                # 获取所有时段的博主
                all_slots = []
                all_bloggers = []
                for slot, bloggers in user_results.items():
                    all_slots.extend([format_time_slot(slot)] * len(bloggers))
                    all_bloggers.extend(bloggers)
                
                # 取前3个结果
                bloggers = all_bloggers[:3]
                slots = all_slots[:3]
                
                # 填充到3个结果
                while len(bloggers) < 3:
                    bloggers.append('')
                    slots.append('')
                
                print(f"{user_id}\t{bloggers[0]}\t{slots[0]}\t{bloggers[1]}\t{slots[1]}\t{bloggers[2]}\t{slots[2]}")
            else:
                print(f"{user_id}\t\t\t\t\t\t")
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main() 