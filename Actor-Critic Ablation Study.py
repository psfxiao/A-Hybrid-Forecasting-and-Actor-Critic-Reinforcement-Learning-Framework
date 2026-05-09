import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
import os
from datetime import timedelta

warnings.filterwarnings('ignore')
np.random.seed(42)

# ------------------------------
# 1. 数据准备（复用原代码）
# ------------------------------
historical_data_2024 = pd.read_excel("The daily fabric outbound volume at all time in 2024.xlsx")
historical_data = pd.concat([historical_data_2024], ignore_index=True)
historical_data['date'] = pd.to_datetime(historical_data['date'])
time_group = historical_data.groupby('time period')['Total fabric outbound quantity (unit: rolls)'].mean().reset_index()
time_group.rename(columns={'Total fabric outbound quantity (unit: rolls)': 'target_demand'}, inplace=True)

forecast_data = pd.read_csv("order_day_time_period_demand_forecast.csv")
forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
forecast_data = forecast_data.merge(time_group, left_on='Time Period', right_on='time period')
forecast_data['time period'] = forecast_data['time period'].astype(str)

# 按时间划分训练/测试集（前80%天训练，后20%天测试）
dates = sorted(forecast_data['Date'].dt.date.unique())
split_idx = int(0.8 * len(dates))
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]

train_data = forecast_data[forecast_data['Date'].dt.date.isin(train_dates)].copy()
test_data = forecast_data[forecast_data['Date'].dt.date.isin(test_dates)].copy()
print(f"训练天数: {len(train_dates)}, 测试天数: {len(test_dates)}")


# 静态折扣（原代码中计算得到，用于无RL配置）
def calc_static_discount(forecast_df, elasticity=-1.5):
    peak_hours = ['15:00-16:00', '16:00-17:00']
    results = []
    time_slots = forecast_df['time period'].unique()
    for slot in time_slots:
        slot_data = forecast_df[forecast_df['time period'] == slot]
        Q = slot_data['Predicted Demand (rolls)'].values
        T = slot_data['target_demand'].values[0]
        if slot in peak_hours:
            discount = 0.0
        else:
            avg_forecast = np.mean(Q)
            if T == 0:
                discount = 0.0
            else:
                demand_gap = (T - avg_forecast) / T
                if abs(demand_gap) < 0.05:
                    discount = 0.0
                else:
                    discount = demand_gap / (elasticity * 2)
                    if discount > 0.15:
                        discount = 0.15 + (discount - 0.15) * 0.3
                    elif discount < -0.15:
                        discount = -0.15 + (discount + 0.15) * 0.3
        discount = np.clip(discount, -0.2, 0.2)
        results.append({'time period': slot, 'discount': round(discount, 4)})
    return pd.DataFrame(results)


static_discount_df = calc_static_discount(forecast_data)


# ------------------------------
# 2. 定义可配置的RL环境
# ------------------------------
class ConfigurableDemandSmoothingEnv:
    def __init__(self, data, elasticity=-1.5, use_constraint=True, full_reward=True, use_state_mean=True):
        self.data = data
        self.elasticity = elasticity
        self.use_constraint = use_constraint  # 是否对折扣施加约束（±0.05）
        self.full_reward = full_reward  # 是否使用完整奖励（含惩罚项）
        self.use_state_mean = use_state_mean  # 状态是否包含历史均值
        self.time_slots = data['time period'].unique()
        self.static_discounts = self._calc_static_discounts()
        self.reset()

    def _calc_static_discounts(self):
        static = {}
        for idx, row in self.data.iterrows():
            slot = row['time period']
            if slot not in static:
                # 从全局静态折扣获取（这里简化，直接用原函数）
                static[slot] = static_discount_df[static_discount_df['time period'] == slot]['discount'].values[0]
        return static

    def reset(self, day=None):
        if day is None:
            self.day = np.random.choice(self.data['Date'].dt.date.unique())
        else:
            self.day = day
        self.day_data = self.data[(self.data['Date'].dt.date == self.day)].sort_values('time period')
        self.current_step = 0
        self.adjusted_demands = []
        return self._get_state()

    def _get_state(self):
        if self.current_step < len(self.day_data):
            current_row = self.day_data.iloc[self.current_step]
            state = [
                self.current_step / len(self.time_slots),
                current_row['Predicted Demand (rolls)'] / 2000,
                current_row['target_demand'] / 2000,
                len(self.adjusted_demands) / len(self.time_slots)
            ]
            if self.use_state_mean:
                mean_adj = np.mean(self.adjusted_demands) / 2000 if self.adjusted_demands else 0
                state.insert(3, mean_adj)  # 插入到第4个位置（索引3）
            else:
                state.insert(3, 0)  # 用0填充
            return np.array(state)
        else:
            return None

    def step(self, action):
        current_row = self.day_data.iloc[self.current_step]
        forecast_demand = current_row['Predicted Demand (rolls)']
        target_demand = current_row['target_demand']
        time_slot = current_row['time period']
        peak_hours = ['15:00-16:00', '16:00-17:00']
        static_discount = self.static_discounts[time_slot]

        if time_slot in peak_hours:
            discount = 0.0
        else:
            if self.use_constraint:
                discount = np.clip(static_discount + action, static_discount - 0.05, static_discount + 0.05)
            else:
                discount = static_discount + action
            discount = np.clip(discount, -0.2, 0.2)  # 全局限制

        adjusted_demand = forecast_demand * (1 + self.elasticity * discount)
        self.adjusted_demands.append(adjusted_demand)

        demand_diff = abs(adjusted_demand - target_demand)
        relative_demand_diff = demand_diff / max(target_demand, 1)
        discount_diff = abs(discount - static_discount)

        if self.full_reward:
            base_reward = 1 / (1 + relative_demand_diff * 10)
            demand_penalty = - (relative_demand_diff * 5) ** 2
            discount_penalty = - (discount_diff * 20) ** 2
            reward = base_reward + demand_penalty + discount_penalty
        else:
            # 简化奖励：只关注需求匹配
            reward = - relative_demand_diff  # 负的差异

        self.current_step += 1
        done = self.current_step >= len(self.day_data)
        next_state = self._get_state() if not done else None
        return next_state, reward, done, {
            'forecast_demand': forecast_demand,
            'target_demand': target_demand,
            'adjusted_demand': adjusted_demand,
            'discount': discount,
            'date': self.day,
            'relative_demand_diff': relative_demand_diff,
            'discount_diff': discount_diff
        }


# ------------------------------
# 3. 定义可配置的Actor/Critic代理
# ------------------------------
class SimpleActor:
    def __init__(self, state_dim, learning_rate=0.001):
        self.weights = np.random.randn(state_dim) * 0.1
        self.bias = 0.0
        self.lr = learning_rate
        self.sigma = 0.05

    def predict(self, state):
        mu = np.tanh(np.dot(state, self.weights) + self.bias) * 0.05
        return mu, self.sigma

    def update(self, states, actions, advantages):
        if len(states) == 0:
            return
        states = np.array(states)
        advantages = np.array(advantages)
        for state, action, advantage in zip(states, actions, advantages):
            mu, _ = self.predict(state)
            grad_weights = advantage * (action - mu) * state
            grad_bias = advantage * (action - mu)
            self.weights += self.lr * grad_weights
            self.bias += self.lr * grad_bias


class SimpleCritic:
    def __init__(self, state_dim, learning_rate=0.01):
        self.weights = np.random.randn(state_dim) * 0.1
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, state):
        return np.dot(state, self.weights) + self.bias

    def update(self, states, returns):
        if len(states) == 0:
            return
        states = np.array(states)
        returns = np.array(returns)
        predictions = self.predict(states)
        errors = returns - predictions
        grad_weights = -2 * np.dot(states.T, errors) / len(states)
        grad_bias = -2 * np.mean(errors)
        self.weights -= self.lr * grad_weights
        self.bias -= self.lr * grad_bias


class ConfigurableACAgent:
    def __init__(self, state_dim, gamma=0.99, random_policy=False):
        self.actor = SimpleActor(state_dim)
        self.critic = SimpleCritic(state_dim)
        self.gamma = gamma
        self.random_policy = random_policy
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def select_action(self, state):
        if self.random_policy:
            action = np.random.normal(0, 0.05)
            value = 0
        else:
            mu, sigma = self.actor.predict(state)
            action = np.random.normal(mu, sigma)
            value = self.critic.predict(state)
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        return action, 0

    def update(self):
        if len(self.rewards) == 0:
            return
        returns = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = returns - np.array(self.values)
        self.actor.update(self.states, self.actions, advantages)
        self.critic.update(self.states, returns)
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def train(self, env, episodes=1000):
        episode_rewards = []
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.rewards.append(reward)
                total_reward += reward
                state = next_state
            self.update()
            episode_rewards.append(total_reward)
        return episode_rewards

    def predict(self, env, day):
        state = env.reset(day)
        results = []
        done = False
        while not done:
            action, _ = self.select_action(state)
            next_state, _, done, info = env.step(action)
            state = next_state
            results.append({
                'date': info['date'].strftime('%Y-%m-%d'),
                'time period': env.day_data.iloc[env.current_step - 1]['time period'],
                'forecast_demand': info['forecast_demand'],
                'target_demand': info['target_demand'],
                'adjusted_demand': info['adjusted_demand'],
                'price_discount': info['discount'],
                'relative_demand_diff': info['relative_demand_diff'],
                'discount_diff': info['discount_diff']
            })
        day_df = pd.DataFrame(results)
        # 修正总需求比例（保持总需求不变）
        total_forecast = day_df['forecast_demand'].sum()
        total_adjusted = day_df['adjusted_demand'].sum()
        if total_adjusted != 0:
            day_df['price_adjusted_demand'] = day_df['adjusted_demand'] * (total_forecast / total_adjusted)
        else:
            day_df['price_adjusted_demand'] = day_df['adjusted_demand']
        return day_df


# ------------------------------
# 4. 定义评估指标计算函数
# ------------------------------
def compute_metrics(result_df):
    """输入一个DataFrame（包含所有测试天的预测结果），计算评估指标"""
    # 需求匹配相关
    mape = np.mean(np.abs(result_df['adjusted_demand'] - result_df['target_demand']) /
                   np.maximum(result_df['target_demand'], 1)) * 100
    mae = np.mean(np.abs(result_df['adjusted_demand'] - result_df['target_demand']))
    rmse = np.sqrt(np.mean((result_df['adjusted_demand'] - result_df['target_demand']) ** 2))
    # 目标达成率（偏差<10%的占比）
    target_achievement = np.mean(np.abs(result_df['adjusted_demand'] - result_df['target_demand']) /
                                 np.maximum(result_df['target_demand'], 1) < 0.1) * 100
    # 折扣稳定性
    discount_std = result_df.groupby('time period')['price_discount'].std().mean()
    # 总需求偏差
    total_forecast = result_df.groupby('date')['forecast_demand'].sum()
    total_adjusted = result_df.groupby('date')['adjusted_demand'].sum()
    total_bias = np.mean((total_adjusted - total_forecast) / total_forecast) * 100
    return {
        'MAPE (%)': mape,
        'MAE (rolls)': mae,
        'RMSE (rolls)': rmse,
        'Target Achievement (%)': target_achievement,
        'Discount Std (avg)': discount_std,
        'Total Demand Bias (%)': total_bias
    }


# ------------------------------
# 5. 实验运行函数
# 5. 实验运行函数
def run_experiment(exp_name, config, train_data, test_data):
    print(f"Running {exp_name}...")

    # 无RL配置：直接使用静态折扣，不训练
    # 无RL配置：直接使用预测值作为基准（最合理的 Baseline）
    if config['no_rl']:
        print("    Using No-RL Baseline: Static Discount Policy")

        # 创建test_data的副本，避免修改原数据
        test_df = test_data.copy()

        # 标准化列名
        test_df.rename(columns={
            'Date': 'date',
            'Time Period': 'time_period',
            'Predicted Demand (rolls)': 'forecast_demand',
            'target_demand': 'target_demand'
        }, inplace=True)

        # 确保time_period列存在
        if 'time_period' not in test_df.columns:
            raise KeyError("test_df does not contain 'time_period' column after rename")

        # 构建静态折扣映射（键为'time period'，但需要匹配）
        static_map = static_discount_df.set_index('time period')['discount'].to_dict()

        # 映射折扣
        test_df['discount'] = test_df['time_period'].map(static_map)
        # 检查是否有未映射到的时段
        missing = test_df['discount'].isna().sum()
        if missing > 0:
            print(f"    Warning: {missing} rows have no discount mapping, will fill with 0")
            test_df['discount'] = test_df['discount'].fillna(0.0)

        # 高峰时段折扣强制为0
        peak_hours = ['15:00-16:00', '16:00-17:00']
        test_df.loc[test_df['time_period'].isin(peak_hours), 'discount'] = 0.0

        # 计算调整后需求
        elasticity = config.get('elasticity', -1.5)
        test_df['adjusted_demand'] = test_df['forecast_demand'] * (1 + elasticity * test_df['discount'])

        # 总需求比例修正（按天进行）
        result_list = []
        for day, day_group in test_df.groupby('date'):
            total_forecast = day_group['forecast_demand'].sum()
            total_adjusted = day_group['adjusted_demand'].sum()
            if total_adjusted != 0:
                day_group['price_adjusted_demand'] = day_group['adjusted_demand'] * (total_forecast / total_adjusted)
            else:
                day_group['price_adjusted_demand'] = day_group['adjusted_demand']
            result_list.append(day_group)
        result_df = pd.concat(result_list, ignore_index=True)

        # 确保列名与compute_metrics期望一致
        result_df['price_discount'] = result_df['discount']
        # compute_metrics使用'adjusted_demand'和'target_demand'，所以保留这些列
        # 注意：result_df中已有'target_demand'列

        # 调试打印：检查是否有NaN
        nan_cols = result_df.isna().sum()
        if nan_cols.any():
            print("    NaN columns:", nan_cols[nan_cols > 0])

        # 计算指标
        metrics = compute_metrics(result_df)

        # 调试打印
        print(f"    Sample: Forecast={result_df['forecast_demand'].iloc[0]:.1f}, "
              f"Target={result_df['target_demand'].iloc[0]:.1f}, "
              f"Adjusted={result_df['adjusted_demand'].iloc[0]:.1f}, "
              f"Discount={result_df['discount'].iloc[0]:.3f}")

        return metrics, result_df


    # RL配置：创建环境，训练代理，在测试集上预测
    # 训练环境使用训练数据
    env_train = ConfigurableDemandSmoothingEnv(
        data=train_data,
        elasticity=config['elasticity'],
        use_constraint=config['use_constraint'],
        full_reward=config['full_reward'],
        use_state_mean=config['use_state_mean']
    )
    # 获取状态维度
    dummy_state = env_train._get_state()
    state_dim = len(dummy_state) if dummy_state is not None else 5
    agent = ConfigurableACAgent(state_dim=state_dim, gamma=0.99, random_policy=config.get('random_policy', False))

    # 训练
    agent.train(env_train, episodes=config.get('episodes', 1000))

    # 在测试集上预测
    env_test = ConfigurableDemandSmoothingEnv(
        data=test_data,
        elasticity=config['elasticity'],
        use_constraint=config['use_constraint'],
        full_reward=config['full_reward'],
        use_state_mean=config['use_state_mean']
    )
    all_results = []
    for day in test_dates:
        day_df = agent.predict(env_test, day)
        all_results.append(day_df)
    result_df = pd.concat(all_results, ignore_index=True)
    metrics = compute_metrics(result_df)
    return metrics, result_df


# ------------------------------
# 6. 定义消融实验配置
# ------------------------------
configs = {
    'A_Full_Model': {
        'no_rl': False,
        'elasticity': -1.5,
        'use_constraint': True,
        'full_reward': True,
        'use_state_mean': True,
        'random_policy': False,
        'episodes': 1000
    },
    'B_No_RL': {
        'no_rl': True,
        'elasticity': -1.5
    },
    'C_No_Constraint': {
        'no_rl': False,
        'elasticity': -1.5,
        'use_constraint': False,
        'full_reward': True,
        'use_state_mean': True,
        'random_policy': False,
        'episodes': 1000
    },
    'D_Simple_Reward': {
        'no_rl': False,
        'elasticity': -1.5,
        'use_constraint': True,
        'full_reward': False,
        'use_state_mean': True,
        'random_policy': False,
        'episodes': 1000
    },
    'E_No_State_Mean': {
        'no_rl': False,
        'elasticity': -1.5,
        'use_constraint': True,
        'full_reward': True,
        'use_state_mean': False,
        'random_policy': False,
        'episodes': 1000
    },
    'F_Random_Policy': {
        'no_rl': False,
        'elasticity': -1.5,
        'use_constraint': True,
        'full_reward': True,
        'use_state_mean': True,
        'random_policy': True,
        'episodes': 1000
    }
}

# ------------------------------
# 7. 运行所有实验
# ------------------------------
results_summary = {}
all_metrics_df = pd.DataFrame()

for name, cfg in configs.items():
    try:
        metrics, _ = run_experiment(name, cfg, train_data, test_data)
        results_summary[name] = metrics
        print(f"{name} completed: MAPE={metrics['MAPE (%)']:.2f}%, Achieve={metrics['Target Achievement (%)']:.2f}%")
    except Exception as e:
        print(f"Error in {name}: {e}")
        results_summary[name] = {k: np.nan for k in ['MAPE (%)', 'MAE (rolls)', 'RMSE (rolls)',
                                                     'Target Achievement (%)', 'Discount Std (avg)',
                                                     'Total Demand Bias (%)']}

# 转换为DataFrame
metrics_df = pd.DataFrame(results_summary).T
metrics_df = metrics_df.round(2)
metrics_df.to_excel("ablation_results.xlsx", index=True)
print("\n消融实验结果已保存至 ablation_results.xlsx")
print(metrics_df)

# ------------------------------
# 8. 可视化
# ------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：MAPE对比（越小越好）
ax = axes[0, 0]
metrics_df_sorted = metrics_df.sort_values('MAPE (%)')
ax.barh(metrics_df_sorted.index, metrics_df_sorted['MAPE (%)'], color='steelblue')
ax.set_xlabel('MAPE (%)')
ax.set_title('Mean Absolute Percentage Error (Lower is better)')
ax.grid(axis='x', alpha=0.5)

# 子图2：目标达成率对比（越大越好）
ax = axes[0, 1]
metrics_df_sorted = metrics_df.sort_values('Target Achievement (%)', ascending=False)
ax.barh(metrics_df_sorted.index, metrics_df_sorted['Target Achievement (%)'], color='darkorange')
ax.set_xlabel('Target Achievement (%)')
ax.set_title('Percentage of Days with <10% Deviation (Higher is better)')
ax.grid(axis='x', alpha=0.5)

# 子图3：折扣标准差（越小越好）
ax = axes[1, 0]
metrics_df_sorted = metrics_df.sort_values('Discount Std (avg)')
ax.barh(metrics_df_sorted.index, metrics_df_sorted['Discount Std (avg)'], color='forestgreen')
ax.set_xlabel('Discount Std (avg)')
ax.set_title('Discount Stability (Lower std is better)')
ax.grid(axis='x', alpha=0.5)

# 子图4：总需求偏差（接近0越好）
ax = axes[1, 1]
metrics_df_sorted = metrics_df.sort_values('Total Demand Bias (%)', key=abs)
ax.barh(metrics_df_sorted.index, metrics_df_sorted['Total Demand Bias (%)'], color='crimson')
ax.set_xlabel('Total Demand Bias (%)')
ax.set_title('Bias in Total Demand (Closer to 0 is better)')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(axis='x', alpha=0.5)

plt.tight_layout()
plt.savefig('ablation_metrics_comparison.png', dpi=300)
plt.show()

# 雷达图展示综合性能（归一化后）
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# 对于MAPE、RMSE、MAE、Discount Std、Total Bias，我们使用反比例（越小越好），目标达成率使用正向
metrics_for_radar = metrics_df.copy()
# 反向指标转化为正向（越小越好转为越大越好）
neg_metrics = ['MAPE (%)', 'MAE (rolls)', 'RMSE (rolls)', 'Discount Std (avg)', 'Total Demand Bias (%)']
for col in neg_metrics:
    # 取最大值减去原值，使其越大越好
    max_val = metrics_for_radar[col].max()
    metrics_for_radar[col] = max_val - metrics_for_radar[col]
# 归一化
scaled = scaler.fit_transform(metrics_for_radar)
scaled_df = pd.DataFrame(scaled, index=metrics_df.index, columns=metrics_df.columns)

# 雷达图
categories = scaled_df.columns.tolist()
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # 闭合

fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
for i, row in scaled_df.iterrows():
    values = row.values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=i)
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9)
ax.set_title('Normalized Performance Radar Chart (Higher is better)', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('ablation_radar.png', dpi=300)
plt.show()

print("\n可视化图片已保存：ablation_metrics_comparison.png, ablation_radar.png")