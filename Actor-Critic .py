import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Preparation and Preprocessing
historical_data_2025 = pd.read_excel("The daily fabric outbound volume at all time in 2025.xlsx")

# Combine historical data
historical_data = pd.concat([historical_data_2025], ignore_index=True)
historical_data['date'] = pd.to_datetime(historical_data['date'])

# Calculate average demand for each time period
time_group = historical_data.groupby('time period')['Total fabric outbound quantity (unit: rolls)'].mean().reset_index()
time_group.rename(columns={'Total fabric outbound quantity (unit: rolls)': 'target_demand'}, inplace=True)

# Read 2026 forecast data
forecast_data = pd.read_csv("order_day_time_period_demand_forecast.csv")
forecast_data['Date'] = pd.to_datetime(forecast_data['Date'])
# Fix column name mismatch
forecast_data = forecast_data.merge(time_group, left_on='Time Period', right_on='time period')


# 2. Improved Static Discount Scheme Calculation
def calculate_static_discount(forecast_df, elasticity=-1.5):
    """Calculate static discount for each time period with improved logic"""
    peak_hours = ['15:00-16:00', '16:00-17:00']

    results = []
    time_slots = forecast_df['time period'].unique()

    print("\nComparison of target demand and average forecast demand by time period:")
    for slot in time_slots:
        slot_data = forecast_df[forecast_df['time period'] == slot]
        avg_forecast = slot_data['Predicted Demand (rolls)'].mean()
        target = slot_data['target_demand'].iloc[0]
        demand_ratio = avg_forecast / target if target > 0 else 1
        print(f"Time period {slot}: Target={target:.2f}, Forecast={avg_forecast:.2f}, Ratio={demand_ratio:.3f}")

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
        discount = round(discount, 4)

        results.append({
            'time period': slot,
            'discount': discount,
            'avg_forecast': avg_forecast,
            'target_demand': T,
            'demand_gap_ratio': (T - avg_forecast) / T if T > 0 else 0
        })

    result_df = pd.DataFrame(results)

    print("\nDetailed static discount analysis:")
    for _, row in result_df.iterrows():
        print(f"Time period {row['time period']}: Discount={row['discount']:.4f}, "
              f"Target={row['target_demand']:.2f}, Forecast={row['avg_forecast']:.2f}, "
              f"Gap Ratio={row['demand_gap_ratio']:.3f}")

    return result_df[['time period', 'discount']]


# Calculate static discount
static_discount_df = calculate_static_discount(forecast_data)
static_discount_df.to_excel("static_time_period_discount_scheme.xlsx", index=False)
print("Static discount scheme saved to file: static_time_period_discount_scheme.xlsx")

# Merge static discount scheme
forecast_data['time period'] = forecast_data['time period'].astype(str)
static_discount_df['time period'] = static_discount_df['time period'].astype(str)
forecast_data = forecast_data.merge(static_discount_df, on='time period', how='left')
forecast_data['discount'] = forecast_data['discount'].fillna(0.0)


# 3. Alternative discount calculation method
def calculate_alternative_discount(forecast_df, elasticity=-1.5):
    """Alternative method using weighted average approach"""
    peak_hours = ['15:00-16:00', '16:00-17:00']

    results = []
    time_slots = forecast_df['time period'].unique()

    print("\nAlternative Discount Calculation")

    for slot in time_slots:
        slot_data = forecast_df[forecast_df['time period'] == slot]
        Q = slot_data['Predicted Demand (rolls)'].values
        T = slot_data['target_demand'].values[0]

        if slot in peak_hours:
            discount = 0.0
        else:
            daily_discounts = []
            weights = []

            for _, row in slot_data.iterrows():
                Q_day = row['Predicted Demand (rolls)']
                T_day = row['target_demand']

                if T_day > 0 and Q_day > 0:
                    required_change = (T_day - Q_day) / Q_day
                    daily_discount = required_change / elasticity if elasticity != 0 else 0
                    daily_discounts.append(daily_discount)
                    weights.append(Q_day)

            if daily_discounts:
                discount = np.average(daily_discounts, weights=weights)
            else:
                discount = 0.0

        discount = np.clip(discount, -0.2, 0.2)
        discount = round(discount, 4)

        results.append({'time period': slot, 'discount_alternative': discount})

    alt_df = pd.DataFrame(results)

    print("\nAlternative discount scheme:")
    for _, row in alt_df.iterrows():
        print(f"Time period {row['time period']}: Discount={row['discount_alternative']:.4f}")

    return alt_df


# Calculate alternative discounts for comparison
alternative_discount_df = calculate_alternative_discount(forecast_data)

# Choose the better discount scheme
final_discount_df = static_discount_df.merge(alternative_discount_df, on='time period')
final_discount_df['discount_final'] = (final_discount_df['discount'] + final_discount_df['discount_alternative']) / 2
final_discount_df['discount_final'] = final_discount_df['discount_final'].clip(-0.2, 0.2)
final_discount_df['discount_final'] = final_discount_df['discount_final'].round(4)

print("\nFinal discount scheme (averaged):")
for _, row in final_discount_df.iterrows():
    print(f"Time period {row['time period']}: Discount={row['discount_final']:.4f}")

# Update forecast_data with final discounts
forecast_data = forecast_data.merge(final_discount_df[['time period', 'discount_final']], on='time period')
forecast_data['discount'] = forecast_data['discount_final']
forecast_data = forecast_data.drop('discount_final', axis=1)


# 4. Calculate static discount demand results
def calculate_static_demand_results(forecast_df, elasticity=-1.5):
    """Calculate demand results under static discount scheme"""
    results = []

    for index, row in forecast_df.iterrows():
        forecast_demand = row['Predicted Demand (rolls)']
        target_demand = row['target_demand']
        time_slot = row['time period']
        discount = row['discount']

        adjusted_demand = forecast_demand * (1 + elasticity * discount)

        demand_diff = abs(adjusted_demand - target_demand)
        relative_demand_diff = demand_diff / max(target_demand, 1)
        discount_diff = 0

        results.append({
            'date': row['Date'],
            'time period': time_slot,
            'forecast_demand': forecast_demand,
            'target_demand': target_demand,
            'adjusted_demand': adjusted_demand,
            'price_discount': discount,
            'relative_demand_diff': relative_demand_diff,
            'discount_diff': discount_diff
        })

    result_df = pd.DataFrame(results)

    daily_totals = result_df.groupby('date').agg({
        'forecast_demand': 'sum',
        'adjusted_demand': 'sum'
    }).reset_index()
    daily_totals['adjustment_ratio'] = daily_totals['forecast_demand'] / daily_totals['adjusted_demand']

    result_df = result_df.merge(daily_totals[['date', 'adjustment_ratio']], on='date')
    result_df['price_adjusted_demand'] = result_df['adjusted_demand'] * result_df['adjustment_ratio']
    result_df = result_df.drop('adjustment_ratio', axis=1)

    return result_df


# Calculate and save static discount results
print("\nCalculating static discount demand results...")
static_demand_results = calculate_static_demand_results(forecast_data)
static_demand_results.to_excel("static_discount_demand_results.xlsx", index=False)
print("Static discount demand results saved to file: static_discount_demand_results.xlsx")


# 5. Analyze discount effectiveness
def analyze_discount_effectiveness(static_results):
    """Analyze how effective the discounts are"""
    effectiveness = static_results.groupby('time period').agg({
        'forecast_demand': 'mean',
        'target_demand': 'mean',
        'adjusted_demand': 'mean',
        'price_discount': 'mean',
        'relative_demand_diff': 'mean'
    }).reset_index()

    effectiveness['improvement'] = (effectiveness['adjusted_demand'] - effectiveness['forecast_demand']) / \
                                   effectiveness['forecast_demand']
    effectiveness['target_achievement'] = effectiveness['adjusted_demand'] / effectiveness['target_demand']

    print("\nDiscount Effectiveness Analysis:")
    print("=" * 80)
    for _, row in effectiveness.iterrows():
        print(f"Time {row['time period']}: Discount={row['price_discount']:.4f}, "
              f"Forecast={row['forecast_demand']:.1f} -> Adjusted={row['adjusted_demand']:.1f}, "
              f"Target={row['target_demand']:.1f}, Achievement={row['target_achievement']:.3f}")

    return effectiveness


effectiveness_analysis = analyze_discount_effectiveness(static_demand_results)


# 6. Reinforcement Learning Environment
class DemandSmoothingEnv:
    def __init__(self, data, elasticity=-1.5):
        self.data = data
        self.elasticity = elasticity
        self.time_slots = data['time period'].unique()
        self.static_discounts = self.calculate_static_discounts()
        self.reset()
        self.episode_counter = 0

    def calculate_static_discounts(self):
        static_discounts = {}
        for index, row in self.data.iterrows():
            slot = row['time period']
            if slot not in static_discounts:
                static_discounts[slot] = row['discount']
        return static_discounts

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
            return np.array([
                self.current_step / len(self.time_slots),
                current_row['Predicted Demand (rolls)'] / 2000,
                current_row['target_demand'] / 2000,
                np.mean(self.adjusted_demands) / 2000 if self.adjusted_demands else 0,
                len(self.adjusted_demands) / len(self.time_slots)
            ])
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
            discount = np.clip(static_discount + action, static_discount - 0.05, static_discount + 0.05)

        adjusted_demand = forecast_demand * (1 + self.elasticity * discount)
        self.adjusted_demands.append(adjusted_demand)

        demand_diff = abs(adjusted_demand - target_demand)
        relative_demand_diff = demand_diff / max(target_demand, 1)
        discount_diff = abs(discount - static_discount)

        base_reward = 1 / (1 + relative_demand_diff * 10)
        demand_penalty = - (relative_demand_diff * 5) ** 2
        discount_penalty = - (discount_diff * 20) ** 2

        reward = base_reward + demand_penalty + discount_penalty
        decay_factor = 1.0 - min(0.5, self.episode_counter * 0.0002)
        initial_amplification_factor = max(0.1, 1 - self.episode_counter / 1000)
        reward *= decay_factor * initial_amplification_factor

        self.current_step += 1
        done = self.current_step >= len(self.day_data)

        if done:
            self.episode_counter += 1

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


# 7. Simplified Actor-Critic using NumPy
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


# 8. Simplified Agent
class SimpleACAgent:
    def __init__(self, state_dim, gamma=0.99):
        self.actor = SimpleActor(state_dim)
        self.critic = SimpleCritic(state_dim)
        self.gamma = gamma
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

    def select_action(self, state):
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

    def train(self, env, episodes=2000):
        episode_rewards = []

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action, _ = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.rewards.append(reward)
                total_reward += reward
                state = next_state

            self.update()
            episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode >= 100 else np.mean(episode_rewards)
                print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}")

        plt.figure(figsize=(10, 6))
        plt.plot(episode_rewards, label='Episode Reward')

        window_size = 100
        moving_avg = [np.mean(episode_rewards[max(0, i - window_size):i + 1])
                      for i in range(len(episode_rewards))]
        plt.plot(moving_avg, 'r-', linewidth=2, label=f'{window_size}-episode Moving Average')

        plt.title("Training Progress - Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_curve_simple.png")
        plt.close()

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
        total_adjusted_demand = day_df['adjusted_demand'].sum()
        total_forecast_demand = day_df['forecast_demand'].sum()
        adjustment_ratio = total_forecast_demand / total_adjusted_demand
        day_df['price_adjusted_demand'] = day_df['adjusted_demand'] * adjustment_ratio

        return day_df


# 9. Main Execution
if __name__ == "__main__":
    # Create environment
    env = DemandSmoothingEnv(forecast_data)

    # Initialize and train agent - ITERATIONS SET HERE (episodes=2000)
    agent = SimpleACAgent(state_dim=5)
    print("\nStarting simplified reinforcement learning model training...")
    agent.train(env, episodes=5000)  # Iterations set here
    print("Training completed!")

    # Generate dynamic discount scheme
    dynamic_results = []
    unique_days = forecast_data['Date'].dt.date.unique()

    print("\nGenerating dynamic discount scheme...")
    for i, day in enumerate(unique_days):
        if (i + 1) % 10 == 0 or i == 0 or i == len(unique_days) - 1:
            print(f"Processing date: {day} ({i + 1}/{len(unique_days)})")
        day_df = agent.predict(env, day)
        dynamic_results.append(day_df)

    dynamic_discount_df = pd.concat(dynamic_results)
    dynamic_discount_df.to_excel("dynamic_time_period_discount_scheme_simple.xlsx", index=False)

    print("\nResults saved to files:")
    print("1. dynamic_time_period_discount_scheme_simple.xlsx")
    print("2. static_time_period_discount_scheme.xlsx")
    print("3. static_discount_demand_results.xlsx")
    print("4. training_curve_simple.png")