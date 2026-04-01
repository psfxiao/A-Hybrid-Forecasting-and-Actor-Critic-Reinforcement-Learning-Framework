import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os
import warnings
import calendar
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans
import random

warnings.filterwarnings('ignore')

# Read data - Option to use single year or multi-year data
USE_MULTI_YEAR = False  # Set to False to use only 2025 data, True to use 2024+2025 data

file_path_2024 = "The daily fabric outbound volume at all time in 2024.xlsx"
file_path_2025 = "The daily fabric outbound volume at all time in 2025.xlsx"

if not os.path.exists(file_path_2025):
    raise FileNotFoundError(f"File does not exist: {file_path_2025}")


def read_data(file_path):
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file format")


# Read data
if USE_MULTI_YEAR and os.path.exists(file_path_2024):
    df_2024 = read_data(file_path_2024)
    df_2025 = read_data(file_path_2025)
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    print("Using 2024+2025 data for prediction")
else:
    df = read_data(file_path_2025)
    print("Using only 2025 data for prediction")

# ===================== Data Preprocessing =====================
# 1. Date processing
print("Original data columns:", df.columns)
print("Original data types:", df.dtypes)

# Date conversion
try:
    df['Date'] = pd.to_datetime(df['date'])
    print("Direct date conversion successful")
except Exception as e:
    print(f"Date conversion issue: {e}")
    # Simplified date processing
    df['Date'] = pd.to_datetime(df['date'], errors='coerce')

# Remove any rows with invalid dates
df = df.dropna(subset=['Date'])
print(f"After date cleaning, {len(df)} rows remaining")

# 2. Extract time period information
time_periods = df['time period'].unique()
print(f"Time periods list: {time_periods}")


# 3. Enhanced order day detection
def is_order_day(date):
    return date.day in [1, 5, 10, 15]


order_days = df[df['Date'].apply(is_order_day)].copy()
order_days['Date'] = pd.to_datetime(order_days['Date'].dt.date)

# Group by date and time period
daily_time_period = order_days.groupby(['Date', 'time period'])[
    'Total fabric outbound quantity (unit: rolls)'].sum().unstack()
daily_time_period.fillna(0, inplace=True)

# Calculate daily total outbound volume
daily_total = daily_time_period.sum(axis=1)

# 4. Create monthly data with enhanced filtering
monthly_total = daily_total.resample('M').sum()
# Remove months with insufficient data (less than 2 order days)
month_counts = daily_total.resample('M').count()
monthly_total = monthly_total[month_counts >= 2]

print(f"Monthly data range: {monthly_total.index.min()} to {monthly_total.index.max()}")
print(f"Monthly data points: {len(monthly_total)}")


# ===================== Enhanced SARIMA-Markov Model =====================
class EnhancedSARIMAMarkov:
    def __init__(self, sarima_order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), n_states=3):
        self.sarima_order = sarima_order
        self.seasonal_order = seasonal_order
        self.n_states = n_states
        self.state_means = None
        self.state_vars = None
        self.transition_matrix = None

    def find_best_parameters(self, data):
        """Automatically find better SARIMA parameters"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, 12)

        # Simple parameter search
        for p in [0, 1, 2]:
            for q in [0, 1, 2]:
                try:
                    model = SARIMAX(data, order=(p, 1, q),
                                    seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    results = model.fit(disp=False)
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, 1, q)
                except:
                    continue

        return best_order, best_seasonal_order

    def fit(self, train_data):
        # Automatic parameter selection
        self.sarima_order, self.seasonal_order = self.find_best_parameters(train_data)
        print(f"Selected SARIMA order: {self.sarima_order}, seasonal order: {self.seasonal_order}")

        # SARIMA model fitting
        self.model = SARIMAX(train_data,
                             order=self.sarima_order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False)
        self.results = self.model.fit(disp=False)

        # Get residuals
        residuals = self.results.resid.dropna()

        if len(residuals) < 10:
            print(f"Warning: Insufficient residuals for Markov modeling")
            self.simple_sarima = True
            return

        self.simple_sarima = False

        # Enhanced state discretization with K-means
        if len(residuals) >= self.n_states:
            try:
                # Use K-means for better state division
                kmeans = KMeans(n_clusters=self.n_states, random_state=42)
                states = kmeans.fit_predict(residuals.values.reshape(-1, 1))

                self.state_means = []
                self.state_vars = []
                for i in range(self.n_states):
                    state_residuals = residuals[states == i]
                    if len(state_residuals) > 0:
                        self.state_means.append(state_residuals.mean())
                        self.state_vars.append(state_residuals.var())
                    else:
                        self.state_means.append(0)
                        self.state_vars.append(0)

                # Calculate transition matrix with smoothing
                self.transition_matrix = np.ones((self.n_states, self.n_states)) * 0.1  # Smoothing factor
                for t in range(len(states) - 1):
                    i = int(states[t])
                    j = int(states[t + 1])
                    self.transition_matrix[i, j] += 1

                # Row normalization
                row_sums = self.transition_matrix.sum(axis=1)
                row_sums[row_sums == 0] = 1
                self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

            except Exception as e:
                print(f"K-means clustering failed: {e}, using quantile discretization")
                self._fallback_discretization(residuals)
        else:
            self._fallback_discretization(residuals)

    def _fallback_discretization(self, residuals):
        """Fallback to quantile discretization"""
        self.discretizer = KBinsDiscretizer(n_bins=self.n_states,
                                            encode='ordinal',
                                            strategy='quantile')
        states = self.discretizer.fit_transform(residuals.values.reshape(-1, 1)).flatten().astype(int)

        self.state_means = []
        self.state_vars = []
        for i in range(self.n_states):
            state_residuals = residuals[states == i]
            if len(state_residuals) > 0:
                self.state_means.append(state_residuals.mean())
                self.state_vars.append(state_residuals.var())
            else:
                self.state_means.append(0)
                self.state_vars.append(0)

        # Transition matrix
        self.transition_matrix = np.zeros((self.n_states, self.n_states))
        for t in range(len(states) - 1):
            i = int(states[t])
            j = int(states[t + 1])
            self.transition_matrix[i, j] += 1

        row_sums = self.transition_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

    def predict(self, steps, alpha=0.05):
        # SARIMA prediction
        forecast = self.results.get_forecast(steps=steps)
        sarima_mean = forecast.predicted_mean
        sarima_var = forecast.var_pred_mean

        if self.simple_sarima or not hasattr(self, 'transition_matrix'):
            # Simple confidence interval calculation
            conf_int = forecast.conf_int(alpha=alpha)
            return pd.DataFrame({
                'Predicted Value': sarima_mean.values,
                'Lower Bound': conf_int.iloc[:, 0],
                'Upper Bound': conf_int.iloc[:, 1]
            }, index=sarima_mean.index)

        # Markov correction with damping factor for long-term predictions
        last_resid = self.results.resid[-1]
        if hasattr(self, 'discretizer'):
            current_state = self.discretizer.transform([[last_resid]])[0, 0].astype(int)
        else:
            # If no discretizer, use recent state
            current_state = 0

        state_probs = np.zeros((steps, self.n_states))
        state_probs[0] = self.transition_matrix[current_state]

        # Apply damping factor to avoid divergence in long-term predictions
        damping_factor = 0.8
        for h in range(1, steps):
            state_probs[h] = state_probs[h - 1] @ self.transition_matrix
            # Reduce Markov correction impact as prediction steps increase
            state_probs[h] = state_probs[h] * (damping_factor ** h)

        # Markov correction
        markov_correction = state_probs @ np.array(self.state_means)
        combined_variance = sarima_var + (state_probs @ np.array(self.state_vars))

        # Combined prediction with bounded correction
        max_correction_ratio = 0.3  # Maximum correction ratio
        correction_bounded = np.clip(markov_correction,
                                     -max_correction_ratio * sarima_mean.values,
                                     max_correction_ratio * sarima_mean.values)

        combined_mean = sarima_mean.values + correction_bounded

        # Confidence interval
        z = 1.96
        lower = combined_mean - z * np.sqrt(combined_variance)
        upper = combined_mean + z * np.sqrt(combined_variance)

        forecast_dates = pd.date_range(start=monthly_total.index[-1] + pd.DateOffset(months=1),
                                       periods=steps, freq='M')

        return pd.DataFrame({
            'Predicted Value': combined_mean,
            'Lower Bound': lower,
            'Upper Bound': upper
        }, index=forecast_dates)


# ===================== Enhanced Hierarchical Forecasting Model =====================
class EnhancedDemandForecaster:
    def __init__(self, n_time_periods, forecast_steps=12):
        self.n_time_periods = n_time_periods
        self.forecast_steps = forecast_steps
        self.time_period_models = {}
        self.monthly_model = None
        self.daily_pattern = {}
        self.time_period_pattern = {}
        self.day_of_month_variation = {}
        self.month_names = [calendar.month_name[i] for i in range(1, 13)]

    def _calculate_day_variation(self, daily_data, daily_time_data):
        """Calculate variation patterns for different days within the same month"""
        day_variation = {}

        for month in range(1, 13):
            month_data = daily_data[daily_data.index.month == month]
            if len(month_data) > 0:
                # Calculate relative variation for each day within the month
                month_mean = month_data.mean()
                if month_mean > 0:
                    day_ratios = {}
                    for day in [1, 5, 10, 15]:
                        day_data = month_data[month_data.index.day == day]
                        if len(day_data) > 0:
                            day_ratios[day] = day_data.mean() / month_mean
                        else:
                            day_ratios[day] = 1.0  # Default ratio

                    # Normalize to ensure sum is reasonable
                    total_ratio = sum(day_ratios.values())
                    if total_ratio > 0:
                        for day in day_ratios:
                            day_ratios[day] = day_ratios[day] / total_ratio * len(day_ratios)

                    day_variation[month] = day_ratios

        return day_variation

    def _enhance_time_period_pattern(self, daily_time_data, daily_data):
        """Enhanced time period pattern recognition, focusing on peak periods"""
        enhanced_pattern = {}

        for time_period in daily_time_data.columns:
            # Calculate basic ratios
            ratios = daily_time_data[time_period] / daily_data
            ratios.replace([np.inf, -np.inf], np.nan, inplace=True)
            ratios.fillna(0, inplace=True)

            # Group by month, using median to reduce outlier impact
            monthly_ratios = ratios.groupby(ratios.index.month).median()

            # Identify peak periods and enhance their patterns
            if '15:00-16:00' in time_period or '14:00-16:00' in time_period:
                # Apply smoothing and enhancement for peak periods
                monthly_ratios = monthly_ratios * 1.1  # Appropriate enhancement
                monthly_ratios = monthly_ratios.clip(upper=0.8)  # Set upper limit

            # Fill missing months
            full_index = pd.Index(range(1, 13), name='month')
            monthly_ratios = monthly_ratios.reindex(full_index, fill_value=0.1)

            # Apply smoothing
            monthly_ratios = monthly_ratios.rolling(window=3, center=True, min_periods=1).mean()

            enhanced_pattern[time_period] = monthly_ratios.to_dict()

        return enhanced_pattern

    def fit(self, daily_data, daily_time_data, monthly_data):
        print("Training enhanced forecasting model...")

        # 1. Train enhanced monthly model
        monthly_model = EnhancedSARIMAMarkov(
            sarima_order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            n_states=3
        )
        monthly_model.fit(monthly_data)
        self.monthly_model = monthly_model

        # 2. Calculate day-of-month variation patterns
        self.day_of_month_variation = self._calculate_day_variation(daily_data, daily_time_data)

        # 3. Calculate enhanced daily demand pattern
        for day in [1, 5, 10, 15]:
            day_data = daily_data[daily_data.index.day == day]
            if len(day_data) > 0:
                monthly_totals = daily_data.resample('M').sum().reindex(monthly_data.index, fill_value=0)
                monthly_totals[monthly_totals == 0] = 1

                ratios = []
                months = []
                for date, value in day_data.items():
                    month_start = pd.Timestamp(year=date.year, month=date.month, day=1)
                    if month_start in monthly_totals.index:
                        month_total = monthly_totals.loc[month_start]
                        if month_total > 0:
                            ratio = value / month_total
                            ratios.append(ratio)
                            months.append(date.month)

                if ratios:
                    ratio_df = pd.DataFrame({'month': months, 'ratio': ratios})
                    # Use median to reduce outlier impact
                    monthly_avg = ratio_df.groupby('month')['ratio'].median()

                    full_index = pd.Index(range(1, 13), name='month')
                    monthly_avg = monthly_avg.reindex(full_index, fill_value=0.1)

                    # Apply day variation adjustment
                    for month in monthly_avg.index:
                        if month in self.day_of_month_variation and day in self.day_of_month_variation[month]:
                            monthly_avg.loc[month] = monthly_avg.loc[month] * self.day_of_month_variation[month][day]

                    monthly_avg = monthly_avg.clip(upper=0.3)  # Set single day ratio upper limit
                    self.daily_pattern[day] = monthly_avg.to_dict()
                else:
                    self.daily_pattern[day] = {m: 0.1 for m in range(1, 13)}

        # 4. Calculate enhanced time period pattern
        self.time_period_pattern = self._enhance_time_period_pattern(daily_time_data, daily_data)

        print("Model training completed")

    def predict(self):
        print("Generating predictions...")

        # Predict monthly total
        monthly_forecast = self.monthly_model.predict(steps=self.forecast_steps)

        if monthly_forecast.empty:
            raise ValueError("Monthly prediction result is empty")

        # Generate forecast dates
        forecast_dates = []
        start_date = daily_time_period.index[-1] + relativedelta(months=1)

        for _ in range(self.forecast_steps):
            for day in [1, 5, 10, 15]:
                try:
                    date = start_date.replace(day=day)
                    forecast_dates.append(date)
                except ValueError:
                    continue
            start_date = start_date + relativedelta(months=1)

        if not forecast_dates:
            raise ValueError("No valid prediction dates generated")

        # Create prediction DataFrame
        forecast_df = pd.DataFrame(index=forecast_dates, columns=daily_time_period.columns)

        # Fill prediction values with realistic variation
        for date in forecast_dates:
            month_str = date.strftime('%Y-%m')

            if month_str in monthly_forecast.index:
                # Extract scalar value from the Series
                month_value = monthly_forecast.loc[month_str, 'Predicted Value']
                if isinstance(month_value, pd.Series):
                    month_value = month_value.iloc[0]
            else:
                # Use weighted average
                similar_months = [m for m in monthly_forecast.index if m.endswith(f'-{date.month:02d}')]
                if similar_months:
                    month_value = monthly_forecast.loc[similar_months, 'Predicted Value'].mean()
                    if isinstance(month_value, pd.Series):
                        month_value = month_value.iloc[0]
                else:
                    month_value = monthly_forecast['Predicted Value'].mean()
                    if isinstance(month_value, pd.Series):
                        month_value = month_value.iloc[0]

            # Ensure month_value is a scalar
            if hasattr(month_value, 'item'):
                month_value = month_value.item()

            # Calculate daily total with realistic variation
            day_pattern = self.daily_pattern.get(date.day, {})
            day_ratio = day_pattern.get(date.month, 0.1)

            # Add random variation to make predictions more realistic
            variation_factor = 0.9 + 0.2 * random.random()  # Random factor between 0.9 and 1.1
            day_ratio = day_ratio * variation_factor

            day_total = month_value * day_ratio

            # Ensure daily total is reasonable
            max_daily_ratio = 0.3  # Single day maximum 30% of monthly total
            day_total = min(day_total, month_value * max_daily_ratio)

            # Allocate time period demand with realistic variation
            total_allocated = 0
            peak_periods = ['15:00-16:00', '14:00-16:00']  # Define peak periods

            for time_period in daily_time_period.columns:
                time_pattern = self.time_period_pattern.get(time_period, {})
                time_ratio = time_pattern.get(date.month, 0.1)

                # Enhance prediction for peak periods
                if any(peak in time_period for peak in peak_periods):
                    time_ratio = time_ratio * 1.1  # Peak period enhancement 10%

                # Add small random variation to time period ratios
                time_variation = 0.95 + 0.1 * random.random()  # Random factor between 0.95 and 1.05
                time_ratio = time_ratio * time_variation

                forecast_value = day_total * time_ratio

                if isinstance(forecast_value, (pd.Series, np.ndarray)):
                    forecast_value = forecast_value.item() if hasattr(forecast_value, 'item') else forecast_value[0]

                forecast_df.loc[date, time_period] = forecast_value
                total_allocated += forecast_value

        return forecast_df, monthly_forecast


# ===================== Model Training and Prediction =====================
# Initialize and train enhanced model
forecaster = EnhancedDemandForecaster(n_time_periods=len(time_periods), forecast_steps=12)
forecaster.fit(daily_total, daily_time_period, monthly_total)

# Predict demand for 2026
forecast_results, monthly_forecast = forecaster.predict()

# ===================== Fixed Visualization Results =====================
# 1. Monthly forecast visualization
plt.figure(figsize=(15, 10))

# Convert indices to proper datetime format for plotting
monthly_total_index = pd.to_datetime(monthly_total.index)
monthly_forecast_index = pd.to_datetime(monthly_forecast.index)

# Historical monthly data
plt.subplot(2, 2, 1)
plt.plot(monthly_total_index, monthly_total.values, 'bo-', label='Historical Monthly Total', markersize=6, linewidth=2)
plt.plot(monthly_forecast_index, monthly_forecast['Predicted Value'], 'r-o',
         label='Monthly Forecast', markersize=6, linewidth=2)
plt.fill_between(monthly_forecast_index,
                 monthly_forecast['Lower Bound'],
                 monthly_forecast['Upper Bound'],
                 color='pink', alpha=0.3, label='95% Confidence Interval')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()
plt.title('Monthly Fabric Outbound Volume Forecast (SARIMA-Markov Model)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Monthly Total Outbound Volume (rolls)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# 2. Daily demand pattern visualization
plt.subplot(2, 2, 2)
daily_pattern_data = pd.DataFrame(forecaster.daily_pattern).T
daily_pattern_data.columns = [calendar.month_name[i] for i in daily_pattern_data.columns]
daily_pattern_data.plot(kind='bar', width=0.8, ax=plt.gca())
plt.title('Order Day Demand Proportion Pattern by Month', fontsize=14)
plt.xlabel('Day')
plt.ylabel('Proportion')
plt.xticks(ticks=[0, 1, 2, 3], labels=['1st', '5th', '10th', '15th'], rotation=0)
plt.legend(title='Month', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)

# 3. Time period demand pattern visualization
plt.subplot(2, 2, 3)
time_pattern_data = pd.DataFrame(forecaster.time_period_pattern).T
time_pattern_data.columns = [calendar.month_name[i] for i in time_pattern_data.columns]
time_pattern_data.plot(kind='bar', width=0.8, ax=plt.gca())
plt.title('Time Period Demand Proportion Pattern by Month', fontsize=14)
plt.xlabel('Time Period')
plt.ylabel('Proportion')
plt.xticks(rotation=45)
plt.legend(title='Month', fontsize=8)
plt.grid(True, linestyle='--', alpha=0.7)

# 4. Confidence interval statistics
plt.subplot(2, 2, 4)
ci_widths = monthly_forecast['Upper Bound'] - monthly_forecast['Lower Bound']
months = [calendar.month_name[i + 1] for i in range(len(ci_widths))]
plt.bar(months, ci_widths, color='lightblue', alpha=0.7)
plt.axhline(y=ci_widths.mean(), color='red', linestyle='--', label=f'Average Width: {ci_widths.mean():.1f}')
plt.title('Monthly Forecast Confidence Interval Width', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Confidence Interval Width')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Forecast result visualization (first 3 months forecast)
sample_dates = forecast_results.index[:12]
plt.figure(figsize=(16, 12))
for i, date in enumerate(sample_dates, 1):
    plt.subplot(3, 4, i)
    date_data = forecast_results.loc[date]
    # Convert to numeric type
    date_data_numeric = pd.to_numeric(date_data, errors='coerce')
    date_data_numeric.plot(kind='bar', color='skyblue', alpha=0.8)
    plt.title(f'{date.strftime("%Y-%m-%d")} Forecast', fontsize=10)
    plt.xlabel('Time Period')
    plt.ylabel('Demand (rolls)')
    plt.xticks(rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)

    # Add value labels
    for j, v in enumerate(date_data_numeric):
        if not np.isnan(v):
            plt.text(j, v, f'{int(v)}', ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('order_day_forecast_samples.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Monthly time period demand forecast (heatmap)
monthly_time_forecast = pd.DataFrame(index=forecast_results.index.month.unique(),
                                     columns=forecast_results.columns)

for month in monthly_time_forecast.index:
    month_data = forecast_results[forecast_results.index.month == month]
    monthly_time_forecast.loc[month] = month_data.apply(pd.to_numeric, errors='coerce').mean()

monthly_time_forecast.index = [calendar.month_name[i] for i in monthly_time_forecast.index]
heatmap_data = monthly_time_forecast.astype(float)

plt.figure(figsize=(14, 10))
plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
plt.colorbar(label='Demand (rolls)')
plt.title('Monthly Time Period Demand Forecast Heatmap', fontsize=16)
plt.xlabel('Time Period')
plt.ylabel('Month')

plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns, rotation=45)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

# Add value labels
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        value = heatmap_data.iloc[i, j]
        if not np.isnan(value):
            plt.text(j, i, f'{int(value)}', ha='center', va='center',
                     color='white' if value > heatmap_data.values.max() * 0.6 else 'black',
                     fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('monthly_time_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Forecast trend analysis
plt.figure(figsize=(15, 10))

# Monthly forecast trend
plt.subplot(2, 2, 1)
plt.plot(monthly_forecast_index, monthly_forecast['Predicted Value'],
         'ro-', linewidth=2, markersize=6, label='Predicted Value')
plt.fill_between(monthly_forecast_index,
                 monthly_forecast['Lower Bound'],
                 monthly_forecast['Upper Bound'],
                 color='red', alpha=0.2, label='Confidence Interval')
plt.title('2026 Monthly Forecast Trend', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Monthly Demand (rolls)')
plt.grid(True, alpha=0.3)
plt.legend()

# Seasonal analysis
plt.subplot(2, 2, 2)
seasonal_pattern = monthly_forecast['Predicted Value'].values
months = [calendar.month_name[i + 1] for i in range(12)]
plt.plot(months, seasonal_pattern, 'gs-', linewidth=2, markersize=6)
plt.title('2026 Seasonal Pattern', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Monthly Demand (rolls)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Forecast uncertainty analysis
plt.subplot(2, 2, 3)
uncertainty = (monthly_forecast['Upper Bound'] - monthly_forecast['Lower Bound']) / monthly_forecast[
    'Predicted Value'] * 100
plt.bar(months, uncertainty, color='orange', alpha=0.7)
plt.axhline(y=uncertainty.mean(), color='red', linestyle='--',
            label=f'Average Uncertainty: {uncertainty.mean():.1f}%')
plt.title('Forecast Relative Uncertainty (%)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Uncertainty (%)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# Cumulative demand forecast
plt.subplot(2, 2, 4)
cumulative_demand = monthly_forecast['Predicted Value'].cumsum()
plt.plot(months, cumulative_demand, 'mo-', linewidth=2, markersize=6, color='purple')
plt.title('2026 Cumulative Demand Forecast', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Cumulative Demand (rolls)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add cumulative value labels
for i, v in enumerate(cumulative_demand):
    plt.text(i, v, f'{int(v)}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('forecast_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Enhanced comparison visualization
plt.figure(figsize=(16, 10))

# Plot historical time period patterns
plt.subplot(2, 2, 1)
historical_avg = daily_time_period.mean()
historical_avg.plot(kind='bar', color='lightblue', alpha=0.7, label='Historical Average')
plt.title('Historical Average by Time Period')
plt.xticks(rotation=45)
plt.ylabel('Average Demand (rolls)')

# Plot predicted patterns
plt.subplot(2, 2, 2)
predicted_avg = forecast_results.mean()
predicted_avg.plot(kind='bar', color='salmon', alpha=0.7, label='Predicted Average')
plt.title('Predicted Average by Time Period')
plt.xticks(rotation=45)
plt.ylabel('Average Demand (rolls)')

# Plot peak period comparison
plt.subplot(2, 2, 3)
peak_period = '15:00-16:00'  # Assuming this is the peak period
if peak_period in daily_time_period.columns:
    historical_peak = daily_time_period[peak_period]
    predicted_peak = forecast_results[peak_period]

    plt.plot(historical_peak.index, historical_peak, 'bo-', label='Historical Peak', alpha=0.7)
    plt.plot(predicted_peak.index, predicted_peak, 'ro-', label='Predicted Peak', alpha=0.7)
    plt.title(f'Peak Period ({peak_period}) Comparison')
    plt.xticks(rotation=45)
    plt.legend()

# Model performance summary
plt.subplot(2, 2, 4)
performance_metrics = {
    'Data Used': '2024+2025' if USE_MULTI_YEAR else '2025 Only',
    'Total Forecast': f"{monthly_forecast['Predicted Value'].sum():.0f} rolls",
    'Avg Monthly': f"{monthly_forecast['Predicted Value'].mean():.0f} rolls",
    'Peak Enhancement': 'Applied',
    'Day Variation': 'Enabled'
}
plt.text(0.1, 0.8, '\n'.join([f'{k}: {v}' for k, v in performance_metrics.items()]),
         transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')
plt.title('Model Performance Summary')

plt.tight_layout()
plt.savefig('enhanced_comparison_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== Forecast Result Output =====================
# Organize forecast results
final_forecast = forecast_results.stack().reset_index()
final_forecast.columns = ['Date', 'Time Period', 'Predicted Demand (rolls)']
final_forecast['Month'] = final_forecast['Date'].dt.month_name()
final_forecast['Date'] = final_forecast['Date'].dt.date

# Save only the detailed forecast results
final_forecast.to_csv('order_day_time_period_demand_forecast.csv', index=False, encoding='utf-8-sig')

print("=" * 60)
print("Enhanced Forecasting Completed Successfully!")
print(f"Data Used: {'2024+2025' if USE_MULTI_YEAR else '2025 Only'}")
print("Peak period patterns have been enhanced in the predictions")
print("Day-to-day variation has been added to make predictions more realistic")
print("=" * 60)
print("\nOrder Day Time Period Demand Forecast Results (first 15 rows):")
print(final_forecast.head(15))
print("=" * 60)
print(f"Prediction results saved to: order_day_time_period_demand_forecast.csv")
print(f"Visualizations saved as PNG files")