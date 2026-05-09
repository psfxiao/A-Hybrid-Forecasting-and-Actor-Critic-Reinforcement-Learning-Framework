import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import os
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')

# Read data
file_path_2024 = "The daily fabric outbound volume at all time in 2024.xlsx"
file_path_2025 = "The daily fabric outbound volume at all time in 2025.xlsx"

if not os.path.exists(file_path_2024):
    raise FileNotFoundError(f"File not found: {file_path_2024}")
if not os.path.exists(file_path_2025):
    raise FileNotFoundError(f"File not found: {file_path_2025}")


# Read both years data
def read_data(file_path):
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding='utf-8')
    else:
        raise ValueError("Unsupported file format")
    return df


df_2024 = read_data(file_path_2024)
df_2025 = read_data(file_path_2025)


# ===================== Data Preprocessing =====================
def preprocess_data(df, year):
    """Preprocess data for a specific year"""
    # Rename columns to English
    df = df.rename(columns={
        'date': 'Date',
        'time period': 'Time_Period',
        'Total fabric outbound quantity (unit: rolls)': 'Total_Outbound_Quantity'
    })

    # Date processing
    df['Date'] = pd.to_datetime(df['Date'])

    # Calculate daily total outbound quantity
    daily_total = df.groupby('Date')['Total_Outbound_Quantity'].sum().reset_index()
    daily_total.rename(columns={'Total_Outbound_Quantity': 'Daily_Total_Outbound'}, inplace=True)

    # Calculate monthly total outbound quantity
    daily_total.set_index('Date', inplace=True)
    monthly_total = daily_total.resample('M').sum().reset_index()
    monthly_total.rename(columns={'Daily_Total_Outbound': 'Monthly_Total_Outbound'}, inplace=True)
    monthly_total['Year'] = year
    monthly_total['Month'] = monthly_total['Date'].dt.month

    return monthly_total


# Preprocess both years
monthly_2024 = preprocess_data(df_2024, 2024)
monthly_2025 = preprocess_data(df_2025, 2025)

# Combine both years data
monthly_total = pd.concat([monthly_2024, monthly_2025], ignore_index=True)
monthly_total.set_index('Date', inplace=True)
monthly_total = monthly_total.sort_index()

# Create time series
ts = monthly_total['Monthly_Total_Outbound']

# Check data length
print(f"Available data months: {len(ts)}")
print("First 10 months data:")
print(ts.head(10))


# ===================== Improved Data Decomposition =====================
class RobustDecomposer:
    """More robust time series decomposer"""

    def __init__(self, ts):
        self.ts = ts.asfreq('M').interpolate()  # Ensure consistent frequency and fill missing values
        self.trend = self._compute_trend()
        self.seasonal = self._compute_seasonal()
        self.resid = self.ts - self.trend - self.seasonal

    def _compute_trend(self):
        """Calculate trend using moving average with adaptive window size"""
        window_size = min(6, len(self.ts) // 2)  # Dynamic window size
        if window_size % 2 == 0:  # Ensure window is odd
            window_size += 1
        return self.ts.rolling(window=window_size, center=True, min_periods=1).mean()

    def _compute_seasonal(self):
        """Calculate seasonal component"""
        detrended = self.ts - self.trend

        # Calculate seasonality by monthly average
        seasonal = detrended.groupby(detrended.index.month).mean()

        # Align with original time index
        seasonal_series = pd.Series(
            seasonal[self.ts.index.month].values,
            index=self.ts.index,
            name='seasonal'
        )

        # Adjust seasonal sum to 0
        return seasonal_series - seasonal_series.mean()

    def plot(self):
        """Plot decomposition results"""
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))

        self.ts.plot(ax=ax1, title='Original Series', color='blue')
        ax1.set_ylabel('Outbound Quantity')

        self.trend.plot(ax=ax2, title='Trend Component', color='orange')
        ax2.set_ylabel('Trend')

        self.seasonal.plot(ax=ax3, title='Seasonal Component', color='green')
        ax3.set_ylabel('Seasonal')

        self.resid.plot(ax=ax4, title='Residual Component', color='red')
        ax4.set_ylabel('Residual')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()


# Use improved decomposer
decomposition = RobustDecomposer(ts)
decomposition.plot()

# Check residuals
print("\nResidual descriptive statistics:")
print(decomposition.resid.describe())


# ===================== Improved SARIMA-Markov Model =====================
class ImprovedSARIMAMarkov:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.state_means = None
        self.state_vars = None
        self.transition_matrix = None
        self.simple_model = False
        self.trend_model = None
        self.seasonal_pattern = None

    def fit(self, ts, decomposition):
        # Extract decomposed components
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()

        print(f"\nResidual range: {residual.min():.2f} ~ {residual.max():.2f}")

        # 1. Trend component modeling
        try:
            # Use more robust ARIMA parameter selection
            self.trend_model = ARIMA(trend, order=(1, 1, 1)).fit()
            print("Trend model fitted successfully:")
            print(self.trend_model.summary())
        except Exception as e:
            print(f"Trend model fitting failed: {str(e)}")
            try:
                # Try simpler model
                self.trend_model = ARIMA(trend, order=(0, 1, 0)).fit()
                print("Simple trend model used successfully")
            except:
                print("Will use last value as trend prediction")
                self.trend_model = None

        # 2. Seasonal component (fixed pattern)
        self.seasonal_pattern = seasonal.groupby(seasonal.index.month).mean()

        # 3. Residual component modeling (Markov model)
        residuals = residual

        if len(residuals) < 6:  # Insufficient residual data
            print(f"Insufficient residual data ({len(residuals)} points), will simplify prediction")
            self.simple_model = True
            return

        # State discretization
        n_states = min(self.n_states, max(2, len(residuals) // 3))
        print(f"Using {n_states} states for discretization")

        try:
            self.discretizer = KBinsDiscretizer(
                n_bins=n_states,
                encode='ordinal',
                strategy='quantile'  # Use quantile discretization
            )

            states = self.discretizer.fit_transform(
                residuals.values.reshape(-1, 1)
            ).flatten().astype(int)

            print("State distribution:")
            print(pd.Series(states).value_counts().sort_index())
        except Exception as e:
            print(f"State discretization failed: {str(e)}")
            self.simple_model = True
            return

        # Calculate state statistics
        self.state_means = []
        self.state_vars = []
        for i in range(n_states):
            mask = (states == i)
            if sum(mask) > 0:
                self.state_means.append(residuals[mask].mean())
                self.state_vars.append(residuals[mask].var())
            else:
                # If no data for a state, use global statistics
                self.state_means.append(residuals.mean())
                self.state_vars.append(residuals.var())

        print("\nState statistics:")
        for i in range(n_states):
            print(f"State {i}: Mean={self.state_means[i]:.2f}, Variance={self.state_vars[i]:.2f}")

        # Calculate transition probability matrix (with Laplace smoothing)
        self.transition_matrix = np.ones((n_states, n_states)) * 0.1  # Stronger smoothing

        for t in range(len(states) - 1):
            i = states[t]
            j = states[t + 1]
            self.transition_matrix[i, j] += 1

        # Row normalization
        row_sums = self.transition_matrix.sum(axis=1)
        self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

        print("\nTransition probability matrix:")
        print(self.transition_matrix)

    def predict(self, steps, last_date, alpha=0.05):
        # 1. Predict trend component
        if self.trend_model:
            try:
                trend_pred = self.trend_model.get_forecast(steps=steps)
                trend_mean = trend_pred.predicted_mean
                trend_var = trend_pred.var_pred_mean
            except:
                # If prediction fails, use last trend value
                last_trend = decomposition.trend.dropna()[-1]
                trend_mean = np.array([last_trend] * steps)
                trend_var = np.var(decomposition.trend.diff().dropna()) * np.ones(steps)
        else:
            last_trend = decomposition.trend.dropna()[-1]
            trend_mean = np.array([last_trend] * steps)
            trend_var = np.var(decomposition.trend.diff().dropna()) * np.ones(steps)

        # 2. Add seasonal component
        seasonal_values = []
        for i in range(1, steps + 1):
            month = (last_date.month + i - 1) % 12 + 1
            seasonal_values.append(self.seasonal_pattern[month])
        seasonal_mean = np.array(seasonal_values)

        # Base prediction
        base_mean = trend_mean + seasonal_mean
        base_var = trend_var  # Seasonal assumed deterministic

        # 3. Residual prediction (Markov correction)
        if self.simple_model or not hasattr(self, 'discretizer'):
            # No residual model
            markov_correction = np.zeros(steps)
            markov_var = np.var(decomposition.resid.dropna()) * np.ones(steps)
        else:
            # Get current residual state
            last_resid = decomposition.resid.dropna()[-1]
            current_state = self.discretizer.transform(
                [[last_resid]]
            )[0, 0].astype(int)

            print(f"\nCurrent residual state: {current_state}")

            # Multi-step state probability
            state_probs = [self.transition_matrix[current_state]]
            for _ in range(1, steps):
                state_probs.append(
                    state_probs[-1] @ self.transition_matrix
                )
            state_probs = np.array(state_probs)

            # Markov correction term (limit amplitude)
            markov_correction = state_probs @ np.array(self.state_means)
            max_correction = 0.2 * np.abs(base_mean)  # Limit correction amplitude
            markov_correction = np.clip(
                markov_correction,
                -max_correction,
                max_correction
            )

            # Residual variance
            markov_var = state_probs @ np.array(self.state_vars)

        # Combined prediction
        combined_mean = base_mean + markov_correction
        combined_var = base_var + markov_var

        # Confidence intervals
        z = 1.96  # 95% confidence interval
        lower = combined_mean - z * np.sqrt(combined_var)
        upper = combined_mean + z * np.sqrt(combined_var)

        # Create result DataFrame
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=steps,
            freq='M'
        )

        return pd.DataFrame({
            'Prediction': combined_mean,
            'Lower_Bound': lower,
            'Upper_Bound': upper
        }, index=forecast_dates)


# ===================== Model Training and Prediction =====================
# Initialize model
model = ImprovedSARIMAMarkov(n_states=3)

# Train model
model.fit(ts, decomposition)

# Predict future 12 months (2026 only)
forecast_steps = 12
forecast = model.predict(forecast_steps, ts.index[-1])

# Filter for 2026 only
forecast_2026 = forecast[forecast.index.year == 2026]

# ===================== Model Evaluation =====================
# In-sample prediction
train_pred = []
for i in range(6, len(ts)):  # Start from 6th data point (more stable with 2 years data)
    train_subset = ts.iloc[:i]
    try:
        temp_decomp = RobustDecomposer(train_subset)
        temp_model = ImprovedSARIMAMarkov(n_states=2)
        temp_model.fit(train_subset, temp_decomp)
        pred = temp_model.predict(1, train_subset.index[-1])
        train_pred.append(pred.iloc[0]['Prediction'])
    except Exception as e:
        print(f"Prediction failed at i={i}: {str(e)}")
        train_pred.append(train_subset.iloc[-1])  # Use most recent value

# Align actual values
actual_values = ts.iloc[6:6 + len(train_pred)]
train_pred_series = pd.Series(train_pred, index=actual_values.index)

# Calculate training set residuals
train_residuals = actual_values - train_pred_series

# Calculate evaluation metrics
mae_train = mean_absolute_error(actual_values, train_pred_series)
mape_train = mean_absolute_percentage_error(actual_values, train_pred_series) * 100

print("\nModel Evaluation Results:")
print(f"Training Set MAE: {mae_train:.1f}")
print(f"Training Set MAPE: {mape_train:.1f}%")

# ===================== Main Prediction Visualization =====================
plt.figure(figsize=(14, 8))

# 1. Historical data - 2024
ts_2024 = ts[ts.index.year == 2024]
plt.plot(ts_2024.index, ts_2024, 'bo-', label='Actual (2024)', markersize=6, linewidth=2)

# 2. Historical data - 2025
ts_2025 = ts[ts.index.year == 2025]
plt.plot(ts_2025.index, ts_2025, 'go-', label='Actual (2025)', markersize=6, linewidth=2)

# 3. Training set fitted values
if len(train_pred_series) > 0:
    plt.plot(train_pred_series.index, train_pred_series, 'y--',
             label='Fitted Values', linewidth=2)

# 4. Prediction - 2026
plt.plot(forecast_2026.index, forecast_2026['Prediction'], 'r-o',
         label='Prediction (2026)', markersize=8, linewidth=2)

# 5. Confidence interval - 2026
plt.fill_between(forecast_2026.index,
                 forecast_2026['Lower_Bound'],
                 forecast_2026['Upper_Bound'],
                 color='pink', alpha=0.3, label='95% Confidence Interval (2026)')

# Set date format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()

# Add title and labels
plt.title('Monthly Fabric Total Outbound Quantity Forecast (2026)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Monthly Total Outbound Quantity (rolls)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add model parameter description
param_text = f"Data Months: {len(ts)} | States: {model.n_states}\n"
param_text += f"Training MAE: {mae_train:.1f} | MAPE: {mape_train:.1f}%"
plt.annotate(param_text,
             xy=(0.02, 0.95),
             xycoords='axes fraction',
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('monthly_fabric_forecast_2026.png', dpi=300)
plt.show()

# ===================== Residual Analysis =====================
if len(train_residuals) >= 5:  # Ensure enough data for residual analysis
    plt.figure(figsize=(14, 10))

    # Residual histogram
    plt.subplot(2, 2, 1)
    plt.hist(train_residuals, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Training Set Residual Distribution')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Residual Q-Q plot
    plt.subplot(2, 2, 2)
    stats.probplot(train_residuals, dist="norm", plot=plt)
    plt.title('Training Set Residual Q-Q Plot')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Residual autocorrelation plot
    plt.subplot(2, 2, 3)
    acf_lags = min(20, len(train_residuals) - 1)
    plot_acf(train_residuals, lags=acf_lags, ax=plt.gca(),
             title='Training Set Residual Autocorrelation', color='blue')
    plt.title('Training Set Residual Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation Coefficient')

    # Residual partial autocorrelation plot
    plt.subplot(2, 2, 4)
    # Ensure lag period doesn't exceed limit
    pacf_lags = min(10, len(train_residuals) // 2 - 1 if len(train_residuals) > 8 else len(train_residuals) - 2)
    if pacf_lags < 1:
        pacf_lags = 1

    plot_pacf(train_residuals, lags=pacf_lags, ax=plt.gca(),
              title='Training Set Residual Partial Autocorrelation', color='green')
    plt.title('Training Set Residual Partial Autocorrelation')
    plt.xlabel('Lag')
    plt.ylabel('Partial Autocorrelation Coefficient')

    plt.tight_layout()
    plt.savefig('residual_analysis_2026.png', dpi=300)
    plt.show()
else:
    print(f"Insufficient residual data ({len(train_residuals)} points), skipping residual analysis plots")

# ===================== Markov State Transition Visualization =====================
if hasattr(model, 'transition_matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(model.transition_matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title('Markov State Transition Probability Matrix')
    plt.xlabel('Target State')
    plt.ylabel('Source State')

    # Add probability value labels
    for i in range(model.transition_matrix.shape[0]):
        for j in range(model.transition_matrix.shape[1]):
            plt.text(j, i, f"{model.transition_matrix[i, j]:.2f}",
                     ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig('markov_transition_matrix_2026.png', dpi=300)
    plt.show()


# ===================== Prediction Results Output =====================
# Calculate year-over-year growth (compared to same month last year)
def calculate_yoy_growth(forecast_df, historical_ts):
    """Calculate year-over-year growth compared to same month in previous year"""
    result = forecast_df.copy()
    result['YoY_Growth'] = 0.0

    for idx, row in result.iterrows():
        # Find same month from previous year
        same_month_prev_year = idx - pd.DateOffset(years=1)
        if same_month_prev_year in historical_ts.index:
            prev_year_value = historical_ts.loc[same_month_prev_year]
            result.loc[idx, 'YoY_Growth'] = (row['Prediction'] / prev_year_value - 1) * 100

    return result


forecast_2026 = calculate_yoy_growth(forecast_2026, ts)


# Create display DataFrame
def prepare_forecast_df(forecast_df, year):
    result = forecast_df.copy()
    result.index = result.index.strftime('%Y-%m')

    # Format values
    result['Prediction'] = result['Prediction'].apply(lambda x: f"{int(x):,}")
    result['Lower_Bound'] = result['Lower_Bound'].apply(lambda x: f"{int(x):,}")
    result['Upper_Bound'] = result['Upper_Bound'].apply(lambda x: f"{int(x):,}")
    result['YoY_Growth'] = result['YoY_Growth'].apply(lambda x: f"{x:.1f}%")

    result.columns = [f'{col}_{year}' for col in result.columns]
    return result


forecast_2026_display = prepare_forecast_df(forecast_2026, '2026')

# Save prediction results
forecast_2026_display.to_csv('monthly_fabric_outbound_forecast_results_2026.csv', index_label='Month')

print("=" * 50)
print("Monthly Fabric Outbound Quantity Forecast Results (2026):")
print(forecast_2026_display)
print("=" * 50)
print(f"Forecast results saved to: monthly_fabric_outbound_forecast_results_2026.csv")

# ===================== Seasonal Analysis =====================
# Extract seasonal pattern
seasonal_pattern = decomposition.seasonal.groupby(decomposition.seasonal.index.month).mean()

plt.figure(figsize=(10, 5))
plt.plot(seasonal_pattern.index, seasonal_pattern.values, 'g-o', linewidth=2, markersize=8)
plt.title('Monthly Seasonal Pattern', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Seasonal Impact (rolls)')
plt.xticks(range(1, 13))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('seasonal_pattern_analysis.png', dpi=300)
plt.show()

# ===================== Trend Analysis =====================
plt.figure(figsize=(10, 5))
plt.plot(decomposition.trend.index, decomposition.trend.values, 'b-o', linewidth=2, markersize=6, label='Actual Trend')
if model.trend_model:
    trend_pred = model.trend_model.predict(start=decomposition.trend.index[0], end=decomposition.trend.index[-1])
    plt.plot(trend_pred.index, trend_pred.values, 'r--', linewidth=2, label='Fitted Trend')
plt.title('Trend Component Analysis', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Trend Value (rolls)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('trend_analysis.png', dpi=300)
plt.show()

# ===================== Yearly Comparison =====================
plt.figure(figsize=(12, 6))

# Group by month for both years
monthly_comparison = monthly_total.groupby([monthly_total.index.year, monthly_total.index.month])[
    'Monthly_Total_Outbound'].mean().unstack(0)

plt.plot(monthly_comparison.index, monthly_comparison[2024], 'bo-', label='2024', linewidth=2, markersize=6)
plt.plot(monthly_comparison.index, monthly_comparison[2025], 'go-', label='2025', linewidth=2, markersize=6)

plt.title('Monthly Outbound Quantity Comparison (2024 vs 2025)', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Monthly Total Outbound Quantity (rolls)')
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('yearly_comparison_2024_2025.png', dpi=300)
plt.show()