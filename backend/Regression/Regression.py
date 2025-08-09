import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from typing import Dict, Any, Optional, Tuple, List
import warnings
import logging
from datetime import datetime
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PharmacySalesPredictor:
    """
    Enhanced pharmacy sales prediction using LSTM with improved visualization and process.
    Features dark mode graphs, confidence intervals, model evaluation metrics, and trend analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the predictor with configuration options.
        
        Args:
            config: Configuration dictionary with enhanced options:
                - data_file: str, path to the parquet file (default: "./supertable.parquet")
                - target_col: str, target column name (default: "sales_sheet")
                - sequence_length: int, LSTM sequence length (default: 10)
                - forecast_periods: int, number of periods to forecast (default: 25)
                - smoothing_span: int, exponential smoothing span (default: 4)
                - lstm_units: int, number of LSTM units (default: 64)
                - lstm_epochs: int, number of training epochs (default: 100)
                - test_size: float, train-test split ratio (default: 0.2)
                - random_state: int, random state for reproducibility (default: 42)
                - remove_last_n: int, number of last periods to remove (default: 2)
                - dropout_rate: float, dropout rate for regularization (default: 0.2)
                - early_stopping: bool, use early stopping (default: True)
                - confidence_level: float, confidence level for intervals (default: 0.95)
                - theme: str, plot theme - 'dark' or 'light' (default: 'dark')
        """
        default_config = {
            'data_file': "./supertable.parquet",
            'target_col': "sales_sheet",
            'sequence_length': 10,
            'forecast_periods': 25,
            'smoothing_span': 4,
            'lstm_units': 64,
            'lstm_epochs': 100,
            'test_size': 0.2,
            'random_state': 42,
            'remove_last_n': 2,
            'dropout_rate': 0.2,
            'early_stopping': True,
            'confidence_level': 0.95,
            'theme': 'dark'
        }
        
        self.config = {**default_config, **(config or {})}
        self.df = None
        self.weekly_df = None
        self.scaler = None
        self.lstm_model = None
        self.ts_series_smoothed = None
        self.ts_series_raw = None
        self.model_history = None
        self.metrics = {}
        
        # Theme configuration
        self.theme_config = self._get_theme_config()
        
    def _get_theme_config(self) -> Dict[str, Any]:
        """Get theme configuration for plots."""
        if self.config['theme'] == 'dark':
            return {
                'template': 'plotly_dark',
                'bg_color': '#1e1e1e',
                'grid_color': '#3a3a3a',
                'text_color': '#ffffff',
                'colors': {
                    'actual': '#00d4ff',      # Cyan
                    'predicted': '#ff6b6b',   # Coral red
                    'forecast': '#4ecdc4',    # Teal
                    'confidence': '#ffbe0b',  # Amber
                    'trend': '#fb5607'        # Orange red
                }
            }
        else:
            return {
                'template': 'plotly_white',
                'bg_color': '#ffffff',
                'grid_color': '#e0e0e0',
                'text_color': '#000000',
                'colors': {
                    'actual': '#1f77b4',      # Blue
                    'predicted': '#ff7f0e',   # Orange
                    'forecast': '#2ca02c',    # Green
                    'confidence': '#d62728',  # Red
                    'trend': '#9467bd'        # Purple
                }
            }

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess the data with enhanced error handling."""
        try:
            logger.info(f"Loading data from {self.config['data_file']}")
            df = pd.read_parquet(self.config['data_file'])
            
            # Data preprocessing
            df.rename(columns={"addeddate": "date_id", "time_": "time_id"}, inplace=True)
            df.drop(columns=["pack_size"], inplace=True, errors="ignore")
            df["date_id"] = pd.to_datetime(df["date_id"], errors="coerce")

            df["time_id"] = (
                df["time_id"].astype(str).str.replace(r"(?i)(am|pm)$", r" \1", regex=True)
            )
            df["time_id"] = pd.to_datetime(df["time_id"], format="%I:%M %p", errors="coerce")

            # Drop only rows missing critical datetime
            initial_rows = len(df)
            df = df.dropna(subset=["sheet", "time_id", self.config['target_col']])
            logger.info(f"Removed {initial_rows - len(df)} rows with missing critical data")
            
            # Add time components
            df["year"] = df["date_id"].dt.isocalendar().year
            df["week"] = df["date_id"].dt.isocalendar().week
            df["day"] = df["date_id"].dt.isocalendar().day

            # Group by year + week with more robust aggregation
            weekly_df = df.groupby(["year", "week"], as_index=False).agg(
                {
                    "sales_pack": "sum",
                    "sheet": "sum",
                    "sales_sheet": "sum",
                    "price_inr": "mean",  # Use mean for price
                }
            )

            # Create week start datetime
            weekly_df["week_start"] = pd.to_datetime(
                weekly_df["year"].astype(str) + "-W" + weekly_df["week"].astype(str) + "-1",
                format="%G-W%V-%u",
            )

            # Sort by date
            weekly_df = weekly_df.sort_values('week_start').reset_index(drop=True)
            
            self.df = df
            self.weekly_df = weekly_df
            logger.info(f"Data loaded successfully: {len(weekly_df)} weekly records")
            return weekly_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def prepare_lstm_data(self) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Prepare data for LSTM training with enhanced preprocessing."""
        if self.weekly_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Create time series
        self.ts_series_raw = self.weekly_df.set_index("week_start")[self.config['target_col']].sort_index()
        self.ts_series_raw = self.ts_series_raw[:-self.config['remove_last_n']]

        # Apply smoothing
        self.ts_series_smoothed = self.ts_series_raw.ewm(
            span=self.config['smoothing_span'], 
            adjust=False
        ).mean()

        # Scale smoothed data
        self.scaler = StandardScaler()
        ts_scaled = self.scaler.fit_transform(self.ts_series_smoothed.values.reshape(-1, 1))

        logger.info(f"Time series prepared: {len(ts_scaled)} data points")
        return ts_scaled, self.ts_series_smoothed.values, self.ts_series_smoothed.index

    def train_lstm(self, ts_scaled: np.ndarray) -> Sequential:
        """Train the LSTM model with enhanced architecture and callbacks."""
        seq_len = self.config['sequence_length']
        generator = TimeseriesGenerator(ts_scaled, ts_scaled, length=seq_len, batch_size=1)

        # Enhanced LSTM architecture
        lstm = Sequential([
            LSTM(self.config['lstm_units'], 
                 activation="relu", 
                 input_shape=(seq_len, 1),
                 return_sequences=True),
            Dropout(self.config['dropout_rate']),
            LSTM(self.config['lstm_units'] // 2, 
                 activation="relu"),
            Dropout(self.config['dropout_rate']),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        
        lstm.compile(optimizer="adam", loss="mse", metrics=['mae'])
        
        # Callbacks
        callbacks = []
        if self.config['early_stopping']:
            early_stop = EarlyStopping(
                monitor='loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        logger.info("Training LSTM model...")
        self.model_history = lstm.fit(
            generator, 
            epochs=self.config['lstm_epochs'], 
            verbose=1,
            callbacks=callbacks
        )
        
        self.lstm_model = lstm
        logger.info("LSTM training completed")
        return lstm

    def generate_predictions(self, ts_scaled: np.ndarray, ts_index: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Generate historical predictions with evaluation metrics."""
        seq_len = self.config['sequence_length']
        generator = TimeseriesGenerator(ts_scaled, ts_scaled, length=seq_len, batch_size=1)
        
        preds = self.lstm_model.predict(generator, verbose=0)
        true_vals = ts_scaled[seq_len:]
        valid_idx = ts_index[seq_len:]
        
        # Calculate metrics
        true_unscaled = self.scaler.inverse_transform(true_vals)
        pred_unscaled = self.scaler.inverse_transform(preds)
        
        self.metrics = {
            'mae': mean_absolute_error(true_unscaled, pred_unscaled),
            'rmse': np.sqrt(mean_squared_error(true_unscaled, pred_unscaled)),
            'r2': r2_score(true_unscaled, pred_unscaled),
            'mape': np.mean(np.abs((true_unscaled - pred_unscaled) / true_unscaled)) * 100
        }
        
        logger.info(f"Model Metrics - MAE: {self.metrics['mae']:.2f}, RMSE: {self.metrics['rmse']:.2f}, R²: {self.metrics['r2']:.3f}")
        
        return preds, true_vals, valid_idx

    def generate_forecast_with_confidence(self, ts_scaled: np.ndarray) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Generate future forecasts with confidence intervals."""
        seq_len = self.config['sequence_length']
        
        # Monte Carlo dropout for uncertainty estimation
        n_samples = 100
        forecasts = []
        
        for _ in range(n_samples):
            future_preds = []
            last_seq = ts_scaled[-seq_len:].copy()
            
            for _ in range(self.config['forecast_periods']):
                # Predict with dropout enabled for uncertainty
                pred = self.lstm_model(last_seq.reshape((1, seq_len, 1)), training=True)
                pred_val = pred.numpy()[0, 0]
                future_preds.append(pred_val)
                last_seq = np.append(last_seq[1:], pred_val)
                
            forecasts.append(future_preds)
        
        forecasts = np.array(forecasts)
        
        # Calculate confidence intervals
        mean_forecast = np.mean(forecasts, axis=0)
        std_forecast = np.std(forecasts, axis=0)
        
        confidence = self.config['confidence_level']
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        lower_bound = mean_forecast - z_score * std_forecast
        upper_bound = mean_forecast + z_score * std_forecast
        
        # Forecast dates
        future_index = pd.date_range(
            start=self.ts_series_smoothed.index[-1] + pd.Timedelta(weeks=1),
            periods=self.config['forecast_periods'],
            freq="W-MON",
        )

        # Convert to original scale
        mean_series = pd.Series(
            self.scaler.inverse_transform(mean_forecast.reshape(-1, 1)).flatten(),
            index=future_index
        )
        
        lower_series = pd.Series(
            self.scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten(),
            index=future_index
        )
        
        upper_series = pd.Series(
            self.scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten(),
            index=future_index
        )
        
        return mean_series, lower_series, upper_series

    def calculate_trend_analysis(self) -> Dict[str, float]:
        """Calculate trend analysis metrics."""
        if self.ts_series_smoothed is None:
            return {}
            
        # Calculate rolling statistics
        recent_data = self.ts_series_smoothed.tail(12)  # Last 12 weeks
        older_data = self.ts_series_smoothed.iloc[-24:-12] if len(self.ts_series_smoothed) >= 24 else self.ts_series_smoothed.head(12)
        
        trend_analysis = {
            'recent_avg': recent_data.mean(),
            'older_avg': older_data.mean(),
            'trend_change': ((recent_data.mean() - older_data.mean()) / older_data.mean() * 100),
            'volatility': recent_data.std(),
            'max_recent': recent_data.max(),
            'min_recent': recent_data.min()
        }
        
        return trend_analysis

    def create_enhanced_plotly_charts(self, preds: np.ndarray, true_vals: np.ndarray, 
                                    valid_idx: pd.DatetimeIndex, forecast_mean: pd.Series,
                                    forecast_lower: pd.Series, forecast_upper: pd.Series) -> str:
        """Create enhanced Plotly charts with dark theme and improved features."""
        
        # Create subplots with custom spacing
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "📈 Historical Prediction vs Actual", 
                "🔮 Forecast with Confidence Intervals",
                "📊 Model Training History",
                "📋 Performance Metrics",
                "🎯 Trend Analysis",
                "📉 Raw vs Smoothed Data"
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "indicator"}],
                   [{"secondary_y": False}, {"secondary_y": False}]],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        colors = self.theme_config['colors']
        
        # 1. Historical predictions
        actual_smoothed = self.scaler.inverse_transform(true_vals).flatten()
        predicted = self.scaler.inverse_transform(preds).flatten()
        
        fig.add_trace(
            go.Scatter(
                x=valid_idx, y=actual_smoothed,
                mode='lines', name='Actual',
                line=dict(color=colors['actual'], width=3),
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=valid_idx, y=predicted,
                mode='lines', name='Predicted',
                line=dict(color=colors['predicted'], width=2, dash='dash'),
                hovertemplate='<b>Predicted</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 2. Forecast with confidence intervals
        fig.add_trace(
            go.Scatter(
                x=self.ts_series_smoothed.index, y=self.ts_series_smoothed.values,
                mode='lines', name='Historical',
                line=dict(color=colors['actual'], width=2),
                showlegend=False,
                hovertemplate='<b>Historical</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Confidence interval
        fig.add_trace(
            go.Scatter(
                x=list(forecast_mean.index) + list(forecast_mean.index[::-1]),
                y=list(forecast_upper.values) + list(forecast_lower.values[::-1]),
                fill='toself',
                fillcolor=f'rgba{(*[int(colors["confidence"][i:i+2], 16) for i in (1, 3, 5)], 0.2)}',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{int(self.config["confidence_level"]*100)}% Confidence',
                showlegend=True,
                hoverinfo='skip'
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=forecast_mean.index, y=forecast_mean.values,
                mode='lines', name='Forecast',
                line=dict(color=colors['forecast'], width=3),
                hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # 3. Training history
        if self.model_history:
            epochs = range(1, len(self.model_history.history['loss']) + 1)
            fig.add_trace(
                go.Scatter(
                    x=list(epochs), y=self.model_history.history['loss'],
                    mode='lines', name='Training Loss',
                    line=dict(color=colors['trend'], width=2),
                    hovertemplate='<b>Training Loss</b><br>Epoch: %{x}<br>Loss: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # 4. Performance metrics indicator
        metrics_text = f"""
        MAE: {self.metrics.get('mae', 0):.2f}<br>
        RMSE: {self.metrics.get('rmse', 0):.2f}<br>
        R²: {self.metrics.get('r2', 0):.3f}<br>
        MAPE: {self.metrics.get('mape', 0):.1f}%
        """
        
        fig.add_trace(
            go.Indicator(
                mode="number+gauge+delta",
                value=self.metrics.get('r2', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={"text": f"<span style='font-size:14px'>Model R² Score</span><br><span style='font-size:10px'>{metrics_text}</span>"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': colors['forecast']},
                       'steps': [
                           {'range': [0, 0.5], 'color': "red"},
                           {'range': [0.5, 0.8], 'color': "yellow"},
                           {'range': [0.8, 1], 'color': "green"}],
                       'threshold': {'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75, 'value': 0.9}}
            ),
            row=2, col=2
        )
        
        # 5. Trend analysis
        trend_data = self.calculate_trend_analysis()
        if trend_data:
            categories = ['Recent Avg', 'Older Avg', 'Max Recent', 'Min Recent']
            values = [trend_data['recent_avg'], trend_data['older_avg'], 
                     trend_data['max_recent'], trend_data['min_recent']]
            
            fig.add_trace(
                go.Bar(
                    x=categories, y=values,
                    marker_color=[colors['forecast'], colors['actual'], colors['confidence'], colors['predicted']],
                    name='Trend Analysis',
                    hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 6. Raw vs Smoothed comparison
        if self.ts_series_raw is not None:
            # Sample data if too many points
            sample_idx = np.linspace(0, len(self.ts_series_raw)-1, min(200, len(self.ts_series_raw)), dtype=int)
            
            fig.add_trace(
                go.Scatter(
                    x=self.ts_series_raw.index[sample_idx], 
                    y=self.ts_series_raw.values[sample_idx],
                    mode='lines', name='Raw Data',
                    line=dict(color=colors['predicted'], width=1, dash='dot'),
                    opacity=0.7,
                    hovertemplate='<b>Raw</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ),
                row=3, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.ts_series_smoothed.index[sample_idx], 
                    y=self.ts_series_smoothed.values[sample_idx],
                    mode='lines', name='Smoothed',
                    line=dict(color=colors['actual'], width=2),
                    hovertemplate='<b>Smoothed</b><br>Date: %{x}<br>Value: %{y:.2f}<extra></extra>'
                ),
                row=3, col=2
            )
        
        # Update layout with enhanced styling
        fig.update_layout(
            title={
                'text': "🏥 Enhanced Pharmacy Sales Prediction Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.theme_config['text_color']}
            },
            height=1200,
            template=self.theme_config['template'],
            showlegend=True,
            plot_bgcolor=self.theme_config['bg_color'],
            paper_bgcolor=self.theme_config['bg_color'],
            font={'color': self.theme_config['text_color']},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor=self.theme_config['grid_color'])
        fig.update_yaxes(showgrid=True, gridcolor=self.theme_config['grid_color'])
        
        # Add annotations
        trend_change = trend_data.get('trend_change', 0) if trend_data else 0
        trend_emoji = "📈" if trend_change > 0 else "📉" if trend_change < 0 else "➡️"
        
        fig.add_annotation(
            text=f"{trend_emoji} Trend: {trend_change:+.1f}%",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=14, color=colors['trend']),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=colors['trend'],
            borderwidth=1
        )
        
        # Convert to HTML with custom configuration
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'pharmacy_sales_prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'height': 1200,
                'width': 1800,
                'scale': 2
            }
        }
        
        html_string = pyo.plot(
            fig, 
            output_type='div', 
            include_plotlyjs=True,
            config=config
        )
        
        return html_string

    def run_prediction(self) -> str:
        """Run the complete enhanced prediction pipeline."""
        try:
            # Load and prepare data
            self.load_data()
            ts_scaled, _, ts_index = self.prepare_lstm_data()
            
            # Train model
            self.train_lstm(ts_scaled)
            
            # Generate predictions and forecasts
            preds, true_vals, valid_idx = self.generate_predictions(ts_scaled, ts_index)
            forecast_mean, forecast_lower, forecast_upper = self.generate_forecast_with_confidence(ts_scaled)
            
            # Create visualization
            html_output = self.create_enhanced_plotly_charts(
                preds, true_vals, valid_idx, 
                forecast_mean, forecast_lower, forecast_upper
            )
            
            logger.info("Prediction pipeline completed successfully")
            return html_output
            
        except Exception as e:
            logger.error(f"Error in prediction pipeline: {str(e)}")
            raise

    def get_comprehensive_results(self) -> Dict[str, Any]:
        """Return comprehensive results including data, metrics, and forecasts."""
        if self.ts_series_smoothed is None:
            raise ValueError("Model not trained. Call run_prediction() first.")
            
        ts_scaled, _, _ = self.prepare_lstm_data()
        forecast_mean, forecast_lower, forecast_upper = self.generate_forecast_with_confidence(ts_scaled)
        trend_analysis = self.calculate_trend_analysis()
        
        return {
            'historical_raw': self.ts_series_raw,
            'historical_smoothed': self.ts_series_smoothed,
            'forecast_mean': forecast_mean,
            'forecast_lower': forecast_lower,
            'forecast_upper': forecast_upper,
            'metrics': self.metrics,
            'trend_analysis': trend_analysis,
            'config': self.config
        }


def main():
    """Enhanced main function with comprehensive configuration and reporting."""
    
    # Enhanced configuration
    enhanced_config = {
        'data_file': "./assets/supertable.parquet",
        'target_col': "sales_sheet",
        'sequence_length': 12,      # Increased for better pattern recognition
        'forecast_periods': 30,     # Extended forecast
        'smoothing_span': 5,        # Better smoothing
        'lstm_units': 128,          # More complex model
        'lstm_epochs': 150,         # More training
        'dropout_rate': 0.3,        # Better regularization
        'early_stopping': True,
        'confidence_level': 0.95,
        'theme': 'dark',            # Dark mode
        'remove_last_n': 1          # Keep more data
    }
    
    print("🚀 Starting Enhanced Pharmacy Sales Prediction...")
    print("=" * 60)
    print(f"📋 Configuration:")
    for key, value in enhanced_config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = PharmacySalesPredictor(config=enhanced_config)
        
        # Run prediction
        html_output = predictor.run_prediction()
        
        # Save HTML output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"enhanced_pharmacy_sales_prediction_{timestamp}.html"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_output)
        
        # Get comprehensive results
        results = predictor.get_comprehensive_results()
        
        # Print summary
        print("\n✅ Prediction completed successfully!")
        print("=" * 60)
        print(f"📊 Visualization saved to: {output_file}")
        print(f"📈 Historical data points: {len(results['historical_smoothed'])}")
        print(f"🔮 Forecast data points: {len(results['forecast_mean'])}")
        print(f"📅 Forecast period: {results['forecast_mean'].index[0].strftime('%Y-%m-%d')} to {results['forecast_mean'].index[-1].strftime('%Y-%m-%d')}")
        
        print(f"\n🎯 Model Performance:")
        metrics = results['metrics']
        print(f"  • MAE: {metrics['mae']:.2f}")
        print(f"  • RMSE: {metrics['rmse']:.2f}")
        print(f"  • R² Score: {metrics['r2']:.3f}")
        print(f"  • MAPE: {metrics['mape']:.1f}%")
        
        print(f"\n📊 Trend Analysis:")
        trend = results['trend_analysis']
        print(f"  • Recent Average: {trend['recent_avg']:.2f}")
        print(f"  • Trend Change: {trend['trend_change']:+.1f}%")
        print(f"  • Volatility: {trend['volatility']:.2f}")
        
        print(f"\n🔮 Sample Forecast (Next 5 weeks):")
        for i, (date, mean_val, lower_val, upper_val) in enumerate(zip(
            results['forecast_mean'].head().index,
            results['forecast_mean'].head().values,
            results['forecast_lower'].head().values,
            results['forecast_upper'].head().values
        )):
            print(f"  Week {i+1} ({date.strftime('%Y-%m-%d')}): {mean_val:.2f} [{lower_val:.2f}, {upper_val:.2f}]")
        
        print("\n" + "=" * 60)
        print("🎉 Analysis complete! Open the HTML file in your browser to view the interactive dashboard.")
        
    except FileNotFoundError:
        print("❌ Error: Data file './supertable.parquet' not found.")
        print("Please ensure the data file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("Please check your data file and configuration.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
