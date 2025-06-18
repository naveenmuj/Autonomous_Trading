import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self, min_points: int = 3, min_pct_height: float = 0.02):
        """
        Initialize Technical Analyzer
        
        Args:
            min_points: Minimum points required to form a trend line
            min_pct_height: Minimum percentage height between points for trend line
        """
        self.min_points = min_points
        self.min_pct_height = min_pct_height
        logger.debug(f"TechnicalAnalyzer initialized with min_points={min_points}, min_pct_height={min_pct_height}")

    def find_swing_points(self, data: pd.DataFrame, window: int = 10) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows in price data"""
        try:
            # Get high and low prices
            highs = data['high'].values
            lows = data['low'].values
            
            # Find local maxima and minima
            high_idx = list(argrelextrema(highs, np.greater, order=window)[0])
            low_idx = list(argrelextrema(lows, np.less, order=window)[0])
            
            logger.debug(f"Found {len(high_idx)} swing highs and {len(low_idx)} swing lows")
            return high_idx, low_idx
            
        except Exception as e:
            logger.error(f"Error finding swing points: {str(e)}")
            return [], []

    def detect_trend_lines(self, data: pd.DataFrame, lookback_period: int = 100) -> Dict[str, List[Dict]]:
        """
        Detect support and resistance trend lines
        
        Returns:
            Dict containing support and resistance trend lines with their properties
        """
        try:
            df = data.copy().tail(lookback_period)
            high_idx, low_idx = self.find_swing_points(df)
            
            support_lines = self._find_trend_lines(df, low_idx, 'support')
            resistance_lines = self._find_trend_lines(df, high_idx, 'resistance')
            
            # Filter and rank trend lines
            support_lines = self._filter_trend_lines(support_lines)
            resistance_lines = self._filter_trend_lines(resistance_lines)
            
            logger.info(f"Detected {len(support_lines)} support and {len(resistance_lines)} resistance trend lines")
            
            return {
                'support': support_lines,
                'resistance': resistance_lines
            }
            
        except Exception as e:
            logger.error(f"Error detecting trend lines: {str(e)}")
            return {'support': [], 'resistance': []}

    def _find_trend_lines(self, data: pd.DataFrame, points_idx: List[int], 
                         line_type: str) -> List[Dict]:
        """Find trend lines connecting swing points"""
        try:
            trend_lines = []
            price_col = 'low' if line_type == 'support' else 'high'
            
            for i in range(len(points_idx) - self.min_points + 1):
                # Get subset of points
                idx_subset = points_idx[i:i + self.min_points]
                points_x = np.array(idx_subset).reshape(-1, 1)
                points_y = data[price_col].iloc[idx_subset].values
                
                # Check minimum height requirement
                height_pct = (max(points_y) - min(points_y)) / min(points_y)
                if height_pct < self.min_pct_height:
                    continue
                
                # Fit line through points
                reg = LinearRegression()
                reg.fit(points_x, points_y)
                
                # Calculate line properties
                slope = reg.coef_[0]
                intercept = reg.intercept_
                score = reg.score(points_x, points_y)
                
                # Calculate current value and angle
                current_val = slope * len(data) + intercept
                angle = np.arctan(slope) * 180 / np.pi
                
                trend_lines.append({
                    'type': line_type,
                    'slope': slope,
                    'intercept': intercept,
                    'score': score,
                    'points': list(zip(idx_subset, points_y)),
                    'current_value': current_val,
                    'angle': angle,
                    'strength': self._calculate_line_strength(data, slope, intercept, line_type)
                })
            
            return trend_lines
            
        except Exception as e:
            logger.error(f"Error finding {line_type} trend lines: {str(e)}")
            return []

    def _filter_trend_lines(self, lines: List[Dict], 
                          min_score: float = 0.8,
                          max_lines: int = 3) -> List[Dict]:
        """Filter trend lines based on quality and limit quantity"""
        try:
            # Filter by R² score
            quality_lines = [line for line in lines if line['score'] >= min_score]
            
            # Sort by strength and score
            sorted_lines = sorted(
                quality_lines,
                key=lambda x: (x['strength'], x['score']),
                reverse=True
            )
            
            return sorted_lines[:max_lines]
            
        except Exception as e:
            logger.error(f"Error filtering trend lines: {str(e)}")
            return []

    def _calculate_line_strength(self, data: pd.DataFrame, slope: float,
                               intercept: float, line_type: str) -> float:
        """Calculate trend line strength based on touches and bounces"""
        try:
            touches = 0
            bounces = 0
            price_col = 'low' if line_type == 'support' else 'high'
            
            # Calculate line values for each point
            x_vals = np.arange(len(data))
            line_vals = slope * x_vals + intercept
            
            # Count touches and bounces
            for i in range(len(data)):
                price = data[price_col].iloc[i]
                line_val = line_vals[i]
                
                # Check if price is near the line
                if abs(price - line_val) / price < 0.01:  # 1% tolerance
                    touches += 1
                    
                    # Check if price bounced from the line
                    if i > 0 and i < len(data) - 1:
                        prev_price = data[price_col].iloc[i-1]
                        next_price = data[price_col].iloc[i+1]
                        
                        if line_type == 'support' and prev_price > line_val and next_price > line_val:
                            bounces += 1
                        elif line_type == 'resistance' and prev_price < line_val and next_price < line_val:
                            bounces += 1
            
            # Calculate strength score (0 to 1)
            strength = (touches + bounces * 2) / len(data)
            return min(strength, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating line strength: {str(e)}")
            return 0.0

    def analyze_breakouts(self, data: pd.DataFrame, trend_lines: Dict[str, List[Dict]],
                         lookback: int = 20) -> Dict[str, List[Dict]]:
        """Analyze potential breakouts from trend lines"""
        try:
            breakouts = {'support': [], 'resistance': []}
            recent_data = data.tail(lookback)
            
            for line_type in ['support', 'resistance']:
                for line in trend_lines[line_type]:
                    # Calculate line values for recent period
                    x_vals = np.arange(len(data) - lookback, len(data))
                    line_vals = line['slope'] * x_vals + line['intercept']
                    
                    # Check for breakouts
                    if line_type == 'support':
                        # Price below support line
                        if recent_data['close'].iloc[-1] < line_vals[-1]:
                            strength = (line_vals[-1] - recent_data['close'].iloc[-1]) / line_vals[-1]
                            breakouts['support'].append({
                                'line': line,
                                'strength': strength,
                                'price': recent_data['close'].iloc[-1]
                            })
                    else:
                        # Price above resistance line
                        if recent_data['close'].iloc[-1] > line_vals[-1]:
                            strength = (recent_data['close'].iloc[-1] - line_vals[-1]) / line_vals[-1]
                            breakouts['resistance'].append({
                                'line': line,
                                'strength': strength,
                                'price': recent_data['close'].iloc[-1]
                            })
            
            logger.info(f"Found {len(breakouts['support'])} support and {len(breakouts['resistance'])} resistance breakouts")
            return breakouts
            
        except Exception as e:
            logger.error(f"Error analyzing breakouts: {str(e)}")
            return {'support': [], 'resistance': []}
