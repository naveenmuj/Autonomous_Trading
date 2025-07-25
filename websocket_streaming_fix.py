"""
Fix for WebSocket streaming without UI refresh
This demonstrates how to get WebSocket data without causing UI to reload
"""

import streamlit as st
import time
from datetime import datetime

# Fix for dashboard.py - replace the problematic _fetch_and_display_live_prices method
def fixed_fetch_and_display_live_prices(self, symbols=None):
    """Display live prices without UI reload - fixed version"""
    try:
        if not self.data_collector:
            st.error("Data collector not available!")
            return
        
        if symbols is None:
            symbols = self._get_symbols()
        
        # Check market status (one time)
        current_time = datetime.now()
        market_open_time = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close_time = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
        is_market_open = market_open_time <= current_time <= market_close_time
        
        # Display market status
        if is_market_open:
            st.success("🟢 Market is OPEN - Live WebSocket streaming")
        else:
            st.info("🔴 Market is CLOSED - Showing Last Traded Prices (LTP)")
        
        # Start WebSocket streaming ONCE
        if not hasattr(self, '_streaming_started') or not self._streaming_started:
            self.websocket_handler.start(symbols=symbols)
            self._streaming_started = True
            st.success("🔴 WebSocket streaming activated!")
        
        # Get WebSocket data (this happens instantly, no delay)
        websocket_data = self.websocket_handler.get_live_data_from_websocket()
        
        # Create columns for symbols
        cols = st.columns(len(symbols))
        
        for i, symbol in enumerate(symbols):
            with cols[i]:
                # Check if we have live WebSocket data
                live_data = websocket_data.get(symbol)
                is_streaming = bool(live_data and live_data.get('source') == 'websocket_live')
                
                if is_streaming:
                    # Use WebSocket data
                    price_data = live_data
                    st.subheader(f"🔴 {symbol.replace('.NS', '')} LIVE")
                else:
                    # Fallback to API
                    try:
                        price_data = self.data_collector.get_live_quote(symbol)
                        st.subheader(symbol.replace('.NS', ''))
                    except Exception as e:
                        st.subheader(symbol.replace('.NS', ''))
                        st.error(f"Error: {str(e)}")
                        continue
                
                if price_data and 'ltp' in price_data and price_data['ltp'] > 0:
                    price = float(price_data['ltp'])
                    change = float(price_data.get('change', 0))
                    change_pct = float(price_data.get('change_percent', 0))
                    
                    # Color based on change
                    if change > 0:
                        delta_color = "normal"
                    elif change < 0:
                        delta_color = "inverse"
                    else:
                        delta_color = "off"
                    
                    # Change text
                    if change != 0:
                        change_text = f"{change:+.2f} ({change_pct:+.2f}%)"
                    else:
                        change_text = "No change"
                    
                    # Display price metric
                    st.metric(
                        "Price",
                        f"₹{price:,.2f}",
                        delta=change_text if change != 0 else None,
                        delta_color=delta_color
                    )
                    
                    # Show source
                    if is_streaming:
                        st.caption("🔴 LIVE WebSocket Stream")
                    else:
                        st.caption("🔄 API data")
                        
                    # Show timestamp
                    timestamp = price_data.get('timestamp')
                    if timestamp:
                        if isinstance(timestamp, str):
                            st.caption(f"⏰ {timestamp}")
                        elif isinstance(timestamp, datetime):
                            time_str = timestamp.strftime('%H:%M:%S')
                            st.caption(f"⏰ {time_str}")
                else:
                    # No data
                    st.metric("Price", "₹ --")
                    if is_market_open:
                        st.caption("⚠️ Waiting for data...")
                    else:
                        st.caption("⚠️ No data available")
                        
    except Exception as e:
        st.error(f"Error updating prices: {str(e)}")

# The key difference: NO st.rerun() call!
# WebSocket data is fetched each time the function runs, but UI doesn't reload
# The WebSocket handler maintains its own background connection
