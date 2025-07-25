"""
Test file to demonstrate Streamlit WebSocket streaming with fragments

This shows the correct implementation approach:
1. Use @st.fragment(run_every="1s") decorator
2. The fragment updates automatically every second
3. Only the fragment content is updated, not the whole page
4. No UI dimming or loading states
"""

import streamlit as st
import time
from datetime import datetime

st.set_page_config(page_title="WebSocket Test", layout="wide")

st.title("WebSocket Streaming Test with Fragments")

# Static content outside fragment (won't change)
st.info("This is static content - it won't change when fragment updates")

# Initialize session state for demo data
if 'demo_prices' not in st.session_state:
    st.session_state.demo_prices = {
        'RELIANCE': {'price': 1400, 'change': -5.20},
        'TCS': {'price': 3150, 'change': 2.50},
        'HDFCBANK': {'price': 2000, 'change': -1.80},
        'INFY': {'price': 1530, 'change': 0.90}
    }

@st.fragment(run_every="1s")  # This will update every second automatically
def streaming_prices():
    """This fragment updates every 1 second without any UI refresh"""
    current_time = datetime.now()
    
    # Status bar
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        st.success("🟢 Market OPEN")
    with col2:
        st.success("🔴 LIVE Stream")
    with col3:
        st.caption(f"⏰ {current_time.strftime('%H:%M:%S')}")
    
    # Simulate live price changes
    import random
    for symbol in st.session_state.demo_prices:
        # Small random price movements
        change = random.uniform(-0.5, 0.5)
        st.session_state.demo_prices[symbol]['price'] += change
        st.session_state.demo_prices[symbol]['change'] += change
    
    # Display prices in columns
    cols = st.columns(4)
    for i, (symbol, data) in enumerate(st.session_state.demo_prices.items()):
        with cols[i]:
            price = data['price']
            change = data['change']
            
            st.subheader(f"🔴 {symbol} LIVE")
            
            # Color logic
            delta_color = "normal" if change > 0 else "inverse" if change < 0 else "off"
            change_text = f"{change:+.2f}" if change != 0 else None
            
            st.metric(
                "Price",
                f"₹{price:,.2f}",
                delta=change_text,
                delta_color=delta_color
            )
            
            st.caption("🔴 LIVE WebSocket Stream")
            st.caption(f"⏰ {current_time.strftime('%H:%M:%S')}")

# Call the fragment function
streaming_prices()

# More static content
st.markdown("---")
st.info("Notice how the prices update every second without the page dimming or showing 'RUNNING...' indicator")
st.info("This is the power of Streamlit fragments for real-time streaming!")
