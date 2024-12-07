import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

@dataclass
class Driver:
    id: int
    weekly_profit: float
    active_days: int
    payment_reliability: float
    preferred_areas: List[str]
    shift_preference: str
    experience_years: float
    pending_tabs: List[float] = None
    
    def __post_init__(self):
        self.pending_tabs = []
        
    def will_pay_tab(self) -> bool:
        return np.random.random() < self.payment_reliability

class KeralaBusinessSimulation:
    KERALA_AREAS = [
        'Thiruvananthapuram', 'Kochi', 'Kozhikode', 'Thrissur', 
        'Kollam', 'Alappuzha', 'Kannur', 'Kottayam'
    ]
    
    SEASONAL_FACTORS = {
        6: 0.8, 7: 0.7, 8: 0.7,  # Monsoon
        9: 1.3, 10: 1.4,         # Festival
        12: 1.3, 1: 1.4, 2: 1.3, # Tourist
        3: 1.0, 4: 1.0, 5: 1.0, 11: 1.0  # Regular
    }
    
    def __init__(
        self,
        platform_fee: float,
        commission_rate: float,
        num_drivers: int,
        simulation_weeks: int = 52,
        max_pending_tabs: int = 4,
        start_date: datetime = datetime(2024, 1, 1)
    ):
        self.platform_fee = platform_fee
        self.commission_rate = commission_rate
        self.num_drivers = num_drivers
        self.simulation_weeks = simulation_weeks
        self.max_pending_tabs = max_pending_tabs
        self.start_date = start_date
        self.drivers = self._initialize_kerala_drivers()
    
    def _initialize_kerala_drivers(self) -> List[Driver]:
        drivers = []
        area_profit_ranges = {
            'urban': (600, 900),
            'suburban': (500, 750),
            'rural': (400, 600)
        }
        
        for i in range(self.num_drivers):
            num_areas = np.random.randint(1, 4)
            areas = np.random.choice(self.KERALA_AREAS, num_areas, replace=False)
            
            area_type = np.random.choice(['urban', 'suburban', 'rural'], p=[0.4, 0.35, 0.25])
            base_min, base_max = area_profit_ranges[area_type]
            
            weekly_profit = np.random.uniform(base_min, base_max)
            active_days = np.random.randint(5, 8)
            experience = np.random.uniform(0.5, 15)
            reliability_base = min(0.5 + (experience / 15) * 0.4, 0.9)
            payment_reliability = np.random.beta(
                reliability_base * 10,
                (1 - reliability_base) * 10
            )
            
            shift_preference = np.random.choice([
                'morning', 'afternoon', 'evening', 'night'
            ])
            
            drivers.append(Driver(
                id=i,
                weekly_profit=weekly_profit,
                active_days=active_days,
                payment_reliability=payment_reliability,
                preferred_areas=list(areas),
                shift_preference=shift_preference,
                experience_years=experience
            ))
        
        return drivers

    def simulate_week(self, current_date: datetime) -> Dict:
        seasonal_factor = self.get_seasonal_factor(current_date)
        
        weekly_stats = {
            'date': current_date,
            'total_platform_fees': 0,
            'total_commission': 0,
            'total_tabs_paid': 0,
            'total_tabs_added': 0,
            'active_drivers': 0,
            'blocked_drivers': 0,
            'seasonal_factor': seasonal_factor,
            'drivers_by_shift': {
                'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0
            },
            'drivers_by_area': {area: 0 for area in self.KERALA_AREAS}
        }
        
        for driver in self.drivers:
            if len(driver.pending_tabs) >= self.max_pending_tabs:
                if driver.will_pay_tab():
                    paid_amount = driver.pending_tabs.pop(0)
                    weekly_stats['total_tabs_paid'] += paid_amount
                else:
                    weekly_stats['blocked_drivers'] += 1
                    continue
            
            weekly_stats['active_drivers'] += 1
            weekly_stats['drivers_by_shift'][driver.shift_preference] += 1
            for area in driver.preferred_areas:
                weekly_stats['drivers_by_area'][area] += 1
            
            adjusted_profit = driver.weekly_profit * seasonal_factor
            platform_fees = self.platform_fee * driver.active_days
            commission = adjusted_profit * self.commission_rate
            
            new_tab = platform_fees + commission
            driver.pending_tabs.append(new_tab)
            
            weekly_stats['total_platform_fees'] += platform_fees
            weekly_stats['total_commission'] += commission
            weekly_stats['total_tabs_added'] += new_tab
        
        return weekly_stats
    
    def get_seasonal_factor(self, date: datetime) -> float:
        return self.SEASONAL_FACTORS.get(date.month, 1.0)
    
    def run_simulation(self) -> pd.DataFrame:
        results = []
        current_date = self.start_date
        
        for week in range(self.simulation_weeks):
            weekly_stats = self.simulate_week(current_date)
            weekly_stats['week'] = week + 1
            weekly_stats['total_revenue'] = (
                weekly_stats['total_tabs_paid'] +
                weekly_stats['total_platform_fees'] +
                weekly_stats['total_commission']
            )
            results.append(weekly_stats)
            current_date += timedelta(days=7)
            
        return pd.DataFrame(results)

def create_interactive_plots(results_df: pd.DataFrame, config: Dict):
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Weekly Revenue Components',
            'Driver Activity Status',
            'Seasonal Variations',
            'Key Metrics'
        )
    )

    # Revenue Plot
    fig.add_trace(
        go.Scatter(x=results_df['week'], y=results_df['total_revenue'],
                  name='Total Revenue', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results_df['week'], y=results_df['total_platform_fees'],
                  name='Platform Fees', line=dict(color='green')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=results_df['week'], y=results_df['total_commission'],
                  name='Commission', line=dict(color='red')),
        row=1, col=1
    )

    # Driver Activity Plot
    fig.add_trace(
        go.Scatter(x=results_df['week'], y=results_df['active_drivers'],
                  name='Active Drivers', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=results_df['week'], y=results_df['blocked_drivers'],
                  name='Blocked Drivers', line=dict(color='red')),
        row=1, col=2
    )

    # Seasonal Factor Plot
    fig.add_trace(
        go.Scatter(x=results_df['week'], y=results_df['seasonal_factor'],
                  name='Seasonal Factor', line=dict(color='purple')),
        row=2, col=1
    )

    # Summary Text
    summary_text = (
        f"Configuration Summary:<br>"
        f"Platform Fee: ₹{config['platform_fee']}/day<br>"
        f"Commission Rate: {config['commission_rate']*100}%<br><br>"
        f"Key Metrics:<br>"
        f"Avg Weekly Revenue: ₹{results_df['total_revenue'].mean():,.0f}<br>"
        f"Avg Active Drivers: {results_df['active_drivers'].mean():.1f}<br>"
        f"Driver Retention: {(results_df['active_drivers'].mean()/config['num_drivers'])*100:.1f}%<br>"
        f"Avg Monthly Revenue: ₹{results_df['total_revenue'].mean()*4:,.0f}"
    )
    
    fig.add_annotation(
        text=summary_text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        row=2, col=2
    )

    fig.update_layout(height=800, showlegend=True)
    return fig

def main():
    st.title("Kerala Business Simulation Dashboard")
    st.write("Configure simulation parameters and analyze the results")

    # Sidebar for parameters
    st.sidebar.header("Simulation Parameters")
    
    platform_fee = st.sidebar.slider(
        "Platform Fee (₹/day)",
        min_value=5,
        max_value=100,
        value=40,
        step=5
    )
    
    commission_rate = st.sidebar.slider(
        "Commission Rate (%)",
        min_value=1.0,
        max_value=20.0,
        value=8.0,
        step=0.5
    ) / 100
    
    num_drivers = st.sidebar.slider(
        "Number of Drivers",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )
    
    simulation_weeks = st.sidebar.slider(
        "Simulation Weeks",
        min_value=12,
        max_value=104,
        value=52,
        step=4
    )

    # Run simulation button
    if st.sidebar.button("Run Simulation"):
        config = {
            'platform_fee': platform_fee,
            'commission_rate': commission_rate,
            'num_drivers': num_drivers,
            'simulation_weeks': simulation_weeks
        }
        
        with st.spinner("Running simulation..."):
            sim = KeralaBusinessSimulation(**config)
            results_df = sim.run_simulation()
            
            # Display interactive plots
            fig = create_interactive_plots(results_df, config)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional metrics in expandable sections
            with st.expander("Detailed Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Annual Revenue",
                        f"₹{results_df['total_revenue'].sum():,.0f}"
                    )
                    st.metric(
                        "Average Weekly Revenue",
                        f"₹{results_df['total_revenue'].mean():,.0f}"
                    )
                
                with col2:
                    st.metric(
                        "Driver Retention Rate",
                        f"{(results_df['active_drivers'].mean()/num_drivers)*100:.1f}%"
                    )
                    st.metric(
                        "Seasonal Impact (std)",
                        f"{results_df['seasonal_factor'].std():.3f}"
                    )
            
            # Show shift distribution
            with st.expander("Driver Shift Distribution"):
                last_week = results_df.iloc[-1]['drivers_by_shift']
                shift_df = pd.DataFrame(last_week.items(), columns=['Shift', 'Count'])
                st.bar_chart(shift_df.set_index('Shift'))
            
            # Show area distribution
            with st.expander("Area Distribution"):
                last_week_areas = results_df.iloc[-1]['drivers_by_area']
                area_df = pd.DataFrame(last_week_areas.items(), columns=['Area', 'Count'])
                st.bar_chart(area_df.set_index('Area'))

if __name__ == "__main__":
    main()