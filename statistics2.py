import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt
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
    
    def get_seasonal_factor(self, date: datetime) -> float:
        return self.SEASONAL_FACTORS.get(date.month, 1.0)
    
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

def plot_results(results_df: pd.DataFrame, config: Dict):
    plt.figure(figsize=(15, 10))
    
    # Revenue Plot
    plt.subplot(2, 2, 1)
    plt.plot(results_df['week'], results_df['total_revenue'], label='Total Revenue')
    plt.plot(results_df['week'], results_df['total_platform_fees'], label='Platform Fees')
    plt.plot(results_df['week'], results_df['total_commission'], label='Commission')
    plt.title('Weekly Revenue Components')
    plt.xlabel('Week')
    plt.ylabel('Amount (₹)')
    plt.legend()
    
    # Driver Activity Plot
    plt.subplot(2, 2, 2)
    plt.plot(results_df['week'], results_df['active_drivers'], label='Active Drivers')
    plt.plot(results_df['week'], results_df['blocked_drivers'], label='Blocked Drivers')
    plt.title('Driver Activity Status')
    plt.xlabel('Week')
    plt.ylabel('Number of Drivers')
    plt.legend()
    
    # Seasonal Factor Plot
    plt.subplot(2, 2, 3)
    plt.plot(results_df['week'], results_df['seasonal_factor'])
    plt.title('Seasonal Variations')
    plt.xlabel('Week')
    plt.ylabel('Seasonal Factor')
    
    # Summary Text
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = (
        f"Configuration Summary:\n"
        f"Platform Fee: ₹{config['platform_fee']}/day\n"
        f"Commission Rate: {config['commission_rate']*100}%\n\n"
        f"Key Metrics:\n"
        f"Avg Weekly Revenue: ₹{results_df['total_revenue'].mean():,.0f}\n"
        f"Avg Active Drivers: {results_df['active_drivers'].mean():.1f}\n"
        f"Driver Retention: {(results_df['active_drivers'].mean()/config['num_drivers'])*100:.1f}%\n"
        f"Avg Monthly Revenue: ₹{results_df['total_revenue'].mean()*4:,.0f}\n"
    )
    plt.text(0.1, 0.5, summary_text, fontsize=10)
    
    plt.tight_layout()
    return plt.gcf()

def analyze_kerala_configuration(
    platform_fee: float,
    commission_rate: float,
    num_drivers: int = 100,
    simulation_weeks: int = 52
) -> Dict:
    config = {
        'platform_fee': platform_fee,
        'commission_rate': commission_rate,
        'num_drivers': num_drivers,
        'simulation_weeks': simulation_weeks
    }
    
    sim = KeralaBusinessSimulation(**config)
    results_df = sim.run_simulation()
    
    fig = plot_results(results_df, config)
    
    metrics = {
        'platform_fee': platform_fee,
        'commission_rate': commission_rate,
        'annual_revenue': results_df['total_revenue'].sum(),
        'avg_weekly_revenue': results_df['total_revenue'].mean(),
        'avg_active_drivers': results_df['active_drivers'].mean(),
        'driver_retention_rate': results_df['active_drivers'].mean() / num_drivers,
        'seasonal_impact': results_df['seasonal_factor'].std(),
        'visualization': fig
    }
    
    return metrics

if __name__ == "__main__":
    # Test configuration
    platform_fee = 5  # ₹40 per day
    commission_rate = 0.05  # 8%
    num_drivers = 30
    
    results = analyze_kerala_configuration(
        platform_fee=platform_fee,
        commission_rate=commission_rate,
        num_drivers=num_drivers
    )
    
    print("\nSimulation Results:")
    print(f"Annual Revenue: ₹{results['annual_revenue']:,.2f}")
    print(f"Average Weekly Revenue: ₹{results['avg_weekly_revenue']:,.2f}")
    print(f"Driver Retention Rate: {results['driver_retention_rate']*100:.1f}%")
    print(f"Seasonal Variation (std): {results['seasonal_impact']:.3f}")
    
    plt.show()