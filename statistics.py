import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import matplotlib.pyplot as plt

@dataclass
class Driver:
    id: int
    weekly_profit: float
    active_days: int
    payment_reliability: float  # 0-1, likelihood of paying tabs promptly
    pending_tabs: List[float] = None
    
    def __post_init__(self):
        self.pending_tabs = []
        
    def will_pay_tab(self) -> bool:
        return np.random.random() < self.payment_reliability

class BusinessSimulation:
    def __init__(
        self,
        platform_fee: float,
        commission_rate: float,
        num_drivers: int,
        simulation_weeks: int = 52,
        max_pending_tabs: int = 4
    ):
        self.platform_fee = platform_fee
        self.commission_rate = commission_rate
        self.num_drivers = num_drivers
        self.simulation_weeks = simulation_weeks
        self.max_pending_tabs = max_pending_tabs
        self.drivers = self._initialize_drivers()
        
    def _initialize_drivers(self) -> List[Driver]:
        """Create a realistic distribution of drivers with varying behaviors"""
        drivers = []
        for i in range(self.num_drivers):
            # Generate realistic driver parameters
            weekly_profit = np.random.normal(7000, 1500)  # Mean ₹7000, SD ₹1500
            active_days = np.random.randint(4, 8)  # 4-7 days per week
            payment_reliability = np.random.beta(7, 3)  # Right-skewed distribution
            
            drivers.append(Driver(
                id=i,
                weekly_profit=max(weekly_profit, 3000),  # Minimum ₹3000
                active_days=active_days,
                payment_reliability=payment_reliability
            ))
        return drivers
    
    def simulate_week(self) -> Dict:
        """Simulate one week of operations"""
        weekly_stats = {
            'total_platform_fees': 0,
            'total_commission': 0,
            'total_tabs_paid': 0,
            'total_tabs_added': 0,
            'active_drivers': 0,
            'blocked_drivers': 0
        }
        
        for driver in self.drivers:
            # Check if driver can operate (less than max pending tabs)
            if len(driver.pending_tabs) >= self.max_pending_tabs:
                if driver.will_pay_tab():
                    # Pay oldest tab
                    paid_amount = driver.pending_tabs.pop(0)
                    weekly_stats['total_tabs_paid'] += paid_amount
                else:
                    weekly_stats['blocked_drivers'] += 1
                    continue
            
            weekly_stats['active_drivers'] += 1
            
            # Calculate fees for the week
            platform_fees = self.platform_fee * driver.active_days
            commission = driver.weekly_profit * self.commission_rate
            
            # Add new tab
            new_tab = platform_fees + commission
            driver.pending_tabs.append(new_tab)
            
            weekly_stats['total_platform_fees'] += platform_fees
            weekly_stats['total_commission'] += commission
            weekly_stats['total_tabs_added'] += new_tab
            
        return weekly_stats
    
    def run_simulation(self) -> pd.DataFrame:
        """Run the complete simulation"""
        results = []
        
        for week in range(self.simulation_weeks):
            weekly_stats = self.simulate_week()
            weekly_stats['week'] = week + 1
            weekly_stats['total_revenue'] = (
                weekly_stats['total_tabs_paid'] +
                weekly_stats['total_platform_fees'] +
                weekly_stats['total_commission']
            )
            results.append(weekly_stats)
            
        return pd.DataFrame(results)

def analyze_configuration(
    platform_fee: float,
    commission_rate: float,
    num_drivers: int = 100,
    simulation_weeks: int = 52
) -> Dict:
    """Analyze a specific fee and commission configuration"""
    sim = BusinessSimulation(
        platform_fee=platform_fee,
        commission_rate=commission_rate,
        num_drivers=num_drivers,
        simulation_weeks=simulation_weeks
    )
    
    results_df = sim.run_simulation()
    
    # Calculate key metrics
    metrics = {
        'platform_fee': platform_fee,
        'commission_rate': commission_rate,
        'annual_revenue': results_df['total_revenue'].sum(),
        'avg_weekly_revenue': results_df['total_revenue'].mean(),
        'avg_active_drivers': results_df['active_drivers'].mean(),
        'avg_blocked_drivers': results_df['blocked_drivers'].mean(),
        'driver_retention_rate': (
            results_df['active_drivers'].mean() / num_drivers
        ),
        'tab_collection_rate': (
            results_df['total_tabs_paid'].sum() /
            results_df['total_tabs_added'].sum()
        )
    }
    
    return metrics

def find_optimal_configuration(
    platform_fees: List[float],
    commission_rates: List[float],
    num_drivers: int = 100
) -> pd.DataFrame:
    """Test different combinations to find optimal configurations"""
    results = []
    
    for fee in platform_fees:
        for rate in commission_rates:
            metrics = analyze_configuration(
                platform_fee=fee,
                commission_rate=rate,
                num_drivers=num_drivers
            )
            results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    # Calculate driver-friendliness score (0-100)
    results_df['driver_friendliness'] = (
        results_df['driver_retention_rate'] * 60 +
        results_df['tab_collection_rate'] * 40
    ) * 100
    
    # Calculate profit-driver balance score (0-100)
    max_revenue = results_df['annual_revenue'].max()
    results_df['revenue_score'] = (
        results_df['annual_revenue'] / max_revenue * 100
    )
    
    results_df['balance_score'] = (
        results_df['revenue_score'] * 0.5 +
        results_df['driver_friendliness'] * 0.5
    )
    
    return results_df

if __name__ == "__main__":
    # Test range of configurations
    platform_fees = [3, 4, 5, 6, 7]  # Daily platform fees in ₹
    commission_rates = [0.05, 0.06, 0.07, 0.08, 0.10]  # 5-10%
    
    results = find_optimal_configuration(platform_fees, commission_rates)
    
    print("\nTop 5 Most Balanced Configurations:")
    print(results.sort_values('balance_score', ascending=False).head())
    
    print("\nTop 5 Most Profitable Configurations:")
    print(results.sort_values('annual_revenue', ascending=False).head())
    
    print("\nTop 5 Most Driver-Friendly Configurations:")
    print(results.sort_values('driver_friendliness', ascending=False).head())