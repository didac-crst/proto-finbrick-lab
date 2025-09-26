"""
FinScenLab Strategies Module

This module contains all the concrete strategy implementations for the FinScenLab system.
Strategies implement the actual behavior for different types of financial instruments
based on their 'kind' discriminator.

Strategy Categories:
- Valuation Strategies: Handle asset valuation and cash flow generation
- Schedule Strategies: Handle liability payment schedules and balance tracking  
- Flow Strategies: Handle cash flow events like income, expenses, and transfers

Registry System:
The module automatically registers all default strategies in the global registries,
making them available for use by bricks with matching kind discriminators.

Key Features:
- Modular design allows easy addition of new strategies
- Consistent interface across all strategy types
- Automatic parameter validation and setup
- Support for complex interdependencies between bricks
"""

from __future__ import annotations
import numpy as np
from .core import *

# ---------- Asset Valuation Strategies ----------

class ValuationCash(IValuationStrategy):
    """
    Cash account valuation strategy (kind: 'a.cash').
    
    This strategy models a simple cash account that receives external cash flows
    and earns interest on the balance. The balance is computed by accumulating
    all routed cash flows plus interest earned each month.
    
    Required Parameters:
        - initial_balance: Starting cash balance (default: 0.0)
        - interest_pa: Annual interest rate (default: 0.0)
        
    External Parameters (set by scenario engine):
        - external_in: Monthly cash inflows from other bricks
        - external_out: Monthly cash outflows to other bricks
        
    Note:
        This strategy is designed to be the central cash account that receives
        all cash flows from other bricks in the scenario.
    """
    
    def prepare(self, brick: ABrick, ctx: ScenarioContext) -> None:
        """
        Prepare the cash account strategy.
        
        Sets up default parameters and validates the configuration.
        
        Args:
            brick: The cash account brick
            ctx: The simulation context
        """
        brick.spec.setdefault("initial_balance", 0.0)
        brick.spec.setdefault("interest_pa", 0.0)
        brick.spec.setdefault("external_in",  np.zeros(len(ctx.t_index)))
        brick.spec.setdefault("external_out", np.zeros(len(ctx.t_index)))

    def simulate(self, brick: ABrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the cash account over the time period.
        
        Calculates the monthly balance by accumulating cash flows and applying
        monthly interest. The balance serves as both the asset value and the
        cash flow source/sink.
        
        Args:
            brick: The cash account brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with cash flows, balance as asset value, and no events
        """
        T = len(ctx.t_index)
        bal = np.zeros(T)
        r_m = brick.spec["interest_pa"] / 12.0  # Monthly interest rate
        cash_in  = brick.spec["external_in"].copy()
        cash_out = brick.spec["external_out"].copy()

        # Calculate balance for first month
        bal[0] = brick.spec["initial_balance"] + cash_in[0] - cash_out[0]
        bal[0] += bal[0] * r_m  # Apply interest
        
        # Calculate balance for remaining months
        for t in range(1, T):
            bal[t] = bal[t-1] + cash_in[t] - cash_out[t]
            bal[t] += bal[t] * r_m  # Apply interest

        return BrickOutput(
            cash_in=np.zeros(T),  # Cash account doesn't generate cash flows, only receives them
            cash_out=np.zeros(T), # Cash account doesn't generate cash outflows
            asset_value=bal, 
            debt_balance=np.zeros(T), 
            events=[]
        )

class ValuationPropertyDiscrete(IValuationStrategy):
    """
    Real estate property valuation strategy (kind: 'a.property').
    
    This strategy models a discrete property purchase with upfront costs and
    simple appreciation over time. The property is purchased at t=0 with
    the specified price and fees, then appreciates at a constant annual rate.
    
    Required Parameters:
        - price: Purchase price of the property
        - fees_pct: Transaction fees as percentage of price (e.g., 0.095 for 9.5%)
        - appreciation_pa: Annual appreciation rate (e.g., 0.02 for 2%)
        
    Optional Parameters:
        - down_payment: Down payment amount (used by linked mortgages)
        - finance_fees: Whether to finance the fees (default: False)
        
    Note:
        This strategy is commonly linked to mortgage bricks that auto-calculate
        their principal from the property price minus down payment.
    """
    
    def prepare(self, brick: ABrick, ctx: ScenarioContext) -> None:
        """
        Prepare the property valuation strategy.
        
        Validates that all required parameters are present.
        
        Args:
            brick: The property brick
            ctx: The simulation context
            
        Raises:
            AssertionError: If required parameters are missing
        """
        required_params = ["price", "fees_pct", "appreciation_pa"]
        for param in required_params: 
            assert param in brick.spec, f"Missing required parameter: {param}"

    def simulate(self, brick: ABrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the property over the time period.
        
        Models the property purchase at t=0 and appreciation over time.
        The property value grows at the specified annual rate, while
        the purchase costs are paid upfront.
        
        Args:
            brick: The property brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with purchase costs, appreciating asset value, and purchase event
        """
        T = len(ctx.t_index)
        cash_in  = np.zeros(T)
        cash_out = np.zeros(T)
        value    = np.zeros(T)

        # Extract parameters
        price = float(brick.spec["price"])
        fees  = price * float(brick.spec["fees_pct"])
        finance_fees = bool(brick.spec.get("finance_fees", False))

        # Settlement at t=0: pay seller + fees (if not financed)
        cash_out[0] += price
        if not finance_fees: 
            cash_out[0] += fees

        # Calculate monthly appreciation rate
        r_m = (1 + float(brick.spec["appreciation_pa"])) ** (1/12) - 1
        
        # Set initial value and calculate appreciation
        value[0] = price
        for t in range(1, T): 
            value[t] = value[t-1] * (1 + r_m)

        return BrickOutput(
            cash_in=cash_in, 
            cash_out=cash_out,
            asset_value=value, 
            debt_balance=np.zeros(T),
            events=[f"t0: Purchase {brick.name} - Price: €{price:,.2f}, Fees: €{fees:,.2f}"]
        )

class ValuationETFUnitized(IValuationStrategy):
    """
    ETF investment valuation strategy (kind: 'a.invest.etf').
    
    This strategy models a unitized investment (like an ETF) with constant
    price drift and optional dividend yield. The investment value is calculated
    as units held multiplied by the current price, which grows at a constant rate.
    
    Required Parameters:
        - initial_units: Number of units held at start (default: 0.0)
        - price0: Initial price per unit (default: 100.0)
        - drift_pa: Annual price drift rate (default: 0.03 for 3%)
        - div_yield_pa: Annual dividend yield (default: 0.0)
        
    Note:
        This is a simplified model for proof-of-concept purposes. Real ETF
        strategies would include more sophisticated price modeling, rebalancing,
        and dividend handling.
    """
    
    def prepare(self, brick: ABrick, ctx: ScenarioContext) -> None:
        """
        Prepare the ETF valuation strategy.
        
        Sets up default parameters for the investment.
        
        Args:
            brick: The ETF investment brick
            ctx: The simulation context
        """
        brick.spec.setdefault("initial_units", 0.0)
        brick.spec.setdefault("price0", 100.0)
        brick.spec.setdefault("drift_pa", 0.03)   # 3% annual drift
        brick.spec.setdefault("div_yield_pa", 0.0)

    def simulate(self, brick: ABrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the ETF investment over the time period.
        
        Calculates the monthly price appreciation and generates dividend income
        based on the current asset value. The investment value grows through
        both price appreciation and dividend reinvestment.
        
        Args:
            brick: The ETF investment brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with dividend income, asset value, and no events
        """
        T = len(ctx.t_index)
        units = np.full(T, float(brick.spec["initial_units"]))
        price = np.zeros(T)
        price[0] = float(brick.spec["price0"])
        
        # Calculate monthly drift and dividend rates
        r_m  = (1 + float(brick.spec["drift_pa"])) ** (1/12) - 1
        divm = float(brick.spec["div_yield_pa"]) / 12.0

        # Calculate price appreciation over time
        for t in range(1, T): 
            price[t] = price[t-1] * (1 + r_m)
        
        # Calculate asset value and dividend income
        asset_value = units * price
        cash_in  = divm * asset_value  # Monthly dividend income
        cash_out = np.zeros(T)

        return BrickOutput(
            cash_in=cash_in, 
            cash_out=cash_out,
            asset_value=asset_value, 
            debt_balance=np.zeros(T),
            events=[]
        )

# ---------- Liability Schedule Strategies ----------

class ScheduleMortgageAnnuity(IScheduleStrategy):
    """
    Fixed-rate mortgage with annuity payment schedule (kind: 'l.mortgage.annuity').
    
    This strategy models a traditional fixed-rate mortgage with equal monthly payments
    that include both principal and interest. The principal can be provided directly
    or automatically calculated from a linked property minus the down payment.
    
    Required Parameters:
        - rate_pa: Annual interest rate (e.g., 0.034 for 3.4%)
        - term_months: Total term in months (e.g., 300 for 25 years)
        - principal: Loan principal amount (can be auto-calculated)
        
    Optional Parameters (for auto-calculation):
        - links: Dictionary with 'auto_principal_from' key pointing to property brick ID
        
    Note:
        If principal is not provided, it will be calculated from the linked property's
        price minus down_payment. This enables automatic mortgage sizing based on
        property purchases.
    """
    
    def prepare(self, brick: LBrick, ctx: ScenarioContext) -> None:
        """
        Prepare the mortgage strategy.
        
        Validates parameters and optionally calculates principal from linked property.
        
        Args:
            brick: The mortgage brick
            ctx: The simulation context
            
        Raises:
            AssertionError: If required parameters are missing or auto-calculation fails
        """
        # Auto-calculate principal from linked property if not provided
        if "principal" not in brick.spec:
            auto_from = brick.links.get("auto_principal_from")
            assert auto_from in ctx.registry, "auto_principal_from link missing or invalid"
            prop: ABrick = ctx.registry[auto_from]  # type: ignore
            price = float(prop.spec["price"])
            down  = float(prop.spec.get("down_payment", 0.0))
            brick.spec["principal"] = price - down
        
        # Validate all required parameters
        required_params = ["rate_pa", "term_months", "principal"]
        for param in required_params: 
            assert param in brick.spec, f"Missing required parameter: {param}"

    def simulate(self, brick: LBrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the mortgage over the time period.
        
        Calculates the annuity payment schedule with equal monthly payments
        that include both principal and interest. The debt balance decreases
        over time as principal is paid down.
        
        Args:
            brick: The mortgage brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with loan drawdown, payment schedule, debt balance, and drawdown event
        """
        T = len(ctx.t_index)
        cash_in  = np.zeros(T)
        cash_out = np.zeros(T)
        debt     = np.zeros(T)

        # Extract parameters
        principal = float(brick.spec["principal"])
        rate_pa   = float(brick.spec["rate_pa"])
        n_total   = int(brick.spec["term_months"])
        n = min(n_total, T)  # Don't exceed simulation period

        # Initial loan drawdown at t=0
        cash_in[0] += principal
        debt[0] = principal

        # Calculate monthly payment using annuity formula
        r_m = rate_pa / 12.0
        if r_m > 0:
            A = principal * (r_m * (1 + r_m) ** n) / ((1 + r_m) ** n - 1)
        else:
            A = principal / n  # Handle zero interest rate case
        
        # Calculate payment schedule
        for t in range(n):
            interest = debt[t] * r_m
            principal_pay = min(A - interest, debt[t])  # Don't overpay
            cash_out[t] += interest + principal_pay
            
            # Update debt balance for next period
            if t + 1 < T:
                debt[t+1] = max(debt[t] - principal_pay, 0.0)

        return BrickOutput(
            cash_in=cash_in, 
            cash_out=cash_out,
            asset_value=np.zeros(T), 
            debt_balance=debt,
            events=[f"t0: Mortgage drawdown - Principal: €{principal:,.2f}"]
        )

# ---------- Cash Flow Strategies ----------

class FlowTransferLumpSum(IFlowStrategy):
    """
    Lump sum transfer flow strategy (kind: 'f.transfer.lumpsum').
    
    This strategy models a one-time cash transfer that occurs at t=0.
    Commonly used for initial capital injections, windfalls, or large
    one-time payments.
    
    Required Parameters:
        - amount: The lump sum amount to transfer at t=0
        
    Note:
        This strategy generates a single cash inflow at the beginning
        of the simulation period.
    """
    
    def prepare(self, brick: FBrick, ctx: ScenarioContext) -> None:
        """
        Prepare the lump sum transfer strategy.
        
        Validates that the amount parameter is present.
        
        Args:
            brick: The transfer flow brick
            ctx: The simulation context
            
        Raises:
            AssertionError: If amount parameter is missing
        """
        assert "amount" in brick.spec, "Missing required parameter: amount"

    def simulate(self, brick: FBrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the lump sum transfer.
        
        Generates a single cash inflow at t=0 with the specified amount.
        
        Args:
            brick: The transfer flow brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with single cash inflow at t=0 and transfer event
        """
        T = len(ctx.t_index)
        cash_in = np.zeros(T)
        cash_in[0] = float(brick.spec["amount"])
        
        return BrickOutput(
            cash_in=cash_in, 
            cash_out=np.zeros(T),
            asset_value=np.zeros(T), 
            debt_balance=np.zeros(T), 
            events=[f"t0: Lump sum transfer - Amount: €{cash_in[0]:,.2f}"]
        )


class FlowIncomeFixed(IFlowStrategy):
    """
    Fixed monthly income flow strategy (kind: 'f.income.salary').
    
    This strategy models a regular monthly income stream with a constant amount.
    Commonly used for salary, pension, rental income, or other regular income sources.
    
    Required Parameters:
        - amount_monthly: The monthly income amount
        
    Note:
        This strategy generates the same cash inflow every month throughout
        the simulation period.
    """
    
    def prepare(self, brick: FBrick, ctx: ScenarioContext) -> None:
        """
        Prepare the fixed income strategy.
        
        Validates that the amount_monthly parameter is present.
        
        Args:
            brick: The income flow brick
            ctx: The simulation context
            
        Raises:
            AssertionError: If amount_monthly parameter is missing
        """
        assert "amount_monthly" in brick.spec, "Missing required parameter: amount_monthly"

    def simulate(self, brick: FBrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the fixed monthly income.
        
        Generates a constant monthly cash inflow throughout the simulation period.
        
        Args:
            brick: The income flow brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with constant monthly cash inflows and no events
        """
        T = len(ctx.t_index)
        cash_in = np.full(T, float(brick.spec["amount_monthly"]))
        
        return BrickOutput(
            cash_in=cash_in, 
            cash_out=np.zeros(T),
            asset_value=np.zeros(T), 
            debt_balance=np.zeros(T), 
            events=[]
        )


class FlowExpenseFixed(IFlowStrategy):
    """
    Fixed monthly expense flow strategy (kind: 'f.expense.living').
    
    This strategy models a regular monthly expense with a constant amount.
    Commonly used for living expenses, insurance, subscriptions, or other
    regular recurring costs.
    
    Required Parameters:
        - amount_monthly: The monthly expense amount
        
    Note:
        This strategy generates the same cash outflow every month throughout
        the simulation period.
    """
    
    def prepare(self, brick: FBrick, ctx: ScenarioContext) -> None:
        """
        Prepare the fixed expense strategy.
        
        Validates that the amount_monthly parameter is present.
        
        Args:
            brick: The expense flow brick
            ctx: The simulation context
            
        Raises:
            AssertionError: If amount_monthly parameter is missing
        """
        assert "amount_monthly" in brick.spec, "Missing required parameter: amount_monthly"

    def simulate(self, brick: FBrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the fixed monthly expense.
        
        Generates a constant monthly cash outflow throughout the simulation period.
        
        Args:
            brick: The expense flow brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with constant monthly cash outflows and no events
        """
        T = len(ctx.t_index)
        cash_out = np.full(T, float(brick.spec["amount_monthly"]))
        
        return BrickOutput(
            cash_in=np.zeros(T), 
            cash_out=cash_out,
            asset_value=np.zeros(T), 
            debt_balance=np.zeros(T), 
            events=[]
        )

# ---------- Strategy Registry Setup ----------

def register_defaults():
    """
    Register all default strategy implementations in the global registries.
    
    This function populates the global strategy registries with the default
    implementations provided by FinScenLab. These strategies are automatically
    available for use by bricks with matching kind discriminators.
    
    Registered Strategies:
        Assets:
            - 'a.cash': Cash account with interest
            - 'a.property': Real estate with appreciation
            - 'a.invest.etf': ETF investment with price drift
            
        Liabilities:
            - 'l.mortgage.annuity': Fixed-rate mortgage with annuity payments
            
        Flows:
            - 'f.transfer.lumpsum': One-time lump sum transfer
            - 'f.income.salary': Fixed monthly income
            - 'f.expense.living': Fixed monthly expenses
            
    Note:
        This function is automatically called when the module is imported.
        Additional strategies can be registered by calling the registry
        dictionaries directly.
    """
    # Register asset valuation strategies
    ValuationRegistry["a.cash"]      = ValuationCash()
    ValuationRegistry["a.property"]  = ValuationPropertyDiscrete()
    ValuationRegistry["a.invest.etf"]= ValuationETFUnitized()
    
    # Register liability schedule strategies
    ScheduleRegistry["l.mortgage.annuity"] = ScheduleMortgageAnnuity()
    
    # Register cash flow strategies
    FlowRegistry["f.transfer.lumpsum"] = FlowTransferLumpSum()
    FlowRegistry["f.income.salary"]    = FlowIncomeFixed()
    FlowRegistry["f.expense.living"]   = FlowExpenseFixed()


# Automatically register default strategies when module is imported
register_defaults()
