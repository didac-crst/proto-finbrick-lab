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
from .kinds import K

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
        
        Sets up default parameters, liquidity policy, and validates the configuration.
        
        Args:
            brick: The cash account brick
            ctx: The simulation context
        """
        brick.spec.setdefault("initial_balance", 0.0)
        brick.spec.setdefault("interest_pa", 0.0)
        brick.spec.setdefault("external_in",  np.zeros(len(ctx.t_index)))
        brick.spec.setdefault("external_out", np.zeros(len(ctx.t_index)))
        
        # Set liquidity policy defaults
        brick.spec.setdefault("overdraft_limit", 0.0)  # how far below 0 cash may go (EUR)
        brick.spec.setdefault("min_buffer", 0.0)       # desired minimum cash balance (EUR)
        
        # Validate non-negative constraints
        assert brick.spec["overdraft_limit"] >= 0, "overdraft_limit must be >= 0"
        assert brick.spec["min_buffer"] >= 0, "min_buffer must be >= 0"
        
        # Warn if min_buffer > initial_balance (policy breach, not config error)
        initial_balance = brick.spec.get("initial_balance", 0.0)
        if brick.spec["min_buffer"] > initial_balance:
            print(f"[WARN] {brick.id}: min_buffer ({brick.spec['min_buffer']:,.2f}) > initial_balance ({initial_balance:,.2f}).")

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
        
        # Calculate fees financing (new logic with percentage support)
        fees_fin_pct = float(brick.spec.get("fees_financed_pct", 1.0 if brick.spec.get("finance_fees") else 0.0))
        fees_fin_pct = max(0.0, min(1.0, fees_fin_pct))  # Clamp to [0,1]
        fees_cash = fees * (1.0 - fees_fin_pct)

        # t0 settlement: pay seller + cash portion of fees ONCE
        cash_out[0] = price + fees_cash

        # Calculate monthly appreciation rate
        r_m = (1 + float(brick.spec["appreciation_pa"])) ** (1/12) - 1
        
        # Set initial value and calculate appreciation
        value[0] = price
        for t in range(1, T): 
            value[t] = value[t-1] * (1 + r_m)

        
        # Create time-stamped events
        events = [
            Event(ctx.t_index[0], "purchase", f"Purchase {brick.name}", {"price": price}),
        ]
        if fees_cash > 0:
            events.append(Event(ctx.t_index[0], "fees_cash", f"Fees paid from cash: €{fees_cash:,.2f}",
                                {"fees": fees, "fees_cash": fees_cash}))
        if fees_fin_pct > 0:
            events.append(Event(ctx.t_index[0], "fees_financed", f"Fees financed: €{fees * fees_fin_pct:,.2f}",
                                {"fees": fees, "fees_financed": fees * fees_fin_pct}))

        return BrickOutput(
            cash_in=cash_in, 
            cash_out=cash_out,
            asset_value=value, 
            debt_balance=np.zeros(T),
            events=events
        )

class ValuationETFUnitized(IValuationStrategy):
    """
    ETF investment valuation strategy (kind: 'a.invest.etf').
    
    This strategy models a unitized investment (like an ETF) with constant
    price drift, optional dividend yield, and support for purchasing shares
    through one-shot buys and dollar-cost averaging (DCA).
    
    Key Features:
        - Initial holdings (pre-owned units with no cash impact)
        - One-shot buy at start (buy_at_start by amount or units)
        - DCA plan by amount or units, with optional annual step-up
        - Dividend reinvestment or cash distribution
        - Configurable event logging
        
    Parameters:
        - initial_units: Number of units held at start (default: 0.0)
        - price0: Initial price per unit (default: 100.0)
        - drift_pa: Annual price drift rate (default: 0.03 for 3%)
        - div_yield_pa: Annual dividend yield (default: 0.0)
        - reinvest_dividends: Whether to reinvest dividends (default: False)
        - buy_at_start: One-shot purchase {"amount": X} or {"units": Y}
        - dca: DCA configuration with mode, amount/units, timing, and step-up
        - round_units_to: Round units to N decimal places (optional)
        - events_level: Event verbosity "none"|"major"|"all" (default: "major")
    """
    
    def prepare(self, brick: ABrick, ctx: ScenarioContext) -> None:
        """
        Prepare the ETF valuation strategy.
        
        Sets up default parameters and validates the configuration.
        
        Args:
            brick: The ETF investment brick
            ctx: The simulation context
        """
        s = brick.spec
        s.setdefault("initial_units", 0.0)
        s.setdefault("price0", 100.0)
        s.setdefault("drift_pa", 0.03)
        s.setdefault("div_yield_pa", 0.0)
        s.setdefault("reinvest_dividends", False)
        s.setdefault("buy_at_start", None)      # {"amount": >0} or {"units": >0}
        s.setdefault("dca", None)               # {"mode": "amount"|"units", ...}
        s.setdefault("round_units_to", None)
        s.setdefault("events_level", "major")   # "none"|"major"|"all"

        # Validate DCA configuration
        dca = s["dca"]
        if dca is not None:
            mode = dca.get("mode")
            assert mode in ("amount", "units"), "dca.mode must be 'amount' or 'units'"
            if mode == "amount":
                assert dca.get("amount", 0) >= 0, "dca.amount must be >= 0"
            else:
                assert dca.get("units", 0) >= 0, "dca.units must be >= 0"
            
            # Normalize offsets
            off = int(dca.get("start_offset_m", 0))
            if off < 0:
                dca["start_offset_m"] = 0
                print(f"[WARN] {brick.id}: dca.start_offset_m < 0 -> clamped to 0")
            dca.setdefault("months", None)
            dca.setdefault("annual_step_pct", 0.0)

        # Validate buy_at_start configuration
        if s["buy_at_start"]:
            bas = s["buy_at_start"]
            assert ("amount" in bas) ^ ("units" in bas), "buy_at_start: provide exactly one of {'amount','units'}"
            if "amount" in bas: 
                assert bas["amount"] >= 0, "buy_at_start.amount must be >= 0"
            if "units" in bas:  
                assert bas["units"] >= 0, "buy_at_start.units must be >= 0"

    def simulate(self, brick: ABrick, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the ETF investment over the time period.
        
        Handles initial holdings, one-shot purchases, DCA contributions,
        dividend payments/reinvestment, and price appreciation.
        
        Args:
            brick: The ETF investment brick
            ctx: The simulation context
            
        Returns:
            BrickOutput with cash flows, asset value, and events
        """
        T = len(ctx.t_index)
        s = brick.spec
        cash_in  = np.zeros(T)    # dividends (if not reinvested)
        cash_out = np.zeros(T)    # purchases
        units    = np.zeros(T)
        price    = np.zeros(T)
        events   = []

        # Price path calculation
        r_m = (1 + float(s["drift_pa"])) ** (1/12) - 1
        price[0] = float(s["price0"])
        for t in range(1, T):
            price[t] = price[t-1] * (1 + r_m)

        # Initial holdings (pre-owned, no cash impact)
        units[0] = float(s["initial_units"])

        # One-shot buy at start (cash impact)
        buy0 = s.get("buy_at_start")
        if buy0:
            if "amount" in buy0 and buy0["amount"] > 0:
                amt = float(buy0["amount"])
                add_u = amt / price[0]
                units[0] += add_u
                cash_out[0] += amt
                if s.get("events_level") in ("major", "all"):
                    events.append(Event(ctx.t_index[0], "buy_start",
                                        f"ETF buy at start: €{amt:,.2f}",
                                        {"amount": amt, "units": add_u, "price": price[0]}))
            elif "units" in buy0 and buy0["units"] > 0:
                u = float(buy0["units"])
                amt = u * price[0]
                units[0] += u
                cash_out[0] += amt
                if s.get("events_level") in ("major", "all"):
                    events.append(Event(ctx.t_index[0], "buy_start",
                                        f"ETF buy at start: {u:,.6f}u",
                                        {"amount": amt, "units": u, "price": price[0]}))

        # Extract configuration for monthly loop
        divm = float(s["div_yield_pa"]) / 12.0
        reinv = bool(s["reinvest_dividends"])
        round_to = s.get("round_units_to")
        dca = s.get("dca")
        ev_lvl = s.get("events_level")

        # Monthly loop for dividends & DCA
        for t in range(T):
            # Carry forward units
            if t > 0:
                units[t] = units[t-1]

            # Dividends BEFORE DCA (based on units at start of month)
            if divm > 0:
                dv = units[t] * price[t] * divm
                if reinv and dv > 0:
                    add_u = dv / price[t]
                    units[t] += add_u
                    if ev_lvl in ("major", "all"):
                        events.append(Event(ctx.t_index[t], "div_reinvest",
                                            f"Dividends reinvested: €{dv:,.2f}",
                                            {"amount": dv, "units": add_u, "price": price[t]}))
                else:
                    cash_in[t] += dv
                    if ev_lvl in ("major", "all") and dv > 0:
                        events.append(Event(ctx.t_index[t], "div_cash",
                                            f"Dividends to cash: €{dv:,.2f}",
                                            {"amount": dv, "price": price[t]}))

            # DCA AFTER dividends
            if dca is not None:
                start_off = int(dca.get("start_offset_m", 0))
                months = dca.get("months", None)
                m_rel = t - start_off
                if m_rel >= 0 and (months is None or m_rel < int(months)):
                    if dca["mode"] == "amount":
                        step_blocks = max(0, m_rel // 12)
                        amt = float(dca["amount"]) * ((1 + float(dca.get("annual_step_pct", 0.0))) ** step_blocks)
                        if amt > 0:
                            add_u = amt / price[t]
                            units[t] += add_u
                            cash_out[t] += amt
                            if ev_lvl == "all":
                                events.append(Event(ctx.t_index[t], "dca_amount",
                                                    f"DCA (amount): €{amt:,.2f}",
                                                    {"amount": amt, "units": add_u, "price": price[t]}))
                    else:  # units mode
                        u = float(dca["units"])
                        if u > 0:
                            amt = u * price[t]  # Use current month's price
                            units[t] += u
                            cash_out[t] += amt
                            if ev_lvl == "all":
                                events.append(Event(ctx.t_index[t], "dca_units",
                                                    f"DCA (units): {u:,.6f}u",
                                                    {"amount": amt, "units": u, "price": price[t]}))

            # Round units after all operations for the month
            if round_to is not None:
                units[t] = np.round(units[t], int(round_to))

        # Calculate final asset values
        asset_value = units * price

        return BrickOutput(
            cash_in=cash_in,
            cash_out=cash_out,
            asset_value=asset_value,
            debt_balance=np.zeros(T),
            events=events
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
            auto_from = (brick.links or {}).get("auto_principal_from")
            assert auto_from in ctx.registry, "auto_principal_from link missing or invalid"
            prop: ABrick = ctx.registry[auto_from]  # type: ignore
            
            price = float(prop.spec["price"])
            down = float(prop.spec.get("down_payment", 0.0))
            fees_pct = float(prop.spec.get("fees_pct", 0.0))
            fees = price * fees_pct
            
            # Handle fees financing
            finance_fees = bool(prop.spec.get("finance_fees", False))
            fees_fin_pct = float(prop.spec.get("fees_financed_pct", 1.0 if finance_fees else 0.0))
            fees_fin_pct = max(0.0, min(1.0, fees_fin_pct))  # Clamp to [0,1]
            fees_financed = fees * fees_fin_pct
            
            # Calculate principal: price - down_payment + financed_fees
            principal = price - down + fees_financed
            brick.spec["principal"] = principal
            
            # Store derived values for logging/validation
            brick.spec["_derived"] = {
                "price": price,
                "down_payment": down,
                "fees": fees,
                "fees_financed": fees_financed
            }
        
        # Validate all required parameters
        required_params = ["rate_pa", "term_months", "principal"]
        for param in required_params: 
            assert param in brick.spec, f"Missing required parameter: {param}"
        
        # Set default first payment offset (1 month is standard)
        brick.spec.setdefault("first_payment_offset", 1)

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
        # Use full term for payment calculation, but limit payments to simulation period

        # Initial loan drawdown at t=0
        cash_in[0] += principal
        debt[0] = principal

        # Calculate monthly payment using annuity formula
        r_m = rate_pa / 12.0
        offset = int(brick.spec["first_payment_offset"])
        
        # Use full term for payment calculation
        if r_m > 0:
            A = principal * (r_m * (1 + r_m) ** n_total) / ((1 + r_m) ** n_total - 1)
        else:
            A = principal / n_total  # Handle zero interest rate case
        
        # Carry forward debt unchanged until first payment
        for t in range(1, min(offset, T)):
            debt[t] = debt[t-1]
        
        # Calculate payment schedule starting from offset
        n_sched = min(n_total, max(0, T - offset))
        for k in range(n_sched):
            t = offset + k
            if t >= T:
                break
                
            prev_debt = debt[t-1] if t > 0 else principal
            if prev_debt > 0:
                interest = prev_debt * r_m
                principal_pay = min(A - interest, prev_debt)
                cash_out[t] = interest + principal_pay
                debt[t] = max(prev_debt - principal_pay, 0.0)
            else:
                debt[t] = 0.0

        # Create time-stamped events
        events = [
            Event(ctx.t_index[0], "loan_draw", f"Mortgage drawdown: €{principal:,.2f}", 
                  {"principal": principal})
        ]
        
        # Add derived info if available
        if "_derived" in brick.spec:
            derived = brick.spec["_derived"]
            events.append(Event(ctx.t_index[0], "loan_details", 
                                f"Price: €{derived['price']:,.2f}, Down: €{derived['down_payment']:,.2f}, Fees financed: €{derived['fees_financed']:,.2f}",
                                derived))

        return BrickOutput(
            cash_in=cash_in, 
            cash_out=cash_out,
            asset_value=np.zeros(T), 
            debt_balance=debt,
            events=events
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
            events=[Event(ctx.t_index[0], "transfer", f"Lump sum transfer: €{cash_in[0]:,.2f}", 
                          {"amount": cash_in[0]})]
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
            events=[]  # No events for regular income flows
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
            events=[]  # No events for regular expense flows
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
    ValuationRegistry[K.A_CASH]      = ValuationCash()
    ValuationRegistry[K.A_PROPERTY]  = ValuationPropertyDiscrete()
    ValuationRegistry[K.A_INV_ETF]   = ValuationETFUnitized()
    
    # Register liability schedule strategies
    ScheduleRegistry[K.L_MORT_ANN]   = ScheduleMortgageAnnuity()
    
    # Register cash flow strategies
    FlowRegistry[K.F_TRANSFER]       = FlowTransferLumpSum()
    FlowRegistry[K.F_INCOME]         = FlowIncomeFixed()
    FlowRegistry[K.F_EXP_LIVING]     = FlowExpenseFixed()


# Automatically register default strategies when module is imported
register_defaults()
