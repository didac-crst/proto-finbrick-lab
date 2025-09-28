"""
FinScenLab Core Module

This module contains the core architecture for the FinScenLab financial scenario modeling system.
It implements a strategy-driven brick architecture where financial instruments are modeled as
self-contained "bricks" that can be composed together to create complex financial scenarios.

Key Components:
- FinBrickABC: Abstract base class for all financial instruments
- ABrick/LBrick/FBrick: Concrete classes for Assets, Liabilities, and Flows
- Strategy Interfaces: Protocols for valuation, scheduling, and flow strategies
- Registry System: Maps kind strings to strategy implementations
- Scenario Engine: Orchestrates simulation and cash flow routing

Architecture Benefits:
- Extensible: Add new behaviors by registering strategies, no class changes
- Modular: Each brick is self-contained with clear interfaces
- Composable: Link bricks together for complex scenarios
- Explicit: Cash flows are explicitly routed, no hidden assumptions
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypedDict, Dict, List, Optional, Callable, NamedTuple, Any
from datetime import date
import numpy as np
import pandas as pd
import copy

# ---------- time utilities ----------

def month_range(start: date, months: int) -> np.ndarray:
    """
    Generate a range of monthly dates starting from a given date.
    
    This utility function creates a numpy array of datetime64 objects representing
    consecutive months, which is used throughout the system for time-based simulations.
    
    Args:
        start: The starting date for the range
        months: Number of months to generate
        
    Returns:
        A numpy array of datetime64 objects representing monthly intervals
        
    Example:
        >>> month_range(date(2026, 1, 1), 12)
        array(['2026-01', '2026-02', '2026-03', ..., '2026-12'], dtype='datetime64[M]')
    """
    s = np.datetime64(start, 'M')
    return s + np.arange(months).astype('timedelta64[M]')

# ---------- event and output structures ----------

class Event(NamedTuple):
    """
    Time-stamped event record for financial brick simulations.
    
    Events provide a structured way to track important occurrences during
    simulation, with timestamps aligned to the simulation time index.
    
    Attributes:
        t: The month when the event occurred (np.datetime64[M])
        kind: Event type identifier (e.g., 'purchase', 'fees', 'loan_draw', 'payment')
        message: Human-readable description of the event
        meta: Optional dictionary with additional event metadata
    """
    t: np.datetime64          # Month when event occurred
    kind: str                 # Event type identifier
    message: str              # Human-readable description
    meta: Optional[Dict[str, Any]] = None  # Additional metadata

class BrickOutput(TypedDict):
    """
    Standard output structure for all financial brick simulations.
    
    This TypedDict defines the common interface that all brick strategies must return.
    It provides a consistent structure for cash flows, asset values, debt balances,
    and event tracking across all types of financial instruments.
    
    Attributes:
        cash_in: Monthly cash inflows (always >= 0)
        cash_out: Monthly cash outflows (always >= 0)  
        asset_value: Monthly asset valuation (0 for non-assets)
        debt_balance: Monthly debt balance (0 for non-liabilities)
        events: List of time-stamped events describing key occurrences
        
    Note:
        All numpy arrays have the same length corresponding to the simulation period.
        Cash flows are always positive values - the direction is implicit in the field name.
        Events are time-stamped and can be used to build a simulation ledger.
    """
    cash_in: np.ndarray        # Monthly cash inflows (>=0)
    cash_out: np.ndarray       # Monthly cash outflows (>=0)
    asset_value: np.ndarray    # Monthly asset value (0 if not an asset)
    debt_balance: np.ndarray   # Monthly debt balance (0 if not a liability)
    events: List[Event]        # Time-stamped events describing key occurrences

# ---------- simulation context ----------

@dataclass
class ScenarioContext:
    """
    Context object passed to all brick strategies during simulation.
    
    This dataclass contains the shared context information that all strategies
    need access to during the simulation process, including the time index,
    currency, and registry of all bricks in the scenario.
    
    Attributes:
        t_index: Array of monthly datetime64 objects representing the simulation timeline
        currency: Base currency for the scenario (e.g., 'EUR', 'USD')
        registry: Dictionary mapping brick IDs to brick instances for cross-references
        
    Note:
        The registry allows bricks to reference other bricks through the links mechanism,
        enabling complex interdependencies like mortgages that auto-calculate from property values.
    """
    t_index: np.ndarray
    currency: str
    registry: Dict[str, "FinBrickABC"]  # id -> brick mapping

# ---------- strategy interfaces (protocols) ----------

class IValuationStrategy(Protocol):
    """
    Protocol for asset valuation strategies.
    
    This protocol defines the interface that all asset valuation strategies must implement.
    Asset strategies handle the valuation and cash flow generation for assets like cash,
    property, investments, etc.
    
    Methods:
        prepare: Initialize the strategy with brick parameters and context
        simulate: Generate the simulation results for the asset
    """
    
    def prepare(self, brick: "ABrick", ctx: ScenarioContext) -> None:
        """
        Prepare the strategy for simulation.
        
        This method is called once before simulation begins to validate parameters,
        perform any necessary calculations, and set up the strategy state.
        
        Args:
            brick: The asset brick being simulated
            ctx: The simulation context containing time index and registry
        """
        ...
    
    def simulate(self, brick: "ABrick", ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the asset over the entire time period.
        
        This method generates the complete simulation results for the asset,
        including cash flows, asset values, and any relevant events.
        
        Args:
            brick: The asset brick being simulated
            ctx: The simulation context containing time index and registry
            
        Returns:
            BrickOutput containing cash flows, asset values, and events
        """
        ...


class IScheduleStrategy(Protocol):
    """
    Protocol for liability scheduling strategies.
    
    This protocol defines the interface that all liability scheduling strategies must implement.
    Liability strategies handle the payment schedules and balance tracking for debts like
    mortgages, loans, credit cards, etc.
    
    Methods:
        prepare: Initialize the strategy with brick parameters and context
        simulate: Generate the simulation results for the liability
    """
    
    def prepare(self, brick: "LBrick", ctx: ScenarioContext) -> None:
        """
        Prepare the strategy for simulation.
        
        This method is called once before simulation begins to validate parameters,
        perform any necessary calculations, and set up the strategy state.
        
        Args:
            brick: The liability brick being simulated
            ctx: The simulation context containing time index and registry
        """
        ...
    
    def simulate(self, brick: "LBrick", ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the liability over the entire time period.
        
        This method generates the complete simulation results for the liability,
        including payment schedules, debt balances, and any relevant events.
        
        Args:
            brick: The liability brick being simulated
            ctx: The simulation context containing time index and registry
            
        Returns:
            BrickOutput containing cash flows, debt balances, and events
        """
        ...


class IFlowStrategy(Protocol):
    """
    Protocol for cash flow strategies.
    
    This protocol defines the interface that all cash flow strategies must implement.
    Flow strategies handle the generation of cash flows for income, expenses,
    transfers, and other cash flow events.
    
    Methods:
        prepare: Initialize the strategy with brick parameters and context
        simulate: Generate the simulation results for the flow
    """
    
    def prepare(self, brick: "FBrick", ctx: ScenarioContext) -> None:
        """
        Prepare the strategy for simulation.
        
        This method is called once before simulation begins to validate parameters,
        perform any necessary calculations, and set up the strategy state.
        
        Args:
            brick: The flow brick being simulated
            ctx: The simulation context containing time index and registry
        """
        ...
    
    def simulate(self, brick: "FBrick", ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the flow over the entire time period.
        
        This method generates the complete simulation results for the flow,
        including cash inflows/outflows and any relevant events.
        
        Args:
            brick: The flow brick being simulated
            ctx: The simulation context containing time index and registry
            
        Returns:
            BrickOutput containing cash flows and events
        """
        ...

# ---------- abstract base class ----------

@dataclass
class FinBrickABC:
    """
    Abstract base class for all financial instruments in FinScenLab.
    
    This class defines the common interface and structure for all financial bricks.
    It serves as the foundation for the strategy pattern implementation, where
    the actual behavior is determined by the 'kind' discriminator and associated
    strategy objects.
    
    Attributes:
        id: Unique identifier for the brick within a scenario
        name: Human-readable name for the brick
        kind: Dot-separated string discriminator (e.g., 'a.cash', 'l.mortgage.annuity')
        currency: Currency code for the brick (default: 'EUR')
        spec: Dictionary containing strategy-specific parameters
        links: Dictionary for referencing other bricks (e.g., {'auto_principal_from': 'house_id'})
        family: Brick family type ('a' for assets, 'l' for liabilities, 'f' for flows)
        start_date: Optional start date for the brick (default: None = starts at scenario start)
        
    Note:
        The 'family' attribute is automatically set by subclasses and should not be
        specified manually when creating brick instances.
        
        The 'start_date' allows bricks to activate at specific times during the simulation,
        enabling scenarios like buying a house in 2028 or starting a new job in 2025.
    """
    id: str
    name: str
    kind: str             # Dot-separated string discriminator
    currency: str = "EUR"
    spec: dict = None     # Strategy-specific parameters
    links: dict = None    # References to other bricks
    family: str = None    # 'a' | 'l' | 'f' - set automatically in subclasses
    start_date: Optional[date] = None  # When this brick becomes active
    end_date: Optional[date] = None    # When this brick becomes inactive
    duration_m: Optional[int] = None   # Duration in months (alternative to end_date)

    def prepare(self, ctx: ScenarioContext) -> None:
        """
        Prepare the brick for simulation.
        
        This method is called once before simulation begins to validate parameters
        and perform any necessary setup. The actual implementation is delegated
        to the appropriate strategy object.
        
        Args:
            ctx: The simulation context containing time index and registry
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError
    
    def simulate(self, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the brick over the entire time period.
        
        This method generates the complete simulation results for the brick.
        The actual implementation is delegated to the appropriate strategy object.
        
        Args:
            ctx: The simulation context containing time index and registry
            
        Returns:
            BrickOutput containing cash flows, values, and events
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError

# ---------- concrete brick implementations ----------

@dataclass
class ABrick(FinBrickABC):
    """
    Asset brick for representing financial assets.
    
    This class represents assets such as cash accounts, real estate, investments,
    vehicles, and other valuable items. The actual behavior is determined by
    the valuation strategy associated with the brick's 'kind' discriminator.
    
    Attributes:
        valuation: The valuation strategy object (set automatically by registry)
        
    Examples:
        Cash account: kind='a.cash'
        Real estate: kind='a.property'  
        ETF investment: kind='a.invest.etf'
    """
    valuation: IValuationStrategy = None
    
    def __post_init__(self):
        """Set the family type to 'a' for assets."""
        self.family = 'a'
    
    def prepare(self, ctx: ScenarioContext) -> None:
        """
        Prepare the asset for simulation.
        
        Delegates to the associated valuation strategy's prepare method.
        
        Args:
            ctx: The simulation context containing time index and registry
        """
        self.valuation.prepare(self, ctx)
    
    def simulate(self, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the asset over the time period.
        
        Delegates to the associated valuation strategy's simulate method.
        
        Args:
            ctx: The simulation context containing time index and registry
            
        Returns:
            BrickOutput containing asset values, cash flows, and events
        """
        return self.valuation.simulate(self, ctx)


@dataclass
class LBrick(FinBrickABC):
    """
    Liability brick for representing financial debts and obligations.
    
    This class represents liabilities such as mortgages, loans, credit cards,
    and other debt instruments. The actual behavior is determined by
    the schedule strategy associated with the brick's 'kind' discriminator.
    
    Attributes:
        schedule: The schedule strategy object (set automatically by registry)
        
    Examples:
        Mortgage: kind='l.mortgage.annuity'
        Personal loan: kind='l.loan.personal'
        Credit card: kind='l.credit.card'
    """
    schedule: IScheduleStrategy = None
    
    def __post_init__(self):
        """Set the family type to 'l' for liabilities."""
        self.family = 'l'
    
    def prepare(self, ctx: ScenarioContext) -> None:
        """
        Prepare the liability for simulation.
        
        Delegates to the associated schedule strategy's prepare method.
        
        Args:
            ctx: The simulation context containing time index and registry
        """
        self.schedule.prepare(self, ctx)
    
    def simulate(self, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the liability over the time period.
        
        Delegates to the associated schedule strategy's simulate method.
        
        Args:
            ctx: The simulation context containing time index and registry
            
        Returns:
            BrickOutput containing debt balances, payment flows, and events
        """
        return self.schedule.simulate(self, ctx)


@dataclass
class FBrick(FinBrickABC):
    """
    Flow brick for representing cash flow events.
    
    This class represents cash flow events such as income, expenses, transfers,
    and other monetary flows. The actual behavior is determined by
    the flow strategy associated with the brick's 'kind' discriminator.
    
    Attributes:
        flow: The flow strategy object (set automatically by registry)
        
    Examples:
        Salary income: kind='f.income.salary'
        Living expenses: kind='f.expense.living'
        Lump sum transfer: kind='f.transfer.lumpsum'
    """
    flow: IFlowStrategy = None
    
    def __post_init__(self):
        """Set the family type to 'f' for flows."""
        self.family = 'f'
    
    def prepare(self, ctx: ScenarioContext) -> None:
        """
        Prepare the flow for simulation.
        
        Delegates to the associated flow strategy's prepare method.
        
        Args:
            ctx: The simulation context containing time index and registry
        """
        self.flow.prepare(self, ctx)
    
    def simulate(self, ctx: ScenarioContext) -> BrickOutput:
        """
        Simulate the flow over the time period.
        
        Delegates to the associated flow strategy's simulate method.
        
        Args:
            ctx: The simulation context containing time index and registry
            
        Returns:
            BrickOutput containing cash flows and events
        """
        return self.flow.simulate(self, ctx)

# ---------- strategy registry system ----------

# Global registries mapping kind strings to strategy implementations
ValuationRegistry: Dict[str, IValuationStrategy] = {}
ScheduleRegistry:  Dict[str, IScheduleStrategy]  = {}
FlowRegistry:      Dict[str, IFlowStrategy]      = {}


def wire_strategies(bricks: List[FinBrickABC]) -> None:
    """
    Attach the correct strategy object to each brick based on its kind discriminator.
    
    This function implements the core of the strategy pattern by looking up the
    appropriate strategy implementation for each brick's 'kind' and attaching
    it to the brick. This allows the same brick classes to exhibit different
    behaviors based on their kind discriminator.
    
    Args:
        bricks: List of all bricks in the scenario
        
    Raises:
        AssertionError: If a brick's kind is not found in the appropriate registry
        
    Note:
        This function modifies the bricks in-place by setting their strategy
        attributes (valuation, schedule, or flow). It also creates deep copies
        of brick specs to ensure per-run isolation and prevent accidental
        carry-over between simulation runs.
    """
    for b in bricks:
        # Create deep copy of spec for per-run isolation
        b.spec = copy.deepcopy(b.spec or {})
        
        if isinstance(b, ABrick):
            assert b.kind in ValuationRegistry, f"Unknown asset kind: {b.kind}"
            b.valuation = ValuationRegistry[b.kind]
        elif isinstance(b, LBrick):
            assert b.kind in ScheduleRegistry, f"Unknown liability kind: {b.kind}"
            b.schedule = ScheduleRegistry[b.kind]
        elif isinstance(b, FBrick):
            assert b.kind in FlowRegistry, f"Unknown flow kind: {b.kind}"
            b.flow = FlowRegistry[b.kind]

# ---------- scenario engine ----------

@dataclass
class Scenario:
    """
    Scenario engine for orchestrating financial simulations.
    
    This class represents a complete financial scenario containing multiple
    financial bricks. It orchestrates the simulation process by:
    1. Wiring strategies to bricks based on their kind discriminators
    2. Preparing all bricks for simulation
    3. Simulating all bricks in the correct order
    4. Routing cash flows to the designated cash account
    5. Aggregating results into summary statistics
    
    Attributes:
        id: Unique identifier for the scenario
        name: Human-readable name for the scenario
        bricks: List of all financial bricks in the scenario
        currency: Base currency for the scenario (default: 'EUR')
        
    Note:
        The scenario expects exactly one cash account brick (kind='a.cash')
        to receive all routed cash flows from other bricks.
    """
    id: str
    name: str
    bricks: List[FinBrickABC]
    currency: str = "EUR"

    def run(self, start: date, months: int, include_cash: bool = True) -> dict:
        """
        Run the complete financial scenario simulation.
        
        This method orchestrates the entire simulation process:
        1. Creates the time index for the simulation period
        2. Wires strategies to bricks based on their kind discriminators
        3. Prepares all bricks for simulation
        4. Simulates all non-cash bricks and routes their cash flows
        5. Simulates the cash account with all routed flows
        6. Aggregates results into summary statistics
        
        Args:
            start: The starting date for the simulation
            months: Number of months to simulate
            
        Returns:
            Dictionary containing:
                - 'outputs': Dict mapping brick IDs to their individual BrickOutput results
                - 'totals': DataFrame with aggregated monthly totals (cash flows, assets, debt, equity)
                
        Raises:
            AssertionError: If there is not exactly one cash account brick (kind='a.cash')
            
        Note:
            The simulation assumes exactly one cash account to receive all routed flows.
            Cash flows from all other bricks are automatically routed to this account.
        """
        # Create time index for the simulation period
        t_index = month_range(start, months)
        ctx = ScenarioContext(t_index=t_index, currency=self.currency,
                              registry={b.id: b for b in self.bricks})

        # Wire strategies to bricks based on their kind discriminators
        wire_strategies(self.bricks)

        # Prepare all bricks for simulation (validate parameters, setup state)
        for b in self.bricks: 
            b.prepare(ctx)

        # Simulate all non-cash bricks first, then route cash flows to cash account
        outputs: Dict[str, BrickOutput] = {}
        cash_ids = [b.id for b in self.bricks if isinstance(b, ABrick) and b.kind == "a.cash"]
        assert len(cash_ids) == 1, "Scenario expects exactly one cash account brick (kind='a.cash')"
        cash_id = cash_ids[0]

        # Accumulate cash flows from all non-cash bricks
        routed_in  = np.zeros(len(t_index))
        routed_out = np.zeros(len(t_index))

        for b in self.bricks:
            if b.id == cash_id: 
                continue  # Skip cash account for now
            
            # Handle delayed brick activation
            if b.start_date is not None:
                # Find the start index for this brick
                start_idx = self._find_start_index(b.start_date, t_index)
                if start_idx is None:
                    # Brick starts after simulation period, skip it
                    continue
            else:
                start_idx = 0  # Brick starts at beginning of simulation
            
            # Create a modified context for this brick with delayed start
            brick_ctx = self._create_delayed_context(ctx, start_idx)
            
            out = b.simulate(brick_ctx)
            
            # Shift the output arrays to the correct time positions
            if start_idx > 0:
                shifted_out = self._shift_output(out, start_idx, len(t_index))
                outputs[b.id] = shifted_out
            else:
                outputs[b.id] = out
            
            # Apply equity-neutral activation window mask
            mask = active_mask(t_index, b.start_date, b.end_date, b.duration_m)
            _apply_window_equity_neutral(outputs[b.id], mask)
            
            # Add window end event if brick has an end
            if b.end_date is not None or b.duration_m is not None:
                end_idx = np.where(mask)[0]
                if len(end_idx) > 0:
                    last_active_idx = end_idx[-1]
                    outputs[b.id]["events"].append(
                        Event(t_index[last_active_idx], "window_end", 
                              f"Brick '{b.name}' window ended", {"brick_id": b.id})
                    )
            
            # Accumulate cash flows for routing
            routed_in  += outputs[b.id]["cash_in"]
            routed_out += outputs[b.id]["cash_out"]

        # Route accumulated cash flows to the cash account
        cash_brick = ctx.registry[cash_id]
        cash_brick.spec.setdefault("external_in",  np.zeros(len(t_index)))
        cash_brick.spec.setdefault("external_out", np.zeros(len(t_index)))
        cash_brick.spec["external_in"]  = routed_in
        cash_brick.spec["external_out"] = routed_out

        # Simulate the cash account with all routed flows
        outputs[cash_id] = cash_brick.simulate(ctx)

        # Aggregate results into summary statistics
        cash_in_tot   = sum(o["cash_in"] for o in outputs.values())
        cash_out_tot  = sum(o["cash_out"] for o in outputs.values())
        assets_tot    = sum(o["asset_value"] for o in outputs.values())
        debt_tot      = sum(o["debt_balance"] for o in outputs.values())
        net_cf        = cash_in_tot - cash_out_tot
        equity        = assets_tot - debt_tot

        # Create summary DataFrame with monthly totals
        totals = pd.DataFrame({
            "t": t_index, 
            "cash_in": cash_in_tot, 
            "cash_out": cash_out_tot,
            "net_cf": net_cf, 
            "assets": assets_tot, 
            "debt": debt_tot, 
            "equity": equity
        }).set_index("t")
        
        # Add cash column if requested
        if include_cash:
            from .kinds import K
            cash_series = None
            for b in self.bricks:
                if isinstance(b, ABrick) and b.kind == K.A_CASH:
                    s = outputs[b.id]["asset_value"]
                    cash_series = s if cash_series is None else (cash_series + s)
            if cash_series is not None:
                totals["cash"] = cash_series
        
        return {"outputs": outputs, "totals": totals, "_scenario_bricks": self.bricks}
    
    def _find_start_index(self, start_date: date, t_index: np.ndarray) -> Optional[int]:
        """
        Find the index in t_index that corresponds to the start_date.
        
        Args:
            start_date: The date when the brick should start
            t_index: The time index array
            
        Returns:
            The index where the brick should start, or None if after simulation period
        """
        start_datetime64 = np.datetime64(start_date, 'M')
        
        # Find the first index where t_index >= start_date
        for i, t in enumerate(t_index):
            if t >= start_datetime64:
                return i
        
        return None  # start_date is after simulation period
    
    def _create_delayed_context(self, ctx: ScenarioContext, start_idx: int) -> ScenarioContext:
        """
        Create a modified context for a brick that starts at a delayed time.
        
        Args:
            ctx: The original simulation context
            start_idx: The index where the brick starts
            
        Returns:
            A new context with time index starting from start_idx
        """
        # Create a new time index starting from the brick's start time
        new_t_index = ctx.t_index[start_idx:]
        
        return ScenarioContext(
            t_index=new_t_index,
            currency=ctx.currency,
            registry=ctx.registry
        )
    
    def _shift_output(self, output: BrickOutput, start_idx: int, total_length: int) -> BrickOutput:
        """
        Shift a brick's output to start at the correct time index.
        
        Args:
            output: The brick's output
            start_idx: The index where the brick starts
            total_length: The total length of the simulation
            
        Returns:
            A new BrickOutput with arrays padded with zeros at the beginning
        """
        # Create arrays of the full simulation length
        full_cash_in = np.zeros(total_length)
        full_cash_out = np.zeros(total_length)
        full_asset_value = np.zeros(total_length)
        full_debt_balance = np.zeros(total_length)
        
        # Place the brick's output at the correct time positions
        brick_length = len(output["cash_in"])
        end_idx = min(start_idx + brick_length, total_length)
        actual_length = end_idx - start_idx
        
        full_cash_in[start_idx:end_idx] = output["cash_in"][:actual_length]
        full_cash_out[start_idx:end_idx] = output["cash_out"][:actual_length]
        full_asset_value[start_idx:end_idx] = output["asset_value"][:actual_length]
        full_debt_balance[start_idx:end_idx] = output["debt_balance"][:actual_length]
        
        return BrickOutput(
            cash_in=full_cash_in,
            cash_out=full_cash_out,
            asset_value=full_asset_value,
            debt_balance=full_debt_balance,
            events=output["events"]  # Events don't need shifting
        )

# ---------- activation window utilities ----------

def active_mask(t_index: np.ndarray, start_date: Optional[date], end_date: Optional[date], duration_m: Optional[int]) -> np.ndarray:
    """
    Create a boolean mask indicating when a brick is active.
    
    Args:
        t_index: Time index array (np.datetime64[M])
        start_date: When the brick becomes active (None = scenario start)
        end_date: When the brick becomes inactive (None = scenario end)
        duration_m: Duration in months (alternative to end_date)
        
    Returns:
        Boolean array where True indicates the brick is active
        
    Note:
        - duration_m includes the start month (duration_m=12 means 12 months including start_date)
        - end_date takes precedence over duration_m if both are provided
        - Inactive periods are masked with False (will be zeroed in outputs)
    """
    # Normalize start date
    if start_date is not None:
        start = np.datetime64(start_date, 'M')
    else:
        start = t_index[0]
    
    # Determine end date
    if end_date is not None:
        end = np.datetime64(end_date, 'M')
        # Warn if both end_date and duration_m are provided
        if duration_m is not None:
            print(f"[WARN] Both end_date and duration_m provided; using end_date {end_date}")
    elif duration_m is not None:
        if duration_m < 1:
            raise ValueError("duration_m must be >= 1")
        end = start + np.timedelta64(duration_m - 1, 'M')  # inclusive
    else:
        end = t_index[-1]
    
    return (t_index >= start) & (t_index <= end)

def _apply_window_equity_neutral(out, mask):
    """
    Apply activation window mask in an equity-neutral way.
    
    Only flows (cash_in, cash_out) are masked to zero outside the window.
    Stock series (asset_value, debt_balance) are NOT zeroed - they carry forward
    the last active value unless explicitly set by terminal disposal/payoff events.
    
    Args:
        out: BrickOutput dictionary with cash_in, cash_out, asset_value, debt_balance
        mask: Boolean array indicating when the brick is active
        
    Note:
        This preserves the accounting identity: equity only changes via explicit flows.
        Terminal disposal/payoff events must book the appropriate cash legs at t_stop.
    """
    import numpy as np
    
    # Mask flows to zero outside the window
    out["cash_in"]  = np.where(mask, out["cash_in"],  0.0)
    out["cash_out"] = np.where(mask, out["cash_out"], 0.0)
    
    # Do NOT touch stocks here; terminal actions set them explicitly at t_stop

def resolve_prepayments_to_month_idx(t_index: np.ndarray, prepayments: list, mortgage_start_date: date) -> dict:
    """
    Resolve prepayment directives to month indices.
    
    Args:
        t_index: Time index array (np.datetime64[M])
        prepayments: List of prepayment directives
        mortgage_start_date: Start date of the mortgage for relative calculations
        
    Returns:
        Dictionary mapping month index to prepayment amount
        
    Note:
        Supports both absolute dates ("t": "YYYY-MM") and periodic schedules
        ({"every": "year", "month": 12, "amount": 5000})
    """
    prepay_map = {}
    
    for prepay in prepayments:
        if "t" in prepay:
            # Absolute date specification
            prepay_date = np.datetime64(prepay["t"], 'M')
            month_idx = np.where(t_index == prepay_date)[0]
            if len(month_idx) > 0:
                idx = month_idx[0]
                if "amount" in prepay:
                    prepay_map[idx] = float(prepay["amount"])
                elif "pct_balance" in prepay:
                    prepay_map[idx] = ("pct", float(prepay["pct_balance"]), float(prepay.get("cap", float('inf'))))
        elif "every" in prepay:
            # Periodic specification
            if prepay["every"] == "year":
                start_year = prepay.get("start_year", mortgage_start_date.year)
                end_year = prepay.get("end_year", start_year + 10)
                month = prepay["month"]
                
                for year in range(start_year, end_year + 1):
                    prepay_date = np.datetime64(f"{year}-{month:02d}", 'M')
                    month_idx = np.where(t_index == prepay_date)[0]
                    if len(month_idx) > 0:
                        idx = month_idx[0]
                        if "amount" in prepay:
                            prepay_map[idx] = float(prepay["amount"])
                        elif "pct_balance" in prepay:
                            prepay_map[idx] = ("pct", float(prepay["pct_balance"]), float(prepay.get("cap", float('inf'))))
    
    return prepay_map

# ---------- validation utilities ----------

def validate_run(res: dict, bricks=None, mode: str = "raise", tol: float = 1e-6) -> None:
    """
    Validate simulation results against key financial invariants.
    
    This function performs several consistency checks on the simulation results
    to catch potential bugs or modeling errors. It can either raise exceptions
    or issue warnings based on the mode parameter.
    
    Args:
        res: The results dictionary returned by Scenario.run()
        mode: Validation mode - 'raise' to raise AssertionError on failures,
              'warn' to print warnings instead
        tol: Numerical tolerance for floating-point comparisons
              
    Raises:
        AssertionError: If validation fails and mode='raise'
        
    Note:
        The validation checks include:
        - Equity identity: equity = assets - debt
        - Debt monotonicity: debt should not increase after initial draws
        - Cash flow consistency: net_cf = cash_in - cash_out
    """
    totals = res["totals"]
    outputs = res["outputs"]
    
    # 1) Identity checks
    fails = []
    
    # Equity identity: equity = assets - debt
    if not np.allclose(totals["equity"].values, (totals["assets"] - totals["debt"]).values, atol=tol):
        fails.append("equity != assets - debt")
    
    # Cash flow consistency: net_cf = cash_in - cash_out
    if not np.allclose(totals["net_cf"].values, (totals["cash_in"] - totals["cash_out"]).values, atol=tol):
        fails.append("net_cf != cash_in - cash_out")
    
    # Debt monotonicity: debt should not increase after initial draws
    debt = totals["debt"].values
    if len(debt) > 1 and not np.all(np.diff(debt[1:]) <= tol):
        fails.append("debt increased after t0")
    
    # 4) Purchase settlement validation (if applicable)
    purchase_ok = True
    purchase_messages = []
    
    # Check for property purchases and their settlement
    for brick_id, output in res["outputs"].items():
        # Look for property bricks that have cash_out at t=0
        if output["cash_out"][0] > 1e-6:  # Has cash outflow at t=0
            # This might be a property purchase - check if it's reasonable
            cash_out_t0 = output["cash_out"][0]
            
            # Find the corresponding brick to get its spec
            brick = None
            for b in res.get("_scenario_bricks", []):
                if b.id == brick_id:
                    brick = b
                    break
            
            if brick and hasattr(brick, 'spec') and "price" in brick.spec:
                price = float(brick.spec["price"])
                fees_pct = float(brick.spec.get("fees_pct", 0.0))
                fees = price * fees_pct
                fees_fin_pct = float(brick.spec.get("fees_financed_pct", 1.0 if brick.spec.get("finance_fees") else 0.0))
                fees_cash = fees * (1.0 - fees_fin_pct)
                expected_cash_out = price + fees_cash
                
                if abs(cash_out_t0 - expected_cash_out) > tol:
                    purchase_ok = False
                    purchase_messages.append(f"{brick_id} cash_out[t0] = €{cash_out_t0:,.2f}, expected €{expected_cash_out:,.2f}")
    
    if not purchase_ok:
        fails.append("purchase settlement mismatch: " + "; ".join(purchase_messages))
    
    # 5) Liquidity constraints (only if we have bricks)
    if bricks is not None:
        from .kinds import K
        for b in bricks:
            if isinstance(b, ABrick) and b.kind == K.A_CASH:
                bal = outputs[b.id]["asset_value"]
                overdraft = float((b.spec or {}).get("overdraft_limit", 0.0))
                minbuf = float((b.spec or {}).get("min_buffer", 0.0))
                
                # Overdraft breach
                if (bal < -overdraft - tol).any():
                    t_idx = int(np.where(bal < -overdraft - tol)[0][0])
                    amt = float(bal[t_idx])
                    msg = (f"Liquidity breach: cash '{b.id}' = {amt:,.2f} < overdraft_limit {overdraft:,.2f}. "
                           f"Suggest: top-up ≥ {abs(amt+overdraft):,.2f} or reduce t₀ outflows / finance fees.")
                    fails.append(msg)
                
                # Buffer breach
                if (bal < minbuf - tol).any():
                    t_idx = int(np.where(bal < minbuf - tol)[0][0])
                    amt = float(bal[t_idx])
                    msg = (f"Buffer breach: cash '{b.id}' = {amt:,.2f} < min_buffer {minbuf:,.2f}. "
                           f"Suggest: top-up ≥ {minbuf-amt:,.2f} or lower min_buffer.")
                    fails.append(msg)
    
    # 6) Balloon payment validation (only if we have bricks)
    if bricks is not None:
        from .kinds import K
        for b in bricks:
            if isinstance(b, LBrick) and b.kind == K.L_MORT_ANN:
                # Check if this mortgage has a balloon policy
                balloon_policy = (b.spec or {}).get("balloon_policy", "payoff")
                if balloon_policy == "payoff":
                    # Check if balloon was properly paid off
                    debt_balance = outputs[b.id]["debt_balance"]
                    cash_out = outputs[b.id]["cash_out"]
                    
                    # Find the last active month
                    mask = active_mask(res["totals"].index.values, b.start_date, b.end_date, b.duration_m)
                    if mask.any():
                        t_stop = np.where(mask)[0][-1]
                        residual_debt = debt_balance[t_stop]
                        
                        if residual_debt > tol:
                            fails.append(f"Balloon inconsistency: mortgage '{b.id}' has residual debt €{residual_debt:,.2f} at end of window but balloon_policy='payoff'")
                        
                        # Check if balloon cash_out includes the residual debt payment
                        # The balloon payment should be at least as large as the residual debt
                        if t_stop > 0:
                            debt_before_balloon = debt_balance[t_stop - 1]
                            balloon_cash_out = cash_out[t_stop]
                            # The balloon payment should be >= the debt before payment (includes regular payment + balloon)
                            if balloon_cash_out > tol and balloon_cash_out < debt_before_balloon - tol:
                                fails.append(f"Balloon payment insufficient: mortgage '{b.id}' balloon cash_out €{balloon_cash_out:,.2f} < debt before payment €{debt_before_balloon:,.2f}")
    
    # 7) ETF units validation (never negative)
    for brick_id, output in outputs.items():
        # Check if this is an ETF brick
        brick = None
        for b in res.get("_scenario_bricks", []):
            if b.id == brick_id:
                brick = b
                break
        
        if brick and hasattr(brick, 'kind') and brick.kind == K.A_INV_ETF:
            asset_value = output["asset_value"]
            # We can't directly check units, but we can check for negative asset values
            if (asset_value < -tol).any():
                t_idx = int(np.where(asset_value < -tol)[0][0])
                val = float(asset_value[t_idx])
                fails.append(f"ETF units negative: '{brick_id}' has negative asset value €{val:,.2f} at month {t_idx}")
    
    # 8) Income escalator monotonicity (when annual_step_pct >= 0)
    for brick_id, output in outputs.items():
        # Check if this is an income brick
        brick = None
        for b in res.get("_scenario_bricks", []):
            if b.id == brick_id:
                brick = b
                break
        
        if brick and hasattr(brick, 'kind') and brick.kind == K.F_INCOME:
            annual_step_pct = float((brick.spec or {}).get("annual_step_pct", 0.0))
            if annual_step_pct >= 0:
                cash_in = output["cash_in"]
                # Get activation mask to only check within active periods
                mask = active_mask(res["totals"].index.values, brick.start_date, brick.end_date, brick.duration_m)
                
                # Check that income is non-decreasing within active periods
                for t in range(1, len(cash_in)):
                    # Only check if both current and previous months are active
                    if mask[t] and mask[t-1] and cash_in[t] < cash_in[t-1] - tol:
                        fails.append(f"Income escalator violation: '{brick_id}' income decreased from €{cash_in[t-1]:,.2f} to €{cash_in[t]:,.2f} at month {t}")
                        break
    
    # 9) Window-end equity identity validation
    if bricks is not None:
        from .kinds import K
        for b in bricks:
            if isinstance(b, (ABrick, LBrick)):
                mask = active_mask(res["totals"].index.values, b.start_date, b.end_date, b.duration_m)
                if not mask.any():
                    continue
                t_stop = int(np.where(mask)[0].max())
                if t_stop + 1 >= len(res["totals"].index):
                    continue
                
                ob = outputs[b.id]
                # Check if there's a stock change at t_stop (auto-dispose/payoff)
                # If stocks change at t_stop, the flows at t_stop should match the change
                d_assets = ob["asset_value"][t_stop+1] - ob["asset_value"][t_stop]
                d_debt = ob["debt_balance"][t_stop+1] - ob["debt_balance"][t_stop]
                flows_t = ob["cash_in"][t_stop] - ob["cash_out"][t_stop]
                
                # Only validate if there's a significant stock change
                if abs(d_assets - d_debt) > 0.01:
                    if abs((d_assets - d_debt) - flows_t) > 0.01:
                        raise ValueError(
                            f"[{b.id}] Window-end equity mismatch at {res['totals'].index[t_stop]}: "
                            f"Δstocks={d_assets - d_debt:.2f} vs flows={flows_t:.2f}. "
                            "Missing sale/payoff or misordered terminal ops?"
                        )
    
    # Handle failures
    if fails:
        full = "Run validation failed: " + " | ".join(fails)
        if mode == "raise":
            raise AssertionError(full)
        else:
            print(f"WARNING: {full}")

# ---------- enhanced export utilities ----------

def export_run_json(path: str, scenario: Scenario, res: dict, include_specs: bool = False, precision: int = 2) -> None:
    """
    Export simulation results to a comprehensive JSON format.
    
    This function creates a structured JSON export that includes:
    - Scenario metadata and brick definitions
    - Time series data for all bricks
    - Time-stamped events with metadata
    - Aggregated totals
    - Validation results and invariants
    
    Args:
        path: Output file path for the JSON file
        scenario: The scenario that was run
        res: Results dictionary from Scenario.run()
        include_specs: Whether to include brick specifications in the export
        precision: Number of decimal places for numeric values
    """
    import json
    import numpy as np
    
    # Convert time index to string format
    t_index = res["totals"].index.strftime("%Y-%m").tolist()
    
    # Extract series data for all bricks
    series = {}
    for brick_id, output in res["outputs"].items():
        series[brick_id] = {}
        for key in ["cash_in", "cash_out", "asset_value", "debt_balance"]:
            if key in output:
                # Convert to list and round to specified precision
                if hasattr(output[key], "tolist"):
                    values = output[key].tolist()
                elif isinstance(output[key], (list, tuple)):
                    values = list(output[key])
                else:
                    values = [output[key]]
                
                if precision >= 0:
                    values = [round(v, precision) if isinstance(v, (int, float)) else v for v in values]
                series[brick_id][key] = values
    
    # Extract and format events
    events = []
    for brick_id, output in res["outputs"].items():
        for event in output.get("events", []):
            event_data = {
                "t": str(event.t.astype("datetime64[M]")),
                "brick_id": brick_id,
                "kind": event.kind,
                "message": event.message,
                "meta": event.meta or {}
            }
            # Add amount if available in meta
            if event.meta and "amount" in event.meta:
                event_data["amount"] = round(event.meta["amount"], precision)
            events.append(event_data)
    
    # Sort events by time
    events.sort(key=lambda x: x["t"])
    
    # Extract totals with precision
    totals = {}
    for col in res["totals"].columns:
        if hasattr(res["totals"][col], "tolist"):
            values = res["totals"][col].tolist()
        else:
            values = list(res["totals"][col])
        
        if precision >= 0:
            values = [round(v, precision) if isinstance(v, (int, float)) else v for v in values]
        totals[col] = values
    
    # Run validation and capture results
    validation_results = {}
    try:
        # Capture validation output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        validate_run(res, mode="warn", tol=1e-6)
        
        sys.stdout = old_stdout
        validation_output = buffer.getvalue()
        
        # Parse validation results
        validation_results = {
            "equity_identity": "equity != assets - debt" not in validation_output,
            "debt_monotone": "debt increased after initial draws" not in validation_output,
            "cash_flow_consistent": "net_cf != cash_in - cash_out" not in validation_output,
            "purchase_settlement_ok": "purchase settlement mismatch" not in validation_output,
            "messages": [line.strip() for line in validation_output.split('\n') if line.strip() and "WARNING:" in line]
        }
    except Exception as e:
        validation_results = {
            "error": str(e),
            "equity_identity": False,
            "debt_monotone": False,
            "cash_flow_consistent": False,
            "purchase_settlement_ok": False,
            "messages": [f"Validation error: {str(e)}"]
        }
    
    # Build the comprehensive JSON structure
    payload = {
        "metadata": {
            "scenario": {
                "id": scenario.id,
                "name": scenario.name
            },
            "simulation_period": {
                "start": t_index[0],
                "end": t_index[-1],
                "months": len(t_index)
            },
            "bricks": [
                {
                    "id": brick.id,
                    "name": brick.name,
                    "family": brick.family,
                    "kind": brick.kind,
                    "start_date": str(brick.start_date) if brick.start_date else None
                }
                for brick in scenario.bricks
            ]
        },
        "t_index": t_index,
        "series": series,
        "events": events,
        "totals": totals,
        "invariants": validation_results
    }
    
    # Optionally include brick specifications
    if include_specs:
        payload["brick_specs"] = {
            brick.id: {
                "spec": brick.spec,
                "links": brick.links
            }
            for brick in scenario.bricks
        }
    
    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)
    
    # Write to file
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

def export_ledger_csv(path: str, res: dict) -> None:
    """
    Export simulation results to a flat ledger CSV format.
    
    This creates a simple CSV with one row per cash flow or event,
    making it easy to eyeball the financial transactions.
    
    Args:
        path: Output file path for the CSV file
        res: Results dictionary from Scenario.run()
    """
    import csv
    
    t_index = res["totals"].index
    rows = []
    
    # Extract cash flows
    for brick_id, output in res["outputs"].items():
        for flow_type in ["cash_in", "cash_out"]:
            arr = output[flow_type]
            for i, val in enumerate(arr):
                if abs(val) > 1e-9:  # Only include non-zero flows
                    rows.append({
                        "t": t_index[i].strftime("%Y-%m"),
                        "brick_id": brick_id,
                        "flow": flow_type,
                        "amount": float(val),
                        "note": ""
                    })
    
    # Extract events
    for brick_id, output in res["outputs"].items():
        for event in output.get("events", []):
            amount = 0.0
            if event.meta and "amount" in event.meta:
                amount = float(event.meta["amount"])
            elif event.meta and "price" in event.meta:
                amount = float(event.meta["price"])
            elif event.meta and "principal" in event.meta:
                amount = float(event.meta["principal"])
            
            rows.append({
                "t": str(event.t.astype("datetime64[M]")),
                "brick_id": brick_id,
                "flow": f"event:{event.kind}",
                "amount": amount,
                "note": event.message
            })
    
    # Sort by time
    rows.sort(key=lambda x: x["t"])
    
    # Write to CSV
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "brick_id", "flow", "amount", "note"])
        writer.writeheader()
        writer.writerows(rows)
