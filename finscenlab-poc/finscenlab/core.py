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
from typing import Protocol, TypedDict, Dict, List, Optional, Callable
from datetime import date
import numpy as np
import pandas as pd

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

# ---------- common output structure for ALL bricks ----------

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
        events: List of textual notes describing key events during simulation
        
    Note:
        All numpy arrays have the same length corresponding to the simulation period.
        Cash flows are always positive values - the direction is implicit in the field name.
    """
    cash_in: np.ndarray        # Monthly cash inflows (>=0)
    cash_out: np.ndarray       # Monthly cash outflows (>=0)
    asset_value: np.ndarray    # Monthly asset value (0 if not an asset)
    debt_balance: np.ndarray   # Monthly debt balance (0 if not a liability)
    events: List[str]          # Textual notes describing key events

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
        
    Note:
        The 'family' attribute is automatically set by subclasses and should not be
        specified manually when creating brick instances.
    """
    id: str
    name: str
    kind: str             # Dot-separated string discriminator
    currency: str = "EUR"
    spec: dict = None     # Strategy-specific parameters
    links: dict = None    # References to other bricks
    family: str = None    # 'a' | 'l' | 'f' - set automatically in subclasses

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
        attributes (valuation, schedule, or flow).
    """
    for b in bricks:
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

    def run(self, start: date, months: int) -> dict:
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
            out = b.simulate(ctx)
            outputs[b.id] = out
            routed_in  += out["cash_in"]
            routed_out += out["cash_out"]

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
        
        return {"outputs": outputs, "totals": totals}
