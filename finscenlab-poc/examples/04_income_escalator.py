#!/usr/bin/env python3
"""
FinScenLab Example 04: Income Escalator

This example demonstrates the enhanced income strategy with:
- Annual escalation with percentage increases
- Custom step months (e.g., June every year)
- Non-annual escalation (e.g., every 18 months)
- Multiple income streams with different patterns
- Career progression modeling

This shows how to model realistic salary growth and career advancement.
"""

# Fix the import path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the core components
from datetime import date
from finscenlab.core import Scenario, ABrick, FBrick, validate_run
from finscenlab.kinds import K
import finscenlab.strategies  # This registers the default strategies

print("💼 FinScenLab Example 04: Income Escalator")
print("=" * 70)

# =============================================================================
# SCENARIO SETUP
# =============================================================================

# Cash account
cash = ABrick(
    id="cash:EUR", 
    name="Personal Cash", 
    kind=K.A_CASH,
    spec={
        "initial_balance": 0.0, 
        "overdraft_limit": 0.0, 
        "min_buffer": 0.0
    }
)

# Initial capital
seed = FBrick(
    id="seed", 
    name="Initial Savings", 
    kind=K.F_TRANSFER, 
    spec={"amount": 50000}  # €50k initial savings
)

# Primary salary with annual escalation
primary_salary = FBrick(
    id="primary_salary", 
    name="Primary Job Salary", 
    kind=K.F_INCOME,
    start_date=date(2026, 1, 1),
    spec={
        "amount_monthly": 5000,  # €5k/month base salary
        "annual_step_pct": 0.04,  # 4% annual increase
        "step_month": 6  # Escalate in June every year
    }
)

# Bonus income with different escalation pattern
bonus = FBrick(
    id="bonus", 
    name="Annual Bonus", 
    kind=K.F_INCOME,
    start_date=date(2026, 1, 1),
    spec={
        "amount_monthly": 800,  # €800/month average (€9.6k/year)
        "annual_step_pct": 0.06,  # 6% annual increase
        "step_month": 12  # Escalate in December
    }
)

# Freelance income with non-annual escalation
freelance = FBrick(
    id="freelance", 
    name="Freelance Income", 
    kind=K.F_INCOME,
    start_date=date(2026, 6, 1),  # Start freelancing in June 2026
    end_date=date(2030, 5, 1),    # Stop after 4 years
    spec={
        "amount_monthly": 1500,  # €1.5k/month
        "step_pct": 0.08,  # 8% increase every 18 months
        "step_every_m": 18  # Escalate every 18 months
    }
)

# Side business with anniversary-based escalation
side_business = FBrick(
    id="side_business", 
    name="Side Business", 
    kind=K.F_INCOME,
    start_date=date(2027, 3, 1),  # Start business in March 2027
    spec={
        "amount_monthly": 2000,  # €2k/month
        "annual_step_pct": 0.10,  # 10% annual increase
        # No step_month specified - uses anniversary of start_date (March)
    }
)

# Living expenses (fixed for simplicity)
living_expenses = FBrick(
    id="living_expenses", 
    name="Living Expenses", 
    kind=K.F_EXP_LIVING,
    start_date=date(2026, 1, 1),
    spec={
        "amount_monthly": 3000  # €3k/month fixed expenses
    }
)

# Create the scenario
scenario = Scenario(
    id="income_escalator", 
    name="Career Progression with Multiple Income Streams",
    bricks=[cash, seed, primary_salary, bonus, freelance, side_business, living_expenses]
)

print("✅ Scenario created successfully!")
print(f"📋 Scenario: {scenario.name}")
print(f"💰 Initial Savings: €{seed.spec['amount']:,.0f}")
print(f"💼 Primary Salary: €{primary_salary.spec['amount_monthly']:,.0f}/month, {primary_salary.spec['annual_step_pct']*100:.1f}% annual (June)")
print(f"🎁 Bonus: €{bonus.spec['amount_monthly']:,.0f}/month, {bonus.spec['annual_step_pct']*100:.1f}% annual (December)")
print(f"🖥️ Freelance: €{freelance.spec['amount_monthly']:,.0f}/month, {freelance.spec['step_pct']*100:.1f}% every 18 months")
print(f"🏪 Side Business: €{side_business.spec['amount_monthly']:,.0f}/month, {side_business.spec['annual_step_pct']*100:.1f}% annual (March)")
print(f"🏠 Living Expenses: €{living_expenses.spec['amount_monthly']:,.0f}/month (fixed)")

# =============================================================================
# RUN SIMULATION
# =============================================================================

print("\n🚀 Running simulation...")
print("-" * 50)

try:
    results = scenario.run(start=date(2026, 1, 1), months=60)  # 5 years
    validate_run(results, scenario.bricks, mode="raise")
    print("✅ Simulation completed successfully!")
    
except AssertionError as e:
    print("❌ Simulation failed validation:")
    print(f"   {str(e)}")
    exit(1)

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

print("\n📊 RESULTS ANALYSIS")
print("=" * 70)

# Extract key outputs
totals = results['totals']
primary_out = results['outputs']['primary_salary']
bonus_out = results['outputs']['bonus']
freelance_out = results['outputs']['freelance']
side_business_out = results['outputs']['side_business']
expenses_out = results['outputs']['living_expenses']

# Show income progression
print("\n💼 INCOME PROGRESSION:")
print("-" * 50)
print("Month | Primary | Bonus  | Freelance | Side Biz | Total   | Expenses | Net")
print("-" * 70)
for i in range(0, 60, 6):  # Every 6 months
    month = i
    year = month // 12 + 2026
    month_name = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][month % 12]
    
    primary = primary_out['cash_in'][i]
    bonus_amt = bonus_out['cash_in'][i]
    freelance_amt = freelance_out['cash_in'][i]
    side_biz = side_business_out['cash_in'][i]
    total_income = totals['cash_in'].iloc[i]
    expenses = expenses_out['cash_out'][i]
    net = total_income - expenses
    
    print(f"{year}-{month_name:3s} | €{primary:6,.0f} | €{bonus_amt:5,.0f} | €{freelance_amt:8,.0f} | €{side_biz:7,.0f} | €{total_income:6,.0f} | €{expenses:7,.0f} | €{net:5,.0f}")

# Show escalation events
print("\n📈 ESCALATION EVENTS:")
print("-" * 50)
all_events = []
for brick_id in ['primary_salary', 'bonus', 'freelance', 'side_business']:
    for event in results['outputs'][brick_id]['events']:
        all_events.append((event.t, brick_id, event.message))

# Sort by date and show
all_events.sort(key=lambda x: x[0])
for date_str, brick_id, message in all_events:
    print(f"  {date_str}: {brick_id} - {message}")

# Show final financial position
print("\n💼 FINAL FINANCIAL POSITION:")
print("-" * 50)
final_cash = totals['cash'].iloc[-1]
final_income = totals['cash_in'].iloc[-1]
final_expenses = totals['cash_out'].iloc[-1]
final_net = final_income - final_expenses

print(f"  Final Cash Balance: €{final_cash:,.0f}")
print(f"  Final Monthly Income: €{final_income:,.0f}")
print(f"  Final Monthly Expenses: €{final_expenses:,.0f}")
print(f"  Final Monthly Net: €{final_net:,.0f}")

# Calculate income growth
print("\n📈 INCOME GROWTH ANALYSIS:")
print("-" * 50)
initial_income = totals['cash_in'].iloc[0]
final_income = totals['cash_in'].iloc[-1]
total_growth = final_income - initial_income
growth_pct = (total_growth / initial_income) * 100

print(f"  Initial Monthly Income: €{initial_income:,.0f}")
print(f"  Final Monthly Income: €{final_income:,.0f}")
print(f"  Total Growth: €{total_growth:,.0f}")
print(f"  Growth Percentage: {growth_pct:.1f}%")

# Show individual income stream growth
print("\n💼 INDIVIDUAL STREAM GROWTH:")
print("-" * 50)
streams = [
    ("Primary Salary", primary_out),
    ("Bonus", bonus_out),
    ("Freelance", freelance_out),
    ("Side Business", side_business_out)
]

for name, output in streams:
    initial = output['cash_in'][0] if output['cash_in'][0] > 0 else output['cash_in'][output['cash_in'] > 0][0] if (output['cash_in'] > 0).any() else 0
    final = output['cash_in'][-1]
    if initial > 0:
        growth = final - initial
        growth_pct = (growth / initial) * 100
        print(f"  {name:15s}: €{initial:6,.0f} → €{final:6,.0f} (+€{growth:5,.0f}, +{growth_pct:4.1f}%)")

# Show savings rate
print("\n💰 SAVINGS RATE ANALYSIS:")
print("-" * 50)
total_income = totals['cash_in'].sum()
total_expenses = totals['cash_out'].sum()
total_savings = total_income - total_expenses
savings_rate = (total_savings / total_income) * 100 if total_income > 0 else 0

print(f"  Total Income (5 years): €{total_income:,.0f}")
print(f"  Total Expenses (5 years): €{total_expenses:,.0f}")
print(f"  Total Savings (5 years): €{total_savings:,.0f}")
print(f"  Average Savings Rate: {savings_rate:.1f}%")

# =============================================================================
# LESSONS LEARNED
# =============================================================================

print("\n🎓 LESSONS LEARNED:")
print("=" * 70)
print("1. 💼 Multiple income streams provide diversification and stability")
print("2. 📈 Different escalation patterns match different career phases")
print("3. 🎯 Custom step months align with company review cycles")
print("4. ⏰ Non-annual escalation models irregular career advancement")
print("5. 🏪 Side businesses can provide significant income growth")
print("6. 📊 Activation windows enable precise timing of income changes")
print("7. 💰 Escalation events provide clear career milestone tracking")

print("\n✅ Example 04 completed successfully!")
print("\nThis example demonstrates realistic career progression modeling:")
print("- Multiple income streams with different escalation patterns")
print("- Annual escalation with custom step months")
print("- Non-annual escalation for irregular advancement")
print("- Activation windows for precise timing")
print("- Comprehensive income growth analysis")
print("- Integration with expense modeling")
