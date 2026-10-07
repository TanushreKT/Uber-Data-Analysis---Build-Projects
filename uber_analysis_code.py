import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("uber_data.csv")
df.columns = df.columns.str.strip()

# Clean booking value & ride distance
df["Booking Value"] = pd.to_numeric(df["Booking Value"], errors="coerce")
df["Ride Distance"] = pd.to_numeric(df["Ride Distance"], errors="coerce")

# ── Q1: Cancellation & Incomplete rates by Vehicle Type ────────────────────────
failure_statuses = ["Cancelled by Customer", "Cancelled by Driver",
                    "No Driver Found", "Incomplete"]

total_by_vtype  = df.groupby("Vehicle Type").size().rename("Total")
failed_by_vtype = (df[df["Booking Status"].isin(failure_statuses)]
                   .groupby("Vehicle Type").size().rename("Failed"))
failure_rate    = (failed_by_vtype / total_by_vtype * 100).rename("Failure Rate %").round(1)
failure_df      = pd.concat([total_by_vtype, failed_by_vtype, failure_rate], axis=1).dropna()
failure_df      = failure_df.sort_values("Failure Rate %", ascending=False)

# Top cancellation reasons (driver side)
driver_cancel_reasons = (df["Driver Cancellation Reason"]
                         .dropna().value_counts().head(5))

# Top cancellation reasons (customer side)
customer_cancel_reasons = (df["Reason for cancelling by Customer"]
                           .dropna().value_counts().head(5))

print("=" * 60)
print("Q1 — FAILURE RATE BY VEHICLE TYPE")
print("=" * 60)
print(failure_df.to_string())
print("\nTop Driver Cancellation Reasons:")
print(driver_cancel_reasons.to_string())
print("\nTop Customer Cancellation Reasons:")
print(customer_cancel_reasons.to_string())

# ── Q2: Revenue & Distance-Value correlation by Vehicle Type ───────────────────
completed = df[df["Booking Status"] == "Completed"].copy()

revenue_df = (completed.groupby("Vehicle Type")
              .agg(
                  Total_Revenue=("Booking Value", "sum"),
                  Avg_Booking_Value=("Booking Value", "mean"),
                  Avg_Distance=("Ride Distance", "mean"),
                  Ride_Count=("Booking Value", "count")
              )
              .round(2)
              .sort_values("Total_Revenue", ascending=False))

overall_corr = completed[["Booking Value", "Ride Distance"]].corr().iloc[0, 1]

print("\n" + "=" * 60)
print("Q2 — REVENUE & AVG BOOKING VALUE BY VEHICLE TYPE")
print("=" * 60)
print(revenue_df.to_string())
print(f"\nOverall correlation (Booking Value vs Ride Distance): {overall_corr:.3f}")

# ── Plots ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Uber Ride Data — Business Analysis", fontsize=16, fontweight="bold", y=1.01)
COLORS = plt.cm.tab10.colors

# Plot 1 – Failure Rate by Vehicle Type
ax1 = axes[0, 0]
bars = ax1.barh(failure_df.index, failure_df["Failure Rate %"],
                color=[COLORS[i % 10] for i in range(len(failure_df))])
ax1.set_xlabel("Failure Rate (%)")
ax1.set_title("Q1 · Failure Rate by Vehicle Type\n(Cancelled + No Driver + Incomplete)")
ax1.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
ax1.set_xlim(0, failure_df["Failure Rate %"].max() * 1.2)

# Plot 2 – Top Cancellation Reasons (combined)
ax2 = axes[0, 1]
all_reasons = pd.concat([driver_cancel_reasons.rename("Driver"),
                         customer_cancel_reasons.rename("Customer")], axis=1).fillna(0)
all_reasons = all_reasons.sort_values("Driver", ascending=True)
y = np.arange(len(all_reasons))
h = 0.35
ax2.barh(y + h/2, all_reasons["Driver"],  h, label="Driver",   color=COLORS[0])
ax2.barh(y - h/2, all_reasons["Customer"], h, label="Customer", color=COLORS[1])
ax2.set_yticks(y)
ax2.set_yticklabels(all_reasons.index, fontsize=8)
ax2.set_xlabel("Count")
ax2.set_title("Q1 · Top Cancellation Reasons\n(Driver vs Customer)")
ax2.legend()

# Plot 3 – Total Revenue by Vehicle Type
ax3 = axes[1, 0]
rev_sorted = revenue_df.sort_values("Total_Revenue", ascending=True)
bars3 = ax3.barh(rev_sorted.index, rev_sorted["Total_Revenue"],
                 color=[COLORS[i % 10] for i in range(len(rev_sorted))])
ax3.set_xlabel("Total Revenue")
ax3.set_title("Q2 · Total Revenue by Vehicle Type\n(Completed Rides)")
ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
ax3.bar_label(bars3,
              labels=[f"₹{v/1e6:.2f}M" for v in rev_sorted["Total_Revenue"]],
              padding=3, fontsize=8)
ax3.set_xlim(0, rev_sorted["Total_Revenue"].max() * 1.25)

# Plot 4 – Avg Booking Value vs Avg Distance scatter
ax4 = axes[1, 1]
for i, (vtype, row) in enumerate(revenue_df.iterrows()):
    ax4.scatter(row["Avg_Distance"], row["Avg_Booking_Value"],
                s=row["Ride_Count"] / 50, color=COLORS[i % 10],
                alpha=0.85, label=vtype)
    ax4.annotate(vtype, (row["Avg_Distance"], row["Avg_Booking_Value"]),
                 textcoords="offset points", xytext=(6, 4), fontsize=8)
ax4.set_xlabel("Avg Ride Distance (km)")
ax4.set_ylabel("Avg Booking Value (₹)")
ax4.set_title(f"Q2 · Avg Booking Value vs Avg Distance\n"
              f"(bubble size ∝ ride count  |  r = {overall_corr:.2f})")

plt.tight_layout()
plt.savefig("uber_analysis.png", dpi=150, bbox_inches="tight")
print("\n✅ Chart saved to uber_analysis.png")
