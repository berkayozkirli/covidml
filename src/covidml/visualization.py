"""Plots used in the exploratory notebook and report."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def pick_series(country_code: str, features: list[str], dataset: pd.DataFrame, picked_measures: list[str]) -> None:
    # 1. Filter and sort by country
    country_data = dataset[dataset['iso_code'] == country_code].sort_values(by='time_step').copy()

    # 2. Handle missing values (crucial for vaccines which are NaN before rollout)
    for col in features:
        if 'vaccin' in col.lower():
            country_data[col] = country_data[col].fillna(0)
        else:
            country_data[col] = country_data[col].ffill().fillna(0)

    # 3. Min-Max Scale features to [0, 1] range for visual trajectory comparison
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(country_data[features])

    fig, ax = plt.subplots(figsize=(15, 7.0))
    # Colors: Blue (cases), Green (Rt), Purple (tests), Pink (vaccines)
    colors = ['#1f77b4', '#2ca02c', '#9467bd', '#e377c2', '#8c564b', '#17becf']

    # 4. Plot each dynamic feature trajectory
    for i, feature in enumerate(features):
        clean_label = " ".join(word.capitalize() for word in feature.split("_"))
        ax.plot(
            country_data["time_step"],
            scaled_values[:, i],
            label=clean_label,
            linewidth=2.5,
            color=colors[i % len(colors)],
            alpha=0.85
        )

    # 5. Plot Harsh Measure Triggers (Red dotted lines)
    added_measure_label = False
    for measure in picked_measures:
        if measure in country_data.columns:
            trigger_days = country_data[country_data[measure] == 1]["time_step"]
            for step in trigger_days:
                label = 'Harsh Measure Implemented' if not added_measure_label else ""
                ax.axvline(x=step, color='#d62728', linestyle=':', alpha=0.4, linewidth=1.5, label=label)
                added_measure_label = True

    # 6. Plot Train / Val / Test Cut-off Boundaries
    ax.axvline(x=241, color='#ff7f0e', linestyle='--', linewidth=2.5, label='Val Start (Day 241)')
    ax.axvline(x=300, color='#333333', linestyle='--', linewidth=2.5, label='Test Start (Day 300 / Vaccines Begin)')

    # Add light shading for the Test Set (to visually emphasize the domain shift)
    max_step = country_data["time_step"].max()
    ax.axvspan(300, max_step, color='#333333', alpha=0.07, label='Test Epoch (Vaccine Era)')

    # 7. Formatting & Aesthetics
    ax.set_title(f"Normalized Trajectories of Important Features: {country_code}", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Time Step (Days since earliest record)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Scaled Trajectory [0, 1]", fontsize=12, fontweight='bold')
    ax.set_xlim(country_data["time_step"].min(), max_step)
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, linestyle='-', alpha=0.2)

    ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True, facecolor='white', framealpha=0.95)

    plt.tight_layout()
    plt.show()
