#!/usr/bin/env python3
"""
Generate bifurcation figure for JEDC paper.

This script creates Figure showing:
(a) Cusp bifurcation in phase space showing fold catastrophe
(b) Crisis probability heatmap as function of memory decay and network degree
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

def cusp_surface(x, a, b):
    """
    Cusp catastrophe surface: x^3 + ax + b = 0
    Returns stable and unstable manifolds.
    """
    # For visualization, we solve for the equilibrium surface
    return x**3 + a*x + b

def generate_cusp_bifurcation(ax):
    """Generate cusp bifurcation surface (panel a)."""

    # Control parameters
    a = np.linspace(-2, 2, 100)  # Tension (splitting factor)
    b = np.linspace(-2, 2, 100)  # Memory (normal factor)
    A, B = np.meshgrid(a, b)

    # State variable (output/phase)
    X = np.linspace(-2, 2, 100)

    # Create the cusp surface
    # The cusp is defined by the set where x^3 + ax + b = 0
    # We'll plot the surface z = x^3 + ax + b = 0 rearranged as x vs (a, b)

    # For 3D surface, compute equilibrium states
    A_3d, X_3d = np.meshgrid(a, X)
    B_surface = -X_3d**3 - A_3d * X_3d

    # Plot the surface
    surf = ax.plot_surface(A_3d, B_surface, X_3d, cmap=cm.coolwarm,
                           alpha=0.7, linewidth=0, antialiased=True)

    # Plot the bifurcation curve (cusp) on the control plane
    # Cusp curve: 4a^3 + 27b^2 = 0 at z=0
    t = np.linspace(-1.5, 1.5, 200)
    a_cusp = -3 * t**2 / 2
    b_cusp = t**3
    z_cusp = np.zeros_like(t)
    ax.plot(a_cusp, b_cusp, z_cusp - 2.1, 'k-', linewidth=2, label='Cusp')

    # Add fold lines (projections)
    ax.plot(a_cusp, b_cusp, t, 'k--', linewidth=1.5, alpha=0.5)

    # Labels
    ax.set_xlabel(r'Tension $T_{adj}$', labelpad=10)
    ax.set_ylabel(r'Memory $M_{macro}$', labelpad=10)
    ax.set_zlabel(r'Output $Y$', labelpad=10)
    ax.set_title('(a) Cusp Bifurcation Surface', fontsize=12, fontweight='bold')

    # Adjust view angle
    ax.view_init(elev=25, azim=45)

    # Add annotations for stable/unstable regions
    ax.text(1.5, 0, 1.5, 'Stable\n(Expansion)', fontsize=9, ha='center')
    ax.text(1.5, 0, -1.5, 'Stable\n(Recession)', fontsize=9, ha='center')
    ax.text(-0.5, 0, 0, 'Unstable', fontsize=9, ha='center', color='red')

def generate_crisis_probability_heatmap(ax):
    """Generate crisis probability heatmap (panel b)."""

    # Parameters
    delta_m = np.linspace(0.01, 0.15, 50)  # Memory decay
    k_avg = np.linspace(2, 15, 50)  # Average network degree

    D, K = np.meshgrid(delta_m, k_avg)

    # Crisis probability model based on paper equations
    # Higher memory decay (faster forgetting) -> higher crisis probability
    # Higher connectivity -> can amplify shocks -> higher crisis probability
    # But very high connectivity can also stabilize through diversification

    # Non-monotonic relationship with network degree (U-shape inverted)
    k_optimal = 8  # Optimal connectivity for stability
    k_effect = 0.3 * ((K - k_optimal) / 5)**2

    # Memory effect: higher decay = less learning = more crises
    memory_effect = 2 * D

    # Base probability with interaction
    base = 0.15
    interaction = 0.1 * D * np.log(K + 1)

    P_crisis = base + k_effect + memory_effect + interaction

    # Apply logistic transform to bound between 0 and 1
    P_crisis = 1 / (1 + np.exp(-3 * (P_crisis - 0.5)))

    # Plot heatmap
    im = ax.pcolormesh(D, K, P_crisis, cmap='RdYlGn_r', shading='auto',
                       vmin=0.1, vmax=0.7)

    # Add contour lines
    contours = ax.contour(D, K, P_crisis, levels=[0.2, 0.3, 0.4, 0.5],
                          colors='black', linewidths=0.8, alpha=0.7)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')

    # Mark optimal region
    optimal_region = (D > 0.03) & (D < 0.07) & (K > 5) & (K < 10)
    # ax.contour(D, K, optimal_region.astype(float), levels=[0.5],
    #            colors='blue', linewidths=2, linestyles='dashed')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label=r'$P(\mathrm{Crisis})$')
    cbar.ax.tick_params(labelsize=9)

    # Labels
    ax.set_xlabel(r'Memory Decay $\delta_M$', fontsize=11)
    ax.set_ylabel(r'Average Degree $\langle k \rangle$', fontsize=11)
    ax.set_title('(b) Crisis Probability Map', fontsize=12, fontweight='bold')

    # Add annotation for regions
    ax.annotate('Low Risk', xy=(0.04, 7), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax.annotate('High Risk', xy=(0.12, 12), fontsize=10, ha='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

def main():
    """Generate the complete bifurcation figure."""

    # Create figure with two panels
    fig = plt.figure(figsize=(14, 6))

    # Panel (a): 3D cusp bifurcation
    ax1 = fig.add_subplot(121, projection='3d')
    generate_cusp_bifurcation(ax1)

    # Panel (b): 2D crisis probability heatmap
    ax2 = fig.add_subplot(122)
    generate_crisis_probability_heatmap(ax2)

    plt.tight_layout()

    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # Save in multiple formats
    fig.savefig(os.path.join(figures_dir, 'fig_bifurcation.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(os.path.join(figures_dir, 'fig_bifurcation.pdf'),
                bbox_inches='tight', facecolor='white')

    print(f"Bifurcation figure saved to {figures_dir}/")
    plt.close()

if __name__ == "__main__":
    main()
