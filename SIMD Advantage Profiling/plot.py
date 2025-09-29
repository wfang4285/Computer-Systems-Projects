import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# --- Data Preparation Function ---

def group_and_aggregate(df):
    """Groups the raw data (5 runs) and calculates median and standard deviation."""

    # Calculate Time Per Element (TPE = Time / N) on the raw data before aggregation
    df['tpe_scalar'] = df['t_scalar'] / df['N']
    df['tpe_simd'] = df['t_simd'] / df['N']

    # Define aggregation structure: we want the median and standard deviation for key metrics
    agg_funcs = {
        't_scalar': ['median', 'std'],
        't_simd': ['median', 'std'],
        'speedup': ['median', 'std'],
        'gflops': ['median', 'std'],
        'tpe_scalar': ['median', 'std'],
        'tpe_simd': ['median', 'std'],
    }

    # Group by all experimental variables (except the run ID) and apply aggregation
    agg_data = df.groupby(['kernel', 'type', 'extra', 'N']).agg(agg_funcs).reset_index()

    # Flatten the multi-level column names for easier access (e.g., 'speedup_median')
    agg_data.columns = [
        '_'.join(col).strip() if col[1] else col[0]
        for col in agg_data.columns.values
    ]

    # Filter out clearly corrupt GFLOPs data (e.g., above 1000 GFLOPs is highly unlikely)
    agg_data = agg_data[agg_data['gflops_median'] < 1000].copy()

    return agg_data

# --- Plotting Functions using Aggregated Data with Error Bars ---

# Function to plot speedup vs N
def plot_speedup(agg_df, kernel_name):
    # Filter out stride/alignment tests for the main locality sweep plot
    subset = agg_df[(agg_df['kernel']==kernel_name) & (~agg_df['extra'].str.contains('stride|alignment'))].copy()

    if subset.empty:
        print(f"Skipping {kernel_name} speedup plot: No valid data after filtering.")
        return

    plt.figure(figsize=(9,7))
    for dtype in subset['type'].unique():
        d = subset[subset['type']==dtype]

        # Plot median with error bars (std)
        plt.errorbar(d['N'], d['speedup_median'], yerr=d['speedup_std'],
                     fmt='-o', capsize=4, label=f'{dtype} Median', alpha=0.8)

    plt.xscale('log')
    # Draw a line at 1.0 for easy reference of no speedup
    plt.axhline(1.0, color='r', linestyle='--', linewidth=1, label='1.0x (No Speedup)')
    max_speedup = subset['speedup_median'].max() if not subset.empty else 1.5
    plt.ylim(0, max_speedup * 1.1)

    plt.xlabel("Array Size (N) - Log Scale")
    plt.ylabel("Speedup (Scalar / SIMD)")
    plt.title(f"{kernel_name} SIMD Speedup vs. Array Size (Median $\pm$ Std Dev)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Data Type")
    plt.tight_layout()
    plt.savefig(f'plots/{kernel_name}_speedup.png')
    plt.close()

# Function to plot GFLOPs vs N
def plot_gflops(agg_df, kernel_name):
    # Filter out stride/alignment tests for the main locality sweep plot
    subset = agg_df[(agg_df['kernel']==kernel_name) & (~agg_df['extra'].str.contains('stride|alignment'))].copy()

    if subset.empty:
        print(f"Skipping {kernel_name} GFLOPs plot: No valid data after filtering.")
        return

    plt.figure(figsize=(9,7))
    for dtype in subset['type'].unique():
        d = subset[subset['type']==dtype]

        # Plot median with error bars (std)
        plt.errorbar(d['N'], d['gflops_median'], yerr=d['gflops_std'],
                     fmt='-o', capsize=4, label=f'{dtype} Median', alpha=0.8)

    plt.xscale('log')

    # Set a sane maximum for GFLOPs
    max_gflops = subset['gflops_median'].max() * 1.1 if not subset.empty else 20
    plt.ylim(0, max_gflops)

    plt.xlabel("Array Size (N) - Log Scale")
    plt.ylabel("Achieved Throughput (GFLOP/s)")
    plt.title(f"{kernel_name} Throughput vs. Array Size (Median $\pm$ Std Dev)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Data Type")
    plt.tight_layout()
    plt.savefig(f'plots/{kernel_name}_gflops.png')
    plt.close()

# Function to plot Time per Element (TPE)
def plot_time_per_element(agg_df, kernel_name):
    # TPE is a good proxy for Cycles Per Element (CPE). Lower is better.
    subset = agg_df[(agg_df['kernel']==kernel_name) & (~agg_df['extra'].str.contains('stride|alignment'))].copy()

    if subset.empty:
        print(f"Skipping {kernel_name} TPE plot: No valid data after filtering.")
        return

    plt.figure(figsize=(9,7))

    for dtype in subset['type'].unique():
        d = subset[subset['type']==dtype]

        # Plot Scalar TPE
        plt.errorbar(d['N'], d['tpe_scalar_median'], yerr=d['tpe_scalar_std'],
                     fmt=':x', capsize=3, label=f'{dtype} (Scalar) Median', alpha=0.7)
        # Plot SIMD TPE
        plt.errorbar(d['N'], d['tpe_simd_median'], yerr=d['tpe_simd_std'],
                     fmt='-o', capsize=4, label=f'{dtype} (SIMD) Median', alpha=0.8)

    plt.xscale('log')
    plt.yscale('log') # Log scale is essential for TPE/CPE locality plots
    plt.xlabel("Array Size (N) - Log Scale")
    plt.ylabel("Time per Element (seconds/element) [Proxy for CPE] - Log Scale")
    plt.title(f"{kernel_name} Locality Effect (SIMD vs. Scalar)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Implementation")
    plt.tight_layout()
    plt.savefig(f'plots/{kernel_name}_tpe_locality.png')
    plt.close()

# Function to plot Stride Effects
def plot_stride_gflops(agg_df, kernel_name='saxpy', dtype='float32'):
    # Plots GFLOPs vs. Stride for the smallest N

    # Filter for the stride runs for the specific target_N (Smallest N)
    target_N = agg_df[agg_df['extra'].str.contains('stride')]['N'].min()

    subset = agg_df[(agg_df['kernel']==kernel_name) &
                    (agg_df['type']==dtype) &
                    (agg_df['extra'].str.contains('stride')) &
                    (agg_df['N']==target_N)].copy()

    if subset.empty:
        print(f"Warning: No stride data found for {kernel_name}, {dtype}, N={target_N}.")
        return

    # Extract the stride number from the 'extra' column. We assume 'strideX' where X is the number.
    subset['stride'] = subset['extra'].str.replace('stride', '').astype(int)

    plt.figure(figsize=(8,6))

    # Plot median with error bars (std)
    plt.errorbar(subset['stride'], subset['gflops_median'], yerr=subset['gflops_std'],
                 fmt='-o', capsize=4, label='SIMD GFLOP/s Median', alpha=0.8)

    plt.xscale('log', base=2)
    plt.xticks(subset['stride'], labels=subset['stride'])

    plt.xlabel("Stride (S) - Log2 Scale")
    plt.ylabel("Achieved Throughput (GFLOP/s)")
    plt.title(f"{kernel_name} Stride Effect on Throughput ({dtype}, N={target_N})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{kernel_name}_{dtype}_stride_gflops.png')
    plt.close()


def plot_alignment_tests(agg_df):
    # Plots the median execution time for dot product alignment tests
    subset = agg_df[(agg_df['kernel']=='alignment')].copy()

    if subset.empty:
        print("Warning: No alignment data found.")
        return

    plt.figure(figsize=(10, 6))

    # Filter out cases where median time is zero or near zero for cleaner plotting
    # Also ensure we only plot the SIMD time for the comparison
    plot_data = subset[subset['t_simd_median'] > 1e-10].copy()

    # Define the order of the load types on the X-axis
    labels_order = ['aligned_load', 'aligned_loadu', 'misaligned_loadu'] # Omit 'scalar' as it's not a SIMD time
    labels = [label for label in labels_order if label in plot_data['extra'].unique()]

    # Get the unique N values and sort them
    n_values = sorted(plot_data['N'].unique())

    # Use a color map for visual separation of N groups
    colors = plt.cm.get_cmap('Set1', len(n_values))

    bar_width = 0.25 # Adjusted width for better spacing
    x_positions_base = np.arange(len(labels))

    # Plot bars for each N value
    for i, N in enumerate(n_values):
        n_data = plot_data[plot_data['N'] == N]

        # Create a lookup map for the N's data and ensure times array matches the length of labels
        n_data_map = n_data.set_index('extra')['t_simd_median'].to_dict()
        # Use .get(label, 0) to handle missing data gracefully (though 0 time won't plot)
        times = [n_data_map.get(label, 0) for label in labels]

        # Offset bars for different N values
        x_positions = x_positions_base + i * bar_width

        plt.bar(x_positions, times, bar_width, label=f'N={N}', color=colors(i))

    # Clean up X-axis labels: center the ticks between the grouped bars
    x_tick_centers = x_positions_base + (len(n_values) - 1) * bar_width / 2
    plt.xticks(x_tick_centers, labels)

    plt.ylabel("Median Execution Time (seconds)")
    plt.title("Effect of Memory Alignment on Dot Product (SIMD Time)")
    plt.legend(title="Array Size (N)")
    plt.grid(axis='y', ls="--", alpha=0.5)

    # Set the Y-axis to start from zero and ensure readability
    max_time = plot_data['t_simd_median'].max()
    plt.ylim(0, max_time * 1.1)

    plt.tight_layout()
    plt.savefig(f'plots/alignment_time_fixed.png')
    plt.close()


# --- Main Execution ---

try:
    df = pd.read_csv('all_results.csv', header=None,
                     names=['run_id', 'kernel', 'type', 'extra', 'N', 't_scalar', 't_simd', 'speedup', 'gflops'])

    # 1. Aggregate the 5 runs into median and std deviation data
    aggregated_df = group_and_aggregate(df)

    # Ensure a 'plots' directory exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 2. Run all core plots for the selected kernels (Speedup, GFLOPs, TPE)
    for kernel in ['saxpy','dot','elem_mult']:
        plot_speedup(aggregated_df, kernel)
        plot_gflops(aggregated_df, kernel)
        plot_time_per_element(aggregated_df, kernel) # CPE Proxy

    # 3. Run stride tests
    plot_stride_gflops(aggregated_df, 'saxpy', 'float32')

    # 4. Run alignment tests
    plot_alignment_tests(aggregated_df)

    print("All required plots (with error bars!) saved successfully in the 'plots/' folder.")
    print("Files created: saxpy_speedup.png, saxpy_gflops.png, saxpy_tpe_locality.png,")
    print("dot_speedup.png, dot_gflops.png, dot_tpe_locality.png,")
    print("elem_mult_speedup.png, elem_mult_gflops.png, elem_mult_tpe_locality.png,")
    print("saxpy_float32_stride_gflops.png, and alignment_time.png.")

except FileNotFoundError:
    print("Error: 'all_results.csv' not found. Please ensure your C++ program has run and generated the data.")
except Exception as e:
    print(f"An unexpected error occurred during plotting: {e}")
