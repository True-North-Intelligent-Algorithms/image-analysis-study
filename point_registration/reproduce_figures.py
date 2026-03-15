"""
Point visualization functions for registration analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix


def plot_pixel_microns_points(
    coordinate_data_file='coordinate_data_deidentified_nuclei.csv',
    ground_truth_file='ground_truth_coords.csv',
    output_filename='Pixel_Microns_Points_Nuclei1.png',
    figsize=(12, 10)
):
    """
    Plot the actual coordinate points for pixel-to-microns IDs on nuclei1.
    
    Parameters:
    -----------
    coordinate_data_file : str
        Path to the coordinate data CSV file (relative to script directory)
    ground_truth_file : str
        Path to the ground truth CSV file (relative to script directory)
    output_filename : str
        Filename to save the plot
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Construct full path to coordinate data file
    data_path = script_dir / coordinate_data_file
    gt_path = script_dir / ground_truth_file
    
    # IDs that correspond to pixel to microns scaling issues
    pixel_microns_ids = ['R_0638fJAPpmzLtu1', 'R_3RxOa8kaiyul4bG']
    
    # Load coordinate data
    print(f"Loading coordinate data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Load ground truth data
    print(f"Loading ground truth data from {gt_path}...")
    df_gt = pd.read_csv(gt_path, index_col=0)
    
    # Filter ground truth for nuclei1 (out_c00_dr90_label.tif)
    gt_nuclei1 = df_gt[df_gt['path'].str.contains('out_c00_dr90_label.tif', case=False, na=False)].copy()
    print(f"Found {len(gt_nuclei1)} ground truth points for nuclei1")
    
    # Filter for nuclei1 and the specific IDs
    nuclei1_data = []
    for pid in pixel_microns_ids:
        # Find rows that have this ID and nuclei1 in the csv_path
        mask = df['csv_path'].str.contains(pid, case=False, na=False) & \
               df['csv_path'].str.contains('nuclei1', case=False, na=False)
        filtered = df[mask].copy()
        if len(filtered) > 0:
            filtered['responseid'] = pid
            nuclei1_data.append(filtered)
            print(f"Found {len(filtered)} points for {pid} nuclei1")
    
    if len(nuclei1_data) == 0:
        print("No data found for pixel_microns_ids on nuclei1")
        return None
    
    # Combine all data
    df_plot = pd.concat(nuclei1_data, ignore_index=True)
    
    # Calculate LSA matching and errors for each ID
    print("\n" + "="*60)
    print("LSA Matching and Error Analysis (Raw Data)")
    print("="*60)
    
    lsa_results_raw = {}
    lsa_results_registered = {}
    registered_coords = {}
    
    for pid in pixel_microns_ids:
        data = df_plot[df_plot['responseid'] == pid]
        if len(data) > 0:
            # Get test and ground truth coordinates
            test_coords = data[['x', 'y', 'z']].values.astype('float64')
            gt_coords = gt_nuclei1[['x', 'y', 'z']].values.astype('float64')
            
            # Calculate distance matrix and perform LSA on raw data
            dm = distance_matrix(gt_coords, test_coords)
            row_ind, col_ind = linear_sum_assignment(dm)
            
            # Calculate matched distances (raw)
            displacement = gt_coords[row_ind] - test_coords[col_ind]
            distances = np.sqrt(np.sum(displacement**2, axis=1))
            
            # Calculate errors (raw)
            mean_error = np.mean(distances)
            mse = np.mean(distances**2)
            std_error = np.std(distances)
            
            lsa_results_raw[pid] = {
                'mean_error': mean_error,
                'mse': mse,
                'std_error': std_error,
                'min_distance': distances.min(),
                'max_distance': distances.max(),
                'num_matched': len(distances)
            }
            
            print(f"\n{pid} (Raw):")
            print(f"  Number of matched points: {len(distances)}")
            print(f"  Mean Error: {mean_error:.4f} um")
            print(f"  MSE: {mse:.4f} um²")
            print(f"  Std Error: {std_error:.4f} um")
            
            # Perform registration: calculate scale and translation separately for XY and Z
            # XY scale (using mean of X and Y standard deviations)
            scale_xy = gt_coords[:, :2].std(axis=0).mean() / test_coords[:, :2].std(axis=0).mean()
            # Z scale (separate)
            scale_z = gt_coords[:, 2].std() / test_coords[:, 2].std()
            
            # Apply XY scale first
            test_coords_scaled = test_coords.copy()
            test_coords_scaled[:, :2] = test_coords[:, :2] * scale_xy
            test_coords_scaled[:, 2] = test_coords[:, 2] * scale_z
            
            # Calculate translation after scaling
            translation = gt_coords.mean(axis=0) - test_coords_scaled.mean(axis=0)
            
            # Apply translation
            test_coords_registered = test_coords_scaled + translation
            registered_coords[pid] = test_coords_registered
            
            print(f"  Scale XY: {scale_xy:.4f}")
            print(f"  Scale Z: {scale_z:.4f}")
            print(f"  Translation: X={translation[0]:.4f}, Y={translation[1]:.4f}, Z={translation[2]:.4f}")
            
            # Calculate LSA on registered data
            dm_reg = distance_matrix(gt_coords, test_coords_registered)
            row_ind_reg, col_ind_reg = linear_sum_assignment(dm_reg)
            
            # Calculate matched distances (registered)
            displacement_reg = gt_coords[row_ind_reg] - test_coords_registered[col_ind_reg]
            distances_reg = np.sqrt(np.sum(displacement_reg**2, axis=1))
            
            # Calculate errors (registered)
            mean_error_reg = np.mean(distances_reg)
            mse_reg = np.mean(distances_reg**2)
            std_error_reg = np.std(distances_reg)
            
            lsa_results_registered[pid] = {
                'mean_error': mean_error_reg,
                'mse': mse_reg,
                'std_error': std_error_reg,
                'min_distance': distances_reg.min(),
                'max_distance': distances_reg.max(),
                'num_matched': len(distances_reg)
            }
            
            print(f"\n{pid} (After Registration):")
            print(f"  Mean Error: {mean_error_reg:.4f} um")
            print(f"  MSE: {mse_reg:.4f} um²")
            print(f"  Std Error: {std_error_reg:.4f} um")
            print(f"  Improvement: {((mse - mse_reg) / mse * 100):.1f}%")
    
    # Create figures directory if it doesn't exist
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Define colors - different color for each ID plus one for ground truth
    colors = {
        pixel_microns_ids[0]: 'green',
        pixel_microns_ids[1]: 'blue',
        'ground_truth': 'red'
    }
    
    # Define markers for each ID
    markers = {
        pixel_microns_ids[0]: 'o',  # circle
        pixel_microns_ids[1]: 's',  # square
        'ground_truth': 'x'
    }
    
    # Create the first plot (raw data)
    print("\nCreating raw data plot...")
    fig = plt.figure(figsize=figsize)
    
    # Create 2D plot (x-y view) - raw
    ax1 = fig.add_subplot(2, 2, 1)
    # Plot ground truth first (so it's in background)
    ax1.scatter(gt_nuclei1['x'], gt_nuclei1['y'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    # Plot each responseid
    for pid in pixel_microns_ids:
        data = df_plot[df_plot['responseid'] == pid]
        if len(data) > 0:
            ax1.scatter(data['x'], data['y'], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax1.set_xlabel('X (um)', fontsize=10)
    ax1.set_ylabel('Y (um)', fontsize=10)
    ax1.set_title('X-Y View - Nuclei1', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Create x-z view - raw
    ax2 = fig.add_subplot(2, 2, 2)
    # Plot ground truth
    ax2.scatter(gt_nuclei1['x'], gt_nuclei1['z'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    # Plot each responseid
    for pid in pixel_microns_ids:
        data = df_plot[df_plot['responseid'] == pid]
        if len(data) > 0:
            ax2.scatter(data['x'], data['z'], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax2.set_xlabel('X (um)', fontsize=10)
    ax2.set_ylabel('Z (um)', fontsize=10)
    ax2.set_title('X-Z View - Nuclei1', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Create y-z view - raw
    ax3 = fig.add_subplot(2, 2, 3)
    # Plot ground truth
    ax3.scatter(gt_nuclei1['y'], gt_nuclei1['z'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    # Plot each responseid
    for pid in pixel_microns_ids:
        data = df_plot[df_plot['responseid'] == pid]
        if len(data) > 0:
            ax3.scatter(data['y'], data['z'], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax3.set_xlabel('Y (um)', fontsize=10)
    ax3.set_ylabel('Z (um)', fontsize=10)
    ax3.set_title('Y-Z View - Nuclei1', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Create 3D plot - raw
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    # Plot ground truth
    ax4.scatter(gt_nuclei1['x'], gt_nuclei1['y'], gt_nuclei1['z'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    # Plot each responseid
    for pid in pixel_microns_ids:
        data = df_plot[df_plot['responseid'] == pid]
        if len(data) > 0:
            ax4.scatter(data['x'], data['y'], data['z'], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax4.set_xlabel('X (um)', fontsize=10)
    ax4.set_ylabel('Y (um)', fontsize=10)
    ax4.set_zlabel('Z (um)', fontsize=10)
    ax4.set_title('3D View - Nuclei1', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Save the raw plot
    plot_path = figures_dir / output_filename
    print(f"Saving raw plot to {plot_path}...")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print("Raw plot saved successfully!")
    
    # Create the second plot (registered data)
    print("\nCreating registered data plot...")
    fig_reg = plt.figure(figsize=figsize)
    
    # Create 2D plot (x-y view) - registered
    ax1 = fig_reg.add_subplot(2, 2, 1)
    ax1.scatter(gt_nuclei1['x'], gt_nuclei1['y'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    for pid in pixel_microns_ids:
        if pid in registered_coords:
            reg_coords = registered_coords[pid]
            ax1.scatter(reg_coords[:, 0], reg_coords[:, 1], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax1.set_xlabel('X (um)', fontsize=10)
    ax1.set_ylabel('Y (um)', fontsize=10)
    ax1.set_title('X-Y View - Nuclei1 (Registered)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Create x-z view - registered
    ax2 = fig_reg.add_subplot(2, 2, 2)
    ax2.scatter(gt_nuclei1['x'], gt_nuclei1['z'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    for pid in pixel_microns_ids:
        if pid in registered_coords:
            reg_coords = registered_coords[pid]
            ax2.scatter(reg_coords[:, 0], reg_coords[:, 2], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax2.set_xlabel('X (um)', fontsize=10)
    ax2.set_ylabel('Z (um)', fontsize=10)
    ax2.set_title('X-Z View - Nuclei1 (Registered)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Create y-z view - registered
    ax3 = fig_reg.add_subplot(2, 2, 3)
    ax3.scatter(gt_nuclei1['y'], gt_nuclei1['z'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    for pid in pixel_microns_ids:
        if pid in registered_coords:
            reg_coords = registered_coords[pid]
            ax3.scatter(reg_coords[:, 1], reg_coords[:, 2], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax3.set_xlabel('Y (um)', fontsize=10)
    ax3.set_ylabel('Z (um)', fontsize=10)
    ax3.set_title('Y-Z View - Nuclei1 (Registered)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Create 3D plot - registered
    ax4 = fig_reg.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(gt_nuclei1['x'], gt_nuclei1['y'], gt_nuclei1['z'], alpha=0.6, s=60, 
               color=colors['ground_truth'], label='Ground Truth', marker=markers['ground_truth'], linewidths=2)
    for pid in pixel_microns_ids:
        if pid in registered_coords:
            reg_coords = registered_coords[pid]
            ax4.scatter(reg_coords[:, 0], reg_coords[:, 1], reg_coords[:, 2], alpha=0.7, s=40, 
                       color=colors[pid], label=f'{pid}', marker=markers[pid])
    ax4.set_xlabel('X (um)', fontsize=10)
    ax4.set_ylabel('Y (um)', fontsize=10)
    ax4.set_zlabel('Z (um)', fontsize=10)
    ax4.set_title('3D View - Nuclei1 (Registered)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    # Save the registered plot
    registered_filename = output_filename.replace('.png', '_Registered.png')
    plot_path_reg = figures_dir / registered_filename
    print(f"Saving registered plot to {plot_path_reg}...")
    fig_reg.savefig(plot_path_reg, dpi=300, bbox_inches='tight')
    print("Registered plot saved successfully!")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"\nGround Truth (nuclei1):")
    print(f"  Number of points: {len(gt_nuclei1)}")
    print(f"  X range: {gt_nuclei1['x'].min():.2f} to {gt_nuclei1['x'].max():.2f}")
    print(f"  Y range: {gt_nuclei1['y'].min():.2f} to {gt_nuclei1['y'].max():.2f}")
    print(f"  Z range: {gt_nuclei1['z'].min():.2f} to {gt_nuclei1['z'].max():.2f}")
    
    for pid in pixel_microns_ids:
        data = df_plot[df_plot['responseid'] == pid]
        if len(data) > 0:
            print(f"\n{pid}:")
            print(f"  Number of points: {len(data)}")
            print(f"  X range: {data['x'].min():.2f} to {data['x'].max():.2f}")
            print(f"  Y range: {data['y'].min():.2f} to {data['y'].max():.2f}")
            print(f"  Z range: {data['z'].min():.2f} to {data['z'].max():.2f}")
    
    return fig


def create_all_nuclei_scatterplots(
    coordinate_data_file='coordinate_data_deidentified_nuclei.csv',
    ground_truth_file='ground_truth_coords.csv',
    combined_data_file='../data_analysis/combined data deidentified.csv',
    figsize=(16, 12)
):
    """
    Create scatter plots for all response IDs across all 4 nuclei datasets.
    Shows before and after registration for each ID.
    Each plot combines all 4 nuclei datasets into one visualization.
    
    Parameters:
    -----------
    coordinate_data_file : str
        Path to the coordinate data CSV file
    ground_truth_file : str
        Path to the ground truth CSV file
    combined_data_file : str
        Path to the combined data deidentified CSV file
    figsize : tuple
        Figure size for each plot
        
    Returns:
    --------
    None
    """
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Construct full paths
    data_path = script_dir / coordinate_data_file
    gt_path = script_dir / ground_truth_file
    combined_path = script_dir / combined_data_file
    
    # All unique response IDs from the attachment
    all_response_ids = [
        'R_3nu9kDs66l1O03t', 'R_2cC2iQifrFmuClP', 'R_3Rxaf07hES8CPgE',
        'R_1M6DoAmYEY3Jrvm', 'R_31bjqd6Mm8wBxN5', 'R_0638fJAPpmzLtu1',
        'R_2cCJjlMU7i9XjMQ', 'R_2c14tLfUPR1Vnua', 'R_22s3aTqiX7gbY4u',
        'R_3RxOa8kaiyul4bG', 'R_1GUZ4XruXifzoPp', 'R_tY87q7yGKRqww5X',
        'R_3lAJ9xY4kGlL99f', 'R_1lxxm7Riv8MvNVz', 'R_2DNRFrAvCDUX1EL',
        'R_3j9w1bGwWGd8yOC', 'R_cHLBH1bftzVtPDH', 'R_eRvCx60GZHVIflf',
        'R_3kpixa7Fm6wlFk2', 'R_24HIjcCJh6uI3bu', 'R_24Nwgngl83ucQ8B',
        'R_1F9A4K8LNlJoksJ', 'R_3kaX79RV1ul5JuI', 'R_2rCMx6wAGE7bFJh',
        'R_2q3KgZzjsLNwxgU', 'R_1qdHCwPCdNvs9xi', 'R_6E6PxT3N1gr93yh'
    ]
    
    # Nuclei dataset mapping
    nuclei_datasets = {
        'nuclei1': 'out_c00_dr90_label.tif',
        'nuclei2': 'out_c90_dr90_label.tif',
        'nuclei3': 'out_c00_dr10_label.tif',
        'nuclei4': 'out_c90_dr10_label.tif'
    }
    
    # Load data
    print(f"Loading coordinate data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loading ground truth data from {gt_path}...")
    df_gt = pd.read_csv(gt_path, index_col=0)
    
    # Load combined data to get software names
    print(f"Loading combined data from {combined_path}...")
    df_combined = pd.read_csv(combined_path)
    
    # Clean up qx.2 column names
    import re
    df_combined['qx.2_clean'] = df_combined['qx.2'].apply(
        lambda x: re.sub(r'\s*\([^)]*\)', '', str(x)).strip() if pd.notna(x) else x
    )
    
    # Create a mapping of responseid to software name
    software_map = df_combined.groupby('responseid')['qx.2_clean'].first().to_dict()
    
    # Create scatterplots directory
    scatterplots_dir = script_dir / 'figures' / 'scatterplots'
    scatterplots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Scatterplots directory: {scatterplots_dir.absolute()}")
    
    # Process each response ID
    for rid in all_response_ids:
        print(f"\n{'='*60}")
        print(f"Processing {rid}")
        print(f"{'='*60}")
        
        # Collect all data for this response ID across all nuclei
        all_gt_coords = []
        all_test_coords_raw = []
        all_test_coords_registered = []
        
        for nuclei_name, nuclei_file in nuclei_datasets.items():
            # Get ground truth for this nuclei dataset
            gt_coords = df_gt[df_gt['path'].str.contains(nuclei_file, case=False, na=False)][['x', 'y', 'z']].values.astype('float64')
            
            if len(gt_coords) == 0:
                print(f"  No ground truth found for {nuclei_name}")
                continue
            
            # Get test data for this ID and nuclei
            mask = df['csv_path'].str.contains(rid, case=False, na=False) & \
                   df['csv_path'].str.contains(nuclei_name, case=False, na=False)
            test_data = df[mask]
            
            if len(test_data) == 0:
                print(f"  No data found for {rid} {nuclei_name}")
                continue
            
            test_coords = test_data[['x', 'y', 'z']].values.astype('float64')
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(test_coords)) or np.any(~np.isfinite(gt_coords)):
                print(f"  WARNING: {nuclei_name} contains NaN or infinite values, skipping")
                continue
            
            print(f"  {nuclei_name}: {len(test_coords)} points")
            
            # Apply flipping for specific IDs (same as registration.py)
            flip_ids = ['R_3lAJ9xY4kGlL99f', 'R_31bjqd6Mm8wBxN5']
            if rid in flip_ids:
                print(f"  Applying XY flip for {rid}")
                test_coords[:, :2] = np.hstack([test_coords[:, 0].mean() - test_coords[:, [0]], test_coords[:, [1]]])
                test_coords[:, :2] = (np.array([[0.0, 1.0],
                                                [-1.0, 0.0]]) @ test_coords[:, :2].T).T
                
            if rid == 'R_6E6PxT3N1gr93yh':
                stop=5
            
            # Calculate registration
            try:
                scale_xy = gt_coords[:, :2].std(axis=0).mean() / test_coords[:, :2].std(axis=0).mean()
                scale_z = gt_coords[:, 2].std() / test_coords[:, 2].std()
                
                if not np.isfinite(scale_xy) or not np.isfinite(scale_z):
                    print(f"  WARNING: {nuclei_name} invalid scale values, skipping")
                    continue
                
                test_coords_scaled = test_coords.copy()
                test_coords_scaled[:, :2] = test_coords[:, :2] * scale_xy
                test_coords_scaled[:, 2] = test_coords[:, 2] * scale_z
                
                translation = gt_coords.mean(axis=0) - test_coords_scaled.mean(axis=0)
                test_coords_registered = test_coords_scaled + translation
                
                # Collect coordinates
                all_gt_coords.append(gt_coords)
                all_test_coords_raw.append(test_coords)
                all_test_coords_registered.append(test_coords_registered)
                
            except Exception as e:
                print(f"  ERROR during registration for {nuclei_name}: {e}")
                continue
        
        # Skip if no valid data
        if len(all_gt_coords) == 0:
            print(f"  No valid data found for {rid}, skipping")
            continue
        
        # Combine all nuclei data
        combined_gt = np.vstack(all_gt_coords)
        combined_raw = np.vstack(all_test_coords_raw)
        combined_registered = np.vstack(all_test_coords_registered)
        
        print(f"  Total points - GT: {len(combined_gt)}, Test: {len(combined_raw)}")
        
        # Create figure with 8 subplots (4 raw + 4 registered)
        fig = plt.figure(figsize=figsize)
        
        # Raw data plots (top 2 rows)
        # X-Y view
        ax1 = fig.add_subplot(2, 4, 1)
        ax1.scatter(combined_gt[:, 0], combined_gt[:, 1], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax1.scatter(combined_raw[:, 0], combined_raw[:, 1], alpha=0.7, s=30, 
                   color='blue', marker='o', label='Test')
        ax1.set_title('X-Y View (Raw)', fontsize=10, fontweight='bold')
        ax1.set_xlabel('X (um)', fontsize=8)
        ax1.set_ylabel('Y (um)', fontsize=8)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.3)
        
        # X-Z view
        ax2 = fig.add_subplot(2, 4, 2)
        ax2.scatter(combined_gt[:, 0], combined_gt[:, 2], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax2.scatter(combined_raw[:, 0], combined_raw[:, 2], alpha=0.7, s=30, 
                   color='blue', marker='o', label='Test')
        ax2.set_title('X-Z View (Raw)', fontsize=10, fontweight='bold')
        ax2.set_xlabel('X (um)', fontsize=8)
        ax2.set_ylabel('Z (um)', fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
        
        # Y-Z view
        ax3 = fig.add_subplot(2, 4, 3)
        ax3.scatter(combined_gt[:, 1], combined_gt[:, 2], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax3.scatter(combined_raw[:, 1], combined_raw[:, 2], alpha=0.7, s=30, 
                   color='blue', marker='o', label='Test')
        ax3.set_title('Y-Z View (Raw)', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Y (um)', fontsize=8)
        ax3.set_ylabel('Z (um)', fontsize=8)
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.3)
        
        # 3D view
        ax4 = fig.add_subplot(2, 4, 4, projection='3d')
        ax4.scatter(combined_gt[:, 0], combined_gt[:, 1], combined_gt[:, 2], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax4.scatter(combined_raw[:, 0], combined_raw[:, 1], combined_raw[:, 2], alpha=0.7, s=30, 
                   color='blue', marker='o', label='Test')
        ax4.set_title('3D View (Raw)', fontsize=10, fontweight='bold')
        ax4.set_xlabel('X', fontsize=7)
        ax4.set_ylabel('Y', fontsize=7)
        ax4.set_zlabel('Z', fontsize=7)
        ax4.legend(fontsize=7)
        
        # Registered data plots (bottom 2 rows)
        # X-Y view
        ax5 = fig.add_subplot(2, 4, 5)
        ax5.scatter(combined_gt[:, 0], combined_gt[:, 1], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax5.scatter(combined_registered[:, 0], combined_registered[:, 1], alpha=0.7, s=30, 
                   color='green', marker='o', label='Registered')
        ax5.set_title('X-Y View (Registered)', fontsize=10, fontweight='bold')
        ax5.set_xlabel('X (um)', fontsize=8)
        ax5.set_ylabel('Y (um)', fontsize=8)
        ax5.legend(fontsize=7)
        ax5.grid(True, alpha=0.3)
        
        # X-Z view
        ax6 = fig.add_subplot(2, 4, 6)
        ax6.scatter(combined_gt[:, 0], combined_gt[:, 2], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax6.scatter(combined_registered[:, 0], combined_registered[:, 2], alpha=0.7, s=30, 
                   color='green', marker='o', label='Registered')
        ax6.set_title('X-Z View (Registered)', fontsize=10, fontweight='bold')
        ax6.set_xlabel('X (um)', fontsize=8)
        ax6.set_ylabel('Z (um)', fontsize=8)
        ax6.legend(fontsize=7)
        ax6.grid(True, alpha=0.3)
        
        # Y-Z view
        ax7 = fig.add_subplot(2, 4, 7)
        ax7.scatter(combined_gt[:, 1], combined_gt[:, 2], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax7.scatter(combined_registered[:, 1], combined_registered[:, 2], alpha=0.7, s=30, 
                   color='green', marker='o', label='Registered')
        ax7.set_title('Y-Z View (Registered)', fontsize=10, fontweight='bold')
        ax7.set_xlabel('Y (um)', fontsize=8)
        ax7.set_ylabel('Z (um)', fontsize=8)
        ax7.legend(fontsize=7)
        ax7.grid(True, alpha=0.3)
        
        # 3D view
        ax8 = fig.add_subplot(2, 4, 8, projection='3d')
        ax8.scatter(combined_gt[:, 0], combined_gt[:, 1], combined_gt[:, 2], alpha=0.6, s=40, 
                   color='red', marker='x', linewidths=2, label='GT')
        ax8.scatter(combined_registered[:, 0], combined_registered[:, 1], combined_registered[:, 2], alpha=0.7, s=30, 
                   color='green', marker='o', label='Registered')
        ax8.set_title('3D View (Registered)', fontsize=10, fontweight='bold')
        ax8.set_xlabel('X', fontsize=7)
        ax8.set_ylabel('Y', fontsize=7)
        ax8.set_zlabel('Z', fontsize=7)
        ax8.legend(fontsize=7)
        
        fig.suptitle(f'{rid} - All Nuclei Combined', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Get software name for this response ID
        software_name = software_map.get(rid, 'Unknown')
        # Clean software name for filename (remove special characters)
        software_name_clean = re.sub(r'[^\w\s-]', '', software_name).replace(' ', '_')
        
        # Create filename with software name at beginning
        filename = f'{software_name_clean}_{rid}_all_nuclei.png'
        plot_path = scatterplots_dir / filename
        print(f"  Software: {software_name}")
        print(f"  Saving plot to {plot_path}...")
        
        # Update figure title to include software name at beginning
        fig.suptitle(f'{software_name} - {rid} - All Nuclei Combined', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    print(f"\n{'='*60}")
    print("All scatterplots created successfully!")
    print(f"{'='*60}")


def create_software_boxplots(
    coordinate_data_file='coordinate_data_deidentified_nuclei.csv',
    ground_truth_file='ground_truth_coords.csv',
    combined_data_file='../data_analysis/combined data deidentified.csv',
    output_filename='Software_LSA_MSE_Boxplot.png',
    figsize=(12, 8)
):
    """
    Create box plots of LSA MSE values grouped by software for nuclei data.
    Calculates LSA MSE with registration applied (same as scatter plots).
    
    Parameters:
    -----------
    coordinate_data_file : str
        Path to the coordinate data CSV file
    ground_truth_file : str
        Path to the ground truth CSV file
    combined_data_file : str
        Path to the combined data deidentified CSV file
    output_filename : str
        Filename to save the box plot
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    df_results : pd.DataFrame
        DataFrame with software and MSE values
    """
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Construct full paths
    data_path = script_dir / coordinate_data_file
    gt_path = script_dir / ground_truth_file
    combined_path = script_dir / combined_data_file
    
    # All unique response IDs
    all_response_ids = [
        'R_3nu9kDs66l1O03t', 'R_2cC2iQifrFmuClP', 'R_3Rxaf07hES8CPgE',
        'R_1M6DoAmYEY3Jrvm', 'R_31bjqd6Mm8wBxN5', 'R_0638fJAPpmzLtu1',
        'R_2cCJjlMU7i9XjMQ', 'R_2c14tLfUPR1Vnua', 'R_22s3aTqiX7gbY4u',
        'R_3RxOa8kaiyul4bG', 'R_1GUZ4XruXifzoPp', 'R_tY87q7yGKRqww5X',
        'R_3lAJ9xY4kGlL99f', 'R_1lxxm7Riv8MvNVz', 'R_2DNRFrAvCDUX1EL',
        'R_3j9w1bGwWGd8yOC', 'R_cHLBH1bftzVtPDH', 'R_eRvCx60GZHVIflf',
        'R_3kpixa7Fm6wlFk2', 'R_24HIjcCJh6uI3bu', 'R_24Nwgngl83ucQ8B',
        'R_1F9A4K8LNlJoksJ', 'R_3kaX79RV1ul5JuI', 'R_2rCMx6wAGE7bFJh',
        'R_2q3KgZzjsLNwxgU', 'R_1qdHCwPCdNvs9xi', 'R_6E6PxT3N1gr93yh'
    ]
    
    # Nuclei dataset mapping
    nuclei_datasets = {
        'nuclei1': 'out_c00_dr90_label.tif',
        'nuclei2': 'out_c90_dr90_label.tif',
        'nuclei3': 'out_c00_dr10_label.tif',
        'nuclei4': 'out_c90_dr10_label.tif'
    }
    
    # Load data
    print(f"Loading coordinate data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loading ground truth data from {gt_path}...")
    df_gt = pd.read_csv(gt_path, index_col=0)
    
    # Load combined data to get software names
    print(f"Loading combined data from {combined_path}...")
    df_combined = pd.read_csv(combined_path)
    
    # Clean up qx.2 column names
    import re
    df_combined['qx.2_clean'] = df_combined['qx.2'].apply(
        lambda x: re.sub(r'\s*\([^)]*\)', '', str(x)).strip() if pd.notna(x) else x
    )
    
    # Create a mapping of responseid to software name
    software_map = df_combined.groupby('responseid')['qx.2_clean'].first().to_dict()
    
    # Store results
    results = []
    
    # Process each response ID
    for rid in all_response_ids:
        print(f"Processing {rid}...")
        
        # Collect all data for this response ID across all nuclei
        all_gt_coords = []
        all_test_coords_registered = []
        
        for nuclei_name, nuclei_file in nuclei_datasets.items():
            # Get ground truth for this nuclei dataset
            gt_coords = df_gt[df_gt['path'].str.contains(nuclei_file, case=False, na=False)][['x', 'y', 'z']].values.astype('float64')
            
            if len(gt_coords) == 0:
                continue
            
            # Get test data for this ID and nuclei
            mask = df['csv_path'].str.contains(rid, case=False, na=False) & \
                   df['csv_path'].str.contains(nuclei_name, case=False, na=False)
            test_data = df[mask]
            
            if len(test_data) == 0:
                continue
            
            test_coords = test_data[['x', 'y', 'z']].values.astype('float64')
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(test_coords)) or np.any(~np.isfinite(gt_coords)):
                continue
            
            # Apply flipping for specific IDs
            flip_ids = ['R_3lAJ9xY4kGlL99f', 'R_31bjqd6Mm8wBxN5']
            if rid in flip_ids:
                test_coords[:, :2] = np.hstack([test_coords[:, 0].mean() - test_coords[:, [0]], test_coords[:, [1]]])
                test_coords[:, :2] = (np.array([[0.0, 1.0], [-1.0, 0.0]]) @ test_coords[:, :2].T).T
            
            # Calculate registration
            try:
                scale_xy = gt_coords[:, :2].std(axis=0).mean() / test_coords[:, :2].std(axis=0).mean()
                scale_z = gt_coords[:, 2].std() / test_coords[:, 2].std()
                
                if not np.isfinite(scale_xy) or not np.isfinite(scale_z):
                    continue
                
                test_coords_scaled = test_coords.copy()
                test_coords_scaled[:, :2] = test_coords[:, :2] * scale_xy
                test_coords_scaled[:, 2] = test_coords[:, 2] * scale_z
                
                translation = gt_coords.mean(axis=0) - test_coords_scaled.mean(axis=0)
                test_coords_registered = test_coords_scaled + translation
                
                # Collect coordinates
                all_gt_coords.append(gt_coords)
                all_test_coords_registered.append(test_coords_registered)
                
            except Exception:
                continue
        
        # Calculate LSA MSE if we have valid data
        if len(all_gt_coords) > 0:
            combined_gt = np.vstack(all_gt_coords)
            combined_registered = np.vstack(all_test_coords_registered)
            
            # Calculate LSA MSE
            dm_reg = distance_matrix(combined_gt, combined_registered)
            row_ind, col_ind = linear_sum_assignment(dm_reg)
            distances_reg = np.sqrt(np.sum((combined_gt[row_ind] - combined_registered[col_ind])**2, axis=1))
            mse_reg = np.mean(distances_reg**2)
            
            # Get software name
            software_name = software_map.get(rid, 'Unknown')
            
            # Store result
            results.append({
                'responseid': rid,
                'software': software_name,
                'lsa_mse_registered': mse_reg
            })
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Filter out Unknown software
    df_results = df_results[df_results['software'] != 'Unknown']
    
    print(f"\nCalculated LSA MSE for {len(df_results)} submissions")
    
    # Create figures directory if it doesn't exist
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Create box plot
    print("Creating box plot...")
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Use seaborn for better box plots
    import seaborn as sns
    sns.boxplot(data=df_results, x='software', y='lsa_mse_registered', ax=ax, 
                color='lightblue', showfliers=False)
    sns.stripplot(data=df_results, x='software', y='lsa_mse_registered', ax=ax,
                  color='red', alpha=0.5, size=6, jitter=True)
    
    ax.set_yscale('log')
    ax.set_title('LSA MSE (Registered) by Software - Nuclei', fontsize=14, fontweight='bold')
    ax.set_xlabel('Software', fontsize=12)
    ax.set_ylabel('LSA MSE Registered (um²) - Log Scale', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = figures_dir / output_filename
    print(f"Saving box plot to {plot_path}...")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print("Box plot saved successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics by Software")
    print("="*60)
    for software in sorted(df_results['software'].unique()):
        group_data = df_results[df_results['software'] == software]
        print(f"\n{software} (n={len(group_data)})")
        print(f"  Mean LSA MSE: {group_data['lsa_mse_registered'].mean():.4f} um²")
        print(f"  Median LSA MSE: {group_data['lsa_mse_registered'].median():.4f} um²")
        print(f"  Std LSA MSE: {group_data['lsa_mse_registered'].std():.4f} um²")
    
    return fig, df_results


def create_software_boxplots_per_nuclei(
    coordinate_data_file='coordinate_data_deidentified_nuclei.csv',
    ground_truth_file='ground_truth_coords.csv',
    combined_data_file='../data_analysis/combined data deidentified.csv',
    output_filename='Software_LSA_MSE_Boxplot_Per_Nuclei.png',
    figsize=(12, 8)
):
    """
    Create box plots of LSA MSE values grouped by software for nuclei data.
    Calculates LSA MSE separately for EACH nuclei dataset (4 points per response ID).
    
    Parameters:
    -----------
    coordinate_data_file : str
        Path to the coordinate data CSV file
    ground_truth_file : str
        Path to the ground truth CSV file
    combined_data_file : str
        Path to the combined data deidentified CSV file
    output_filename : str
        Filename to save the box plot
    figsize : tuple
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    df_results : pd.DataFrame
        DataFrame with software, nuclei, and MSE values
    """
    
    # Get the script directory
    script_dir = Path(__file__).parent
    
    # Construct full paths
    data_path = script_dir / coordinate_data_file
    gt_path = script_dir / ground_truth_file
    combined_path = script_dir / combined_data_file
    
    # All unique response IDs
    all_response_ids = [
        'R_3nu9kDs66l1O03t', 'R_2cC2iQifrFmuClP', 'R_3Rxaf07hES8CPgE',
        'R_1M6DoAmYEY3Jrvm', 'R_31bjqd6Mm8wBxN5', 'R_0638fJAPpmzLtu1',
        'R_2cCJjlMU7i9XjMQ', 'R_2c14tLfUPR1Vnua', 'R_22s3aTqiX7gbY4u',
        'R_3RxOa8kaiyul4bG', 'R_1GUZ4XruXifzoPp', 'R_tY87q7yGKRqww5X',
        'R_3lAJ9xY4kGlL99f', 'R_1lxxm7Riv8MvNVz', 'R_2DNRFrAvCDUX1EL',
        'R_3j9w1bGwWGd8yOC', 'R_cHLBH1bftzVtPDH', 'R_eRvCx60GZHVIflf',
        'R_3kpixa7Fm6wlFk2', 'R_24HIjcCJh6uI3bu', 'R_24Nwgngl83ucQ8B',
        'R_1F9A4K8LNlJoksJ', 'R_3kaX79RV1ul5JuI', 'R_2rCMx6wAGE7bFJh',
        'R_2q3KgZzjsLNwxgU', 'R_1qdHCwPCdNvs9xi', 'R_6E6PxT3N1gr93yh'
    ]
    
    # Nuclei dataset mapping
    nuclei_datasets = {
        'nuclei1': 'out_c00_dr90_label.tif',
        'nuclei2': 'out_c90_dr90_label.tif',
        'nuclei3': 'out_c00_dr10_label.tif',
        'nuclei4': 'out_c90_dr10_label.tif'
    }
    
    # Load data
    print(f"Loading coordinate data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loading ground truth data from {gt_path}...")
    df_gt = pd.read_csv(gt_path, index_col=0)
    
    # Load combined data to get software names
    print(f"Loading combined data from {combined_path}...")
    df_combined = pd.read_csv(combined_path)
    
    # Clean up qx.2 column names
    import re
    df_combined['qx.2_clean'] = df_combined['qx.2'].apply(
        lambda x: re.sub(r'\s*\([^)]*\)', '', str(x)).strip() if pd.notna(x) else x
    )
    
    # Create a mapping of responseid to software name
    software_map = df_combined.groupby('responseid')['qx.2_clean'].first().to_dict()
    
    # Store results - one row per nuclei per response ID
    results = []
    
    # Process each response ID
    for rid in all_response_ids:
        print(f"Processing {rid}...")
        
        # Get software name
        software_name = software_map.get(rid, 'Unknown')
        
        if software_name == 'Unknown':
            continue
        
        # Process EACH nuclei dataset separately
        for nuclei_name, nuclei_file in nuclei_datasets.items():
            # Get ground truth for this nuclei dataset
            gt_coords = df_gt[df_gt['path'].str.contains(nuclei_file, case=False, na=False)][['x', 'y', 'z']].values.astype('float64')
            
            if len(gt_coords) == 0:
                continue
            
            # Get test data for this ID and nuclei
            mask = df['csv_path'].str.contains(rid, case=False, na=False) & \
                   df['csv_path'].str.contains(nuclei_name, case=False, na=False)
            test_data = df[mask]
            
            if len(test_data) == 0:
                continue
            
            test_coords = test_data[['x', 'y', 'z']].values.astype('float64')
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(test_coords)) or np.any(~np.isfinite(gt_coords)):
                continue
            
            # Apply flipping for specific IDs
            flip_ids = ['R_3lAJ9xY4kGlL99f', 'R_31bjqd6Mm8wBxN5']
            if rid in flip_ids:
                test_coords[:, :2] = np.hstack([test_coords[:, 0].mean() - test_coords[:, [0]], test_coords[:, [1]]])
                test_coords[:, :2] = (np.array([[0.0, 1.0], [-1.0, 0.0]]) @ test_coords[:, :2].T).T
            
            # Calculate registration
            try:
                scale_xy = gt_coords[:, :2].std(axis=0).mean() / test_coords[:, :2].std(axis=0).mean()
                scale_z = gt_coords[:, 2].std() / test_coords[:, 2].std()
                
                if not np.isfinite(scale_xy) or not np.isfinite(scale_z):
                    continue
                
                test_coords_scaled = test_coords.copy()
                test_coords_scaled[:, :2] = test_coords[:, :2] * scale_xy
                test_coords_scaled[:, 2] = test_coords[:, 2] * scale_z
                
                translation = gt_coords.mean(axis=0) - test_coords_scaled.mean(axis=0)
                test_coords_registered = test_coords_scaled + translation
                
                # Calculate LSA for registered data
                dm_reg = distance_matrix(gt_coords, test_coords_registered)
                row_ind_reg, col_ind_reg = linear_sum_assignment(dm_reg)
                
                # Calculate matched distances (registered)
                displacement_reg = gt_coords[row_ind_reg] - test_coords_registered[col_ind_reg]
                distances_reg = np.sqrt(np.sum(displacement_reg**2, axis=1))
                
                # Calculate errors (registered)
                mean_error_reg = np.mean(distances_reg)
                mse_reg = np.mean(distances_reg**2)
                std_error_reg = np.std(distances_reg)
                
                # Store result for this specific nuclei
                results.append({
                    'responseid': rid,
                    'software': software_name,
                    'nuclei': nuclei_name,
                    'lsa_mse_registered': mse_reg
                })
                
            except Exception as e:
                print(f"  ERROR for {rid} {nuclei_name}: {e}")
                continue
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\nCalculated LSA MSE for {len(df_results)} nuclei-response combinations")
    
    # Create figures directory if it doesn't exist
    figures_dir = script_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # Create box plot
    print("Creating per-nuclei box plot...")
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Use seaborn for better box plots
    import seaborn as sns
    sns.boxplot(data=df_results, x='software', y='lsa_mse_registered', ax=ax, 
                color='lightblue', showfliers=False)
    sns.stripplot(data=df_results, x='software', y='lsa_mse_registered', ax=ax,
                  color='red', alpha=0.5, size=4, jitter=True)
    
    ax.set_yscale('log')
    ax.set_title('LSA MSE (Registered) by Software - Per Nuclei Dataset', fontsize=14, fontweight='bold')
    ax.set_xlabel('Software', fontsize=12)
    ax.set_ylabel('LSA MSE Registered (um²) - Log Scale', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = figures_dir / output_filename
    print(f"Saving per-nuclei box plot to {plot_path}...")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print("Per-nuclei box plot saved successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics by Software (Per Nuclei)")
    print("="*60)
    for software in sorted(df_results['software'].unique()):
        group_data = df_results[df_results['software'] == software]
        print(f"\n{software} (n={len(group_data)} nuclei datasets)")
        print(f"  Mean LSA MSE: {group_data['lsa_mse_registered'].mean():.4f} um²")
        print(f"  Median LSA MSE: {group_data['lsa_mse_registered'].median():.4f} um²")
        print(f"  Std LSA MSE: {group_data['lsa_mse_registered'].std():.4f} um²")
    
    return fig, df_results


if __name__ == '__main__':
    # Example usage
    # fig = plot_pixel_microns_points()
    # plt.show()
    
    # Create all scatter plots
    create_all_nuclei_scatterplots()
    
    # Create software box plot (averaged across all nuclei)
    fig_box, df_results = create_software_boxplots()
    
    # Create software box plot (per nuclei dataset)
    fig_box_per_nuclei, df_results_per_nuclei = create_software_boxplots_per_nuclei()
    
    plt.show()
