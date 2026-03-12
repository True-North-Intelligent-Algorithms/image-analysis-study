"""
Plotting functions for point registration analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path


def create_lsa_mse_grouped_plots(
    registration_stats_file='registration_stats.csv',
    combined_data_file='../data_analysis/combined data deidentified.csv',
    output_filename_raw='LSA MSE Raw - Grouped by QX.2.png',
    output_filename_transformed='LSA MSE Transformed - Grouped by QX.2.png',
    figsize=(10, 8)
):
    """
    Create and save two separate LSA MSE plots grouped by QX.2 for nuclei data.
    
    This function performs the following steps:
    1. Loads registration stats and combines raw_data and fully_transformed analysis levels
    2. Merges with combined data deidentified CSV
    3. Cleans up QX.2 column names by removing parentheses
    4. Filters for nuclei only
    5. Creates two separate box plots with strip plots overlay for lsa_mse_raw and lsa_mse_transformed
    6. Saves each plot to separate files
    
    Parameters:
    -----------
    registration_stats_file : str
        Path to the registration stats CSV file
    combined_data_file : str
        Path to the combined data deidentified CSV file
    output_filename_raw : str
        Filename to save the raw MSE plot (default: 'LSA MSE Raw - Grouped by QX.2.png')
    output_filename_transformed : str
        Filename to save the transformed MSE plot (default: 'LSA MSE Transformed - Grouped by QX.2.png')
    figsize : tuple
        Figure size (width, height) in inches for each individual plot
        
    Returns:
    --------
    fig_raw : matplotlib.figure.Figure
        The created figure object for raw MSE plot
    fig_transformed : matplotlib.figure.Figure
        The created figure object for transformed MSE plot
    df_nuclei_analysis : pd.DataFrame
        The filtered nuclei analysis dataframe used for plotting
    """
    
    # Step 1: Load registration stats CSV
    print("Loading registration stats...")
    df = pd.read_csv(registration_stats_file)
    
    # Step 2: Combine raw_data and fully_transformed rows
    print("Combining raw_data and fully_transformed analysis levels...")
    df_raw = df[df['analysis_level'] == 'raw_data'].copy()
    df_transformed = df[df['analysis_level'] == 'fully_transformed'].copy()
    
    # Columns to take only from fully_transformed
    transform_only_cols = ['scale_xy', 'translation_x', 'translation_y', 'angle_xy']
    
    # Columns to take from both
    metrics_cols = ['lsa_mean (um)', 'lsa_std (um)', 'nn_mean (um)', 'nn_std (um)', 'lsa_mse (um^2)']
    
    # Rename the metrics columns
    df_raw_renamed = df_raw.rename(columns={col: f'raw_{col}' for col in metrics_cols})
    df_transformed_renamed = df_transformed.rename(columns={col: f'transformed_{col}' for col in metrics_cols})
    
    # Select columns for merge
    raw_cols = ['id', 'ground_truth_name'] + [f'raw_{col}' for col in metrics_cols]
    df_raw_selected = df_raw_renamed[raw_cols]
    
    transformed_cols = ['id', 'ground_truth_name'] + transform_only_cols + [f'transformed_{col}' for col in metrics_cols]
    df_transformed_selected = df_transformed_renamed[transformed_cols]
    
    # Merge on id and ground_truth_name
    df_combined = pd.merge(
        df_raw_selected,
        df_transformed_selected,
        on=['id', 'ground_truth_name'],
        how='inner'
    )
    
    # Rename LSA columns to the desired format
    rename_mapping = {
        'raw_lsa_mean (um)': 'lsa_mean_raw',
        'transformed_lsa_mean (um)': 'lsa_mean_transformed',
        'raw_lsa_std (um)': 'lsa_std_raw',
        'transformed_lsa_std (um)': 'lsa_std_transformed',
        'raw_lsa_mse (um^2)': 'lsa_mse_raw',
        'transformed_lsa_mse (um^2)': 'lsa_mse_transformed'
    }
    df_combined = df_combined.rename(columns=rename_mapping)
    
    print(f"Combined dataframe rows: {len(df_combined)}")
    
    # Step 3: Load and update combined data deidentified CSV
    print("Loading combined data deidentified CSV...")
    df_data_analysis = pd.read_csv(combined_data_file)
    
    update_cols = ['lsa_mean_raw', 'lsa_mean_transformed', 'lsa_std_raw', 
                   'lsa_std_transformed', 'lsa_mse_raw', 'lsa_mse_transformed']
    
    # Merge with df_combined to get the new values
    df_combined_for_merge = df_combined[['id', 'ground_truth_name'] + update_cols].copy()
    df_combined_for_merge = df_combined_for_merge.rename(columns={
        'id': 'responseid',
        'ground_truth_name': 'setName'
    })
    
    # Drop old columns and merge with new values
    cols_to_drop = [col for col in update_cols if col in df_data_analysis.columns]
    df_data_analysis_without_old = df_data_analysis.drop(columns=cols_to_drop)
    
    df_data_analysis_updated = pd.merge(
        df_data_analysis_without_old,
        df_combined_for_merge,
        on=['responseid', 'setName'],
        how='left'
    )
    
    # Step 4: Clean up qx.2 column names
    print("Cleaning QX.2 column names...")
    df_data_analysis_updated['qx.2_clean'] = df_data_analysis_updated['qx.2'].apply(
        lambda x: re.sub(r'\s*\([^)]*\)', '', str(x)).strip() if pd.notna(x) else x
    )
    
    # Step 5: Filter for nuclei only
    print("Filtering for nuclei only...")
    df_nuclei_analysis = df_data_analysis_updated[
        df_data_analysis_updated['setName'].str.contains('nuclei', case=False, na=False)
    ].copy()
    
    print(f"Nuclei rows: {len(df_nuclei_analysis)}")
    
    # Create figures directory if it doesn't exist
    figures_dir = Path('figures')
    figures_dir.mkdir(exist_ok=True)
    print(f"Figures directory: {figures_dir.absolute()}")
    
    # Step 6: Create the first plot - LSA MSE Raw
    print("Creating LSA MSE Raw plot...")
    fig_raw, ax_raw = plt.subplots(1, 1, figsize=figsize)
    
    sns.boxplot(data=df_nuclei_analysis, x='qx.2_clean', y='lsa_mse_raw', ax=ax_raw, color='lightblue')
    sns.stripplot(data=df_nuclei_analysis, x='qx.2_clean', y='lsa_mse_raw', ax=ax_raw, 
                  color='red', alpha=0.5, size=4, jitter=True)
    ax_raw.set_yscale('log')
    ax_raw.set_title('LSA MSE Raw - Grouped by QX.2 (Nuclei Only)', fontsize=14, fontweight='bold')
    ax_raw.set_xlabel('QX.2', fontsize=12)
    ax_raw.set_ylabel('LSA MSE Raw (um²) - Log Scale', fontsize=12)
    ax_raw.grid(True, alpha=0.3, which='both')
    ax_raw.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the raw plot in figures directory
    raw_plot_path = figures_dir / output_filename_raw
    print(f"Saving raw plot to {raw_plot_path}...")
    fig_raw.savefig(raw_plot_path, dpi=300, bbox_inches='tight')
    print("Raw plot saved successfully!")
    
    # Step 7: Create the second plot - LSA MSE Transformed
    print("Creating LSA MSE Transformed plot...")
    fig_transformed, ax_transformed = plt.subplots(1, 1, figsize=figsize)
    
    sns.boxplot(data=df_nuclei_analysis, x='qx.2_clean', y='lsa_mse_transformed', ax=ax_transformed, color='lightblue')
    sns.stripplot(data=df_nuclei_analysis, x='qx.2_clean', y='lsa_mse_transformed', ax=ax_transformed, 
                  color='red', alpha=0.5, size=4, jitter=True)
    ax_transformed.set_yscale('log')
    ax_transformed.set_title('LSA MSE Transformed - Grouped by QX.2 (Nuclei Only)', fontsize=14, fontweight='bold')
    ax_transformed.set_xlabel('QX.2', fontsize=12)
    ax_transformed.set_ylabel('LSA MSE Transformed (um²) - Log Scale', fontsize=12)
    ax_transformed.grid(True, alpha=0.3, which='both')
    ax_transformed.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the transformed plot in figures directory
    transformed_plot_path = figures_dir / output_filename_transformed
    print(f"Saving transformed plot to {transformed_plot_path}...")
    fig_transformed.savefig(transformed_plot_path, dpi=300, bbox_inches='tight')
    print("Transformed plot saved successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics by QX.2")
    print("="*60)
    for qx_value in sorted(df_nuclei_analysis['qx.2_clean'].dropna().unique()):
        group_data = df_nuclei_analysis[df_nuclei_analysis['qx.2_clean'] == qx_value]
        print(f"\nQX.2 = {qx_value} (n={len(group_data)})")
        print(f"  LSA MSE Raw - Mean: {group_data['lsa_mse_raw'].mean():.4f}, Median: {group_data['lsa_mse_raw'].median():.4f}")
        print(f"  LSA MSE Transformed - Mean: {group_data['lsa_mse_transformed'].mean():.4f}, Median: {group_data['lsa_mse_transformed'].median():.4f}")
    
    return fig_raw, fig_transformed, df_nuclei_analysis


if __name__ == '__main__':
    # Example usage
    fig_raw, fig_transformed, df_nuclei = create_lsa_mse_grouped_plots()
    plt.show()
