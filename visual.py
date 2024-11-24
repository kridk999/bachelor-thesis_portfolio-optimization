import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def visual_main(base_path='CVAR_results'):
    # Define paths for result folders and visualization output

    visualizations_path = os.path.join(base_path, 'visualizations')
    os.makedirs(visualizations_path, exist_ok=True)  # Ensure the visualizations folder exists

    result_folders = {
        '2021': ['2021_1', '2021_2', '2021_random', '2021_baseline'],
        '2022': ['2022_1', '2022_2', '2022_random', '2022_baseline'],
        '2023': ['2023_1', '2023_2', '2023_random', '2023_baseline'],

    }

    # Load simulation data and allocation data
    def load_simulation_data(folder):
        if base_path == 'LSTM_results' or base_path == 'LSTM_tests':
            simulation_file = [file for file in os.listdir(folder) if file.startswith('LSTM')][0]
        else:
            simulation_file = [file for file in os.listdir(folder) if file.startswith('simulation')][0]
        return pd.read_csv(os.path.join(folder, simulation_file))

    def load_allocation_data(folder):
        if base_path == 'LSTM_results' or base_path == 'LSTM_tests':
            results_file = [file for file in os.listdir(folder) if file.startswith('allocation')][0]
        else:
            results_file = [file for file in os.listdir(folder) if file.startswith('results')][0]
        return pd.read_csv(os.path.join(folder, results_file))

    # Baseline Average Return Calculation
    def calculate_baseline_average_return(file_path, start_date, end_date):
        returns_df = pd.read_csv(file_path, parse_dates=['date'])
        returns_df.set_index('date', inplace=True)
        returns_df = returns_df[start_date:end_date]
        cumulative_returns = (returns_df + 1).prod(axis=0) - 1
        total_years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
        annualized_return = (1 + cumulative_returns.mean()) ** (1 / total_years) - 1
        return annualized_return


    # Plot Portfolio Values with Baseline for Each Year and save the plot
    def plot_portfolio_with_baseline(year, folder1, folder3, folder4, baseline_return, baseline_file_path):
        # Load data
        folder1_data = load_simulation_data(folder1)
        folder3_data = load_simulation_data(folder3)
        folder4_data = load_simulation_data(folder4)
        
        # Set initial values and dates
        dates = pd.to_datetime(folder1_data['Date'])
        portfolio_start = folder1_data['PortfolioValue'].iloc[0]
        num_days = (dates - dates.min()).dt.days

        # Baseline values
        baseline_values = portfolio_start * (1 + baseline_return) ** (num_days / 365)


        # Plot portfolios and baselines
        plt.figure(figsize=(12, 8))
        plt.plot(dates, folder1_data['PortfolioValue'], label=f"{os.path.basename(folder1)}", color='blue')
        plt.plot(dates, folder3_data['PortfolioValue'], label=f"{os.path.basename(folder3)}", color='purple')
        plt.plot(dates, folder4_data['PortfolioValue'], label=f"{os.path.basename(folder4)}", color='red')
        plt.plot(dates, baseline_values, label="Baseline Average Return", linestyle='--', color='red')
        
        # Formatting the x-axis
        plt.title(f"Portfolio Value Comparison - {year}")
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to the visualizations folder
        plot_filename = os.path.join(visualizations_path, f"Portfolio_Comparison_{year}.png")
        plt.savefig(plot_filename)
        plt.close()

    # Plot Top 10 Allocations as Heatmap and save the plot
    def plot_top_allocations_heatmap(folder, title):
        allocations_df = load_allocation_data(folder)
        
        # Retrieve asset names from the simulation file
        simulation_data = load_simulation_data(folder)
        asset_names = simulation_data.columns[2:]  # Skip 'Date' and 'PortfolioValue' columns

        # Convert allocations strings to lists and create a DataFrame
        allocations = allocations_df['Allocations'].apply(lambda x: eval(x))
        allocations_df = pd.DataFrame(allocations.tolist(), index=allocations_df['Date'], columns=asset_names)
        
        
        # Calculate the average allocation for each asset and select the top 10
        average_allocations = allocations_df.mean().sort_values(ascending=False).head(10)
        top_10_assets = average_allocations.index
        top_allocations_df = allocations_df[top_10_assets]

        # Plot heatmap with improved styling
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            top_allocations_df.T, 
            cmap="YlGnBu", 
            cbar_kws={'label': 'Average Allocation'}, 
            fmt=".2f", 
            linewidths=0.5, 
            linecolor='lightgrey',
            annot=False  # Turn off cell annotations for a cleaner look
        )
        
        plt.title(f'Top 10 Allocations Heatmap - {title}', fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Top Assets", fontsize=12)
        
        # Rotate and improve x-tick labels
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)  # Set y-axis labels horizontal for readability
        plt.tight_layout()

        # Save the plot to the visualizations folder
        plot_filename = os.path.join(visualizations_path, f"Top_Allocations_Heatmap_{title}.png")
        plt.savefig(plot_filename)
        plt.close()

    # Run visualizations for each year
    def run_visualizations():
        baseline_file_path = 'data/grouped_data_return_daily.csv'
        
        for year, folders in result_folders.items():
            start_date = f'{year[:4]}-01-01'
            end_date = f'{year[-4:]}-12-31' if '2021-2023' not in year else '2023-12-31'

            # Calculate baseline average return
            baseline_return = calculate_baseline_average_return(baseline_file_path, start_date, end_date)

            # Plot each pair of models with the baseline for the year
            folder1_path = os.path.join(base_path, folders[0])
            #folder2_path = os.path.join(base_path, folders[1])
            folder3_path = os.path.join(base_path, folders[2])
            folder4_path = os.path.join(base_path, folders[3])
            
            plot_portfolio_with_baseline(year, folder1_path,  folder3_path, folder4_path, baseline_return, baseline_file_path)

    # Run allocation heatmap for each year
    def run_heatmaps():
        for year, folders in result_folders.items():
            for folder in folders:
                folder_path = os.path.join(base_path, folder)
                title = f"{folder} - {year}"
                plot_top_allocations_heatmap(folder_path, title)
 

    run_visualizations()
    run_heatmaps()

if __name__ == "__main__":

    visual_main()

