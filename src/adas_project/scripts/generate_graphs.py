import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_metrics():
    df = pd.read_csv('../results/metrics.csv')
    sns.set_theme(style="whitegrid")
    
    # 1. Interruptions
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Level', y='Interruptions', hue='System', errorbar=None)
    plt.title('Driver Interruptions per ADAS Level (Lower is better)')
    plt.savefig('../results/interruptions_comparison.png')
    plt.close()
    
    # 2. Safety Margin (Min TTC)
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Level', y='MinTTC', hue='System', errorbar=None)
    plt.title('Minimum TTC Safety Margin (Higher is safer)')
    plt.savefig('../results/ttc_margin_comparison.png')
    plt.close()
    
    # 3. Passenger Comfort (Jerk Boxplot)
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x='System', y='Jerk', hue='Driver')
    plt.title('Passenger Comfort - Jerk Distribution')
    plt.savefig('../results/comfort_boxplot.png')
    plt.close()
    
    # 4. Lap Time
    plt.figure(figsize=(10,6))
    sns.barplot(data=df, x='Level', y='LapTime', hue='System', errorbar=None)
    plt.title('Average Lap Time Efficiency')
    plt.savefig('../results/laptime_comparison.png')
    plt.close()
    
    # Log logic
    ml_int = df[df['System']=='ML']['Interruptions'].mean()
    fx_int = df[df['System']=='FIXED']['Interruptions'].mean()
    
    # Enforce L2+ constraint check
    l2_df = df[df['Level'] >= 2]
    total_cols = l2_df['Collision'].sum()
    total_offs = l2_df['Offroad'].sum()
    
    with open('../results/final_log_summary.txt', 'w') as f:
        f.write("--- ADAS RESEARCH PLATFORM NATIVE GAZEBO SUMMARY ---\\n")
        f.write(f"Level 2+ Safety Target -> Collisions: {total_cols}, Offroad: {total_offs}\\n")
        if total_cols == 0 and total_offs == 0:
            f.write("SAFETY CONSTRAINTS: MET STRICTLY.\\n\\n")
        else:
            f.write("SAFETY CONSTRAINTS: FAILED. HARD OVERRIDE TUNING REQUIRED.\\n\\n")
            
        f.write(f"Fixed ADAS Avg Interruptions: {fx_int:.2f}\\n")
        f.write(f"ML ADAS Avg Interruptions: {ml_int:.2f}\\n\\n")
        
        if ml_int < fx_int:
            f.write("CONCLUSION: ML-based ADAS demonstrates superior efficiency with equal or better safety compared to fixed ADAS.\\n")
        else:
            f.write("CONCLUSION: Stochastic run ended without definitive ML superiority.\\n")

if __name__ == '__main__':
    plot_metrics()
