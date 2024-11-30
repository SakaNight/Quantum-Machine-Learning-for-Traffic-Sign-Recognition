import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_class_accuracy(eval_results_path, output_dir="quantum_data/analysis"):
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.read_csv(eval_results_path)

    class_results = results_df[~results_df['Class'].isin(['macro_avg', 'weighted_avg'])]
    
    class_names = {
        0:'University of Waterloo', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
        3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
        6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
        9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
        12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
        16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
        19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
        22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
        25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
        29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
        32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
        35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
        38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
        41:'End of no passing', 42:'End no passing veh > 3.5 tons'
    }

    class_results['Class_Name'] = class_results['Class'].map(class_names)
    
    stats = {
        'Average Accuracy': class_results['Accuracy'].mean(),
        'Median Accuracy': class_results['Accuracy'].median(),
        'Std Accuracy': class_results['Accuracy'].std(),
        'Min Accuracy': class_results['Accuracy'].min(),
        'Max Accuracy': class_results['Accuracy'].max()
    }
    
    best_classes = class_results.nlargest(5, 'Accuracy')[['Class', 'Class_Name', 'Accuracy']]
    worst_classes = class_results.nsmallest(5, 'Accuracy')[['Class', 'Class_Name', 'Accuracy']]
    
    with open(os.path.join(output_dir, 'accuracy_analysis_report.txt'), 'w') as f:
        f.write("Classification Accuracy Analysis Report\n")
        f.write("=====================================\n\n")
        
        f.write("Overall Statistics:\n")
        f.write("-----------------\n")
        for stat_name, value in stats.items():
            f.write(f"{stat_name}: {value:.4f}\n")
        f.write("\n")
        
        f.write("Top 5 Best Performing Classes:\n")
        f.write("---------------------------\n")
        for _, row in best_classes.iterrows():
            f.write(f"Class {int(row['Class'])} ({row['Class_Name']}): {row['Accuracy']:.4f}\n")
        f.write("\n")
        
        f.write("Top 5 Worst Performing Classes:\n")
        f.write("----------------------------\n")
        for _, row in worst_classes.iterrows():
            f.write(f"Class {int(row['Class'])} ({row['Class_Name']}): {row['Accuracy']:.4f}\n")
    
    plt.figure(figsize=(20, 10))
    sns.barplot(data=class_results, x='Class', y='Accuracy')
    plt.title('Classification Accuracy by Class')
    plt.xlabel('Class ID')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy_distribution.png'))
    plt.close()
    
    plt.figure(figsize=(15, 8))
    accuracy_data = class_results.pivot_table(index='Class', values='Accuracy')
    sns.heatmap(accuracy_data.T, cmap='RdYlGn', annot=True, fmt='.3f')
    plt.title('Classification Accuracy Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'))
    plt.close()
    
    detailed_results = class_results[['Class', 'Class_Name', 'Accuracy', 'Precision', 'Recall', 'F1', 'Support']]
    detailed_results.to_csv(os.path.join(output_dir, 'detailed_class_accuracy.csv'), index=False)
    
    return stats, best_classes, worst_classes

def main():
    eval_results_path = "quantum_data/evaluation_results/evaluation_results_classical_20241126_001917.csv"
    output_dir = "quantum_data/analysis"
    
    stats, best_classes, worst_classes = analyze_class_accuracy(eval_results_path, output_dir)
    
    print("\nAnalysis Results:")
    print("\nOverall Statistics:")
    for stat_name, value in stats.items():
        print(f"{stat_name}: {value:.4f}")
    
    print("\nTop 5 Best Performing Classes:")
    for _, row in best_classes.iterrows():
        print(f"Class {int(row['Class'])} ({row['Class_Name']}): {row['Accuracy']:.4f}")
    
    print("\nTop 5 Worst Performing Classes:")
    for _, row in worst_classes.iterrows():
        
        print(f"Class {int(row['Class'])} ({row['Class_Name']}): {row['Accuracy']:.4f}")
    
    print(f"\nDetailed results have been saved to {output_dir}")

if __name__ == "__main__":
    main()