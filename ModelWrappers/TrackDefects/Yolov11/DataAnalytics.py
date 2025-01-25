import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

class YOLODatasetReporter:
    def __init__(self, dataset_path: str, output_dir: str = None):
        """Initialize analyzer and reporter for YOLO dataset"""
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist")
        
        self.dataset_path = dataset_path
        self.output_dir = output_dir or os.path.join(dataset_path, 'analysis_report')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Expected YOLO dataset structure
        self.splits = ['train', 'valid', 'test']
        self.class_names = self._load_class_names()
        self.labels_by_split = self._load_all_splits()
        
    def _load_class_names(self) -> Dict[int, str]:
        """Load class names from data.yaml"""
        yaml_path = os.path.join(self.dataset_path, 'data.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                try:
                    data = yaml.safe_load(f)
                    return {i: name for i, name in enumerate(data.get('names', []))}
                except:
                    return {}
        return {}

    def _load_labels(self, labels_path: str) -> pd.DataFrame:
        """Load labels from a specific directory"""
        labels = []
        for filename in os.listdir(labels_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(labels_path, filename)
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            labels.append({
                                'filename': filename,
                                'class_id': int(parts[0]),
                                'x_center': float(parts[1]),
                                'y_center': float(parts[2]),
                                'width': float(parts[3]),
                                'height': float(parts[4])
                            })
        return pd.DataFrame(labels)

    def _load_all_splits(self) -> Dict[str, pd.DataFrame]:
        """Load labels from all splits"""
        labels_by_split = {}
        for split in self.splits:
            split_path = os.path.join(self.dataset_path, split, 'labels')
            if os.path.exists(split_path):
                labels_by_split[split] = self._load_labels(split_path)
        return labels_by_split

    def generate_comprehensive_report(self):
        """Generate comprehensive visualization report"""
        plt.close('all')  # Close any existing plots
        
        # Combine all labels
        all_labels = pd.concat(list(self.labels_by_split.values()), keys=self.labels_by_split.keys())
        
        # Create a 2x3 subplot grid
        fig, axs = plt.subplots(2, 3, figsize=(20, 12))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # 1. Overall Class Distribution
        class_dist = all_labels['class_id'].value_counts()
        class_names = [self.class_names.get(k, str(k)) for k in class_dist.index]
        axs[0, 0].bar(class_names, class_dist.values)
        axs[0, 0].set_title('Overall Class Distribution')
        axs[0, 0].set_xlabel('Classes')
        axs[0, 0].set_ylabel('Number of Instances')
        plt.setp(axs[0, 0].get_xticklabels(), rotation=45, ha='right')
        
        # 2. Bounding Box Size Distribution
        bbox_areas = all_labels['width'] * all_labels['height']
        sns.histplot(bbox_areas, ax=axs[0, 1], kde=True)
        axs[0, 1].set_title('Bounding Box Area Distribution')
        axs[0, 1].set_xlabel('Bounding Box Area')
        axs[0, 1].set_ylabel('Frequency')
        
        # 3. Split-wise Class Distribution
        split_dist = all_labels.groupby([all_labels.index.get_level_values(0), 'class_id']).size().unstack(fill_value=0)
        split_dist.plot(kind='box', ax=axs[0, 2])
        axs[0, 2].set_title('Class Distribution Across Splits')
        axs[0, 2].set_xlabel('Classes')
        axs[0, 2].set_ylabel('Number of Instances')
        
        # 4. Bounding Box Aspect Ratio
        aspect_ratios = all_labels['width'] / all_labels['height']
        sns.histplot(aspect_ratios, ax=axs[1, 0], kde=True)
        axs[1, 0].set_title('Bounding Box Aspect Ratio')
        axs[1, 0].set_xlabel('Width/Height Ratio')
        axs[1, 0].set_ylabel('Frequency')
        
        # 5. Heatmap of Object Locations
        sns.scatterplot(data=all_labels, x='x_center', y='y_center', ax=axs[1, 1])
        axs[1, 1].set_title('Object Location Heatmap')
        axs[1, 1].set_xlabel('X Center')
        axs[1, 1].set_ylabel('Y Center')
        
        # 6. Pie Chart of Split Composition
        split_sizes = all_labels.groupby(level=0).size()
        axs[1, 2].pie(
            split_sizes.values, 
            labels=split_sizes.index, 
            autopct='%1.1f%%'
        )
        axs[1, 2].set_title('Dataset Split Composition')
        
        # Save the comprehensive plot
        plt.tight_layout()
        report_path = os.path.join(self.output_dir, 'dataset_analysis_report.png')
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive report saved to {report_path}")

# Example usage
if __name__ == "__main__":
    try:
        dataset_path = r'C:\Users\shiva\Desktop\EXCEED\Railway-Track-3'
        output_dir = r'C:\Users\shiva\Desktop\EXCEED\results\YOLOV11_Dataset_Analytics'
        reporter = YOLODatasetReporter(dataset_path, output_dir)
        reporter.generate_comprehensive_report()
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()