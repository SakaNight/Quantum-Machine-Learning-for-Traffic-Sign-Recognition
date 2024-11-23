import pandas as pd
import pickle
import os
import random
import numpy as np
from collections import Counter
from glob import glob



class DatasetPklGenerator:
    def __init__(self):
        self.class_distribution = None

    def analyze_class_distribution(self, class_samples):
        """
        Analyze the distribution of samples across classes
        """
        mean_samples = np.mean(list(class_samples.values()))
        max_samples = max(class_samples.values())

        stats = {
            'class_counts': class_samples,
            'mean_samples': mean_samples,
            'max_samples': max_samples
        }

        self.class_distribution = stats
        return stats

    def collect_training_data(self, root_dir, selected_classes, num_classes, datasize=0):
        """
        Collect training data from folder structure where each class has its own folder

        Args:
            root_dir (str): Root directory containing class folders
            selected_classes (list): List of class IDs to include. If None, include all classes

        Returns:
            tuple: (all_samples, class_counts)
        """
        all_samples = []
        class_counts = {}

        # Get all class folders
        class_folders = sorted(glob(os.path.join(root_dir, '*')))

        for folder in class_folders:
            class_id = int(os.path.basename(folder))

            # Skip if not in selected classes
            if selected_classes is not None and class_id not in selected_classes:
                continue

            if class_id == 0:
                samples = []
                image_paths = glob(os.path.join(folder, '*.jpg'))
                if datasize != 0:
                    num_pick = int(datasize/(num_classes))
                    sampled_paths = random.sample(image_paths, num_pick)
                    for image_path in sampled_paths:
                        samples.append((image_path, class_id))
                else:
                    for image_path in image_paths:
                        samples.append((image_path, class_id))

            else:
                # Read class CSV file
                csv_files = glob(os.path.join(folder, '*.csv'))
                if not csv_files:
                    print(f"Warning: No CSV file found in {folder}")
                    continue

                df = pd.read_csv(csv_files[0], sep=';')
                if datasize != 0:
                    df = df.sample(n=int(datasize/(num_classes))).reset_index(drop=True)

                # Collect samples
                samples = []
                for _, row in df.iterrows():
                    image_path = os.path.join(folder, row['Filename'])
                    samples.append((image_path, class_id))

            all_samples.extend(samples)
            class_counts[class_id] = len(samples)

        return all_samples, class_counts

    def collect_test_data(self, csv_path, image_dir, selected_classes, num_classes, datasize=0):
        """
        Collect test data from a single CSV file

        Args:
            csv_path (str): Path to the test set CSV file
            image_dir (str): Base directory for test images
            selected_classes (list): List of class IDs to include. If None, include all classes

        Returns:
            tuple: (all_samples, class_counts)
        """
        all_samples = []
        class_counts = {}

        # Read CSV file
        df = pd.read_csv(csv_path, sep=';')

        # Filter classes if needed
        if selected_classes is not None:
            df = df[df['ClassId'].isin(selected_classes)]

        if datasize !=0:
            df = df.groupby('ClassId').apply(lambda x: x.sample(n=min(int(datasize/num_classes), len(x)), random_state=42)).reset_index(
                drop=True)

        # Collect samples
        for _, row in df.iterrows():
            class_id = int(row['ClassId'])
            image_path = os.path.join(image_dir, row['Filename']) if image_dir else row['Filename']
            all_samples.append((image_path, class_id))

            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1

        return all_samples, class_counts

    def upsample_minority_classes(self, samples, class_counts):
        """
        Upsample minority classes to min(3 * original_count, max_class_count)
        """
        if self.class_distribution is None:
            self.analyze_class_distribution(class_counts)

        max_samples = self.class_distribution['max_samples']
        upsampled_samples = []

        # Group samples by class
        class_samples = {}
        for sample in samples:
            class_id = sample[1]
            if class_id not in class_samples:
                class_samples[class_id] = []
            class_samples[class_id].append(sample)

        # Upsample each class as needed

        for class_id, class_samples_list in class_samples.items():
            print(class_id, len(class_samples_list))
            count = len(class_samples_list)

            if count < self.class_distribution['mean_samples']:
                target_count = min(count * 3, max_samples)
                num_copies = target_count // count
                remainder = target_count % count

                upsampled_class = class_samples_list * num_copies
                if remainder > 0:
                    indices = np.random.choice(len(class_samples_list), remainder, replace=True)
                    additional_samples = [class_samples_list[i] for i in indices]
                    upsampled_class.extend(additional_samples)

                upsampled_samples.extend(upsampled_class)
            else:
                upsampled_samples.extend(class_samples_list)


        return upsampled_samples

    def create_dataset_pkl(self, output_pkl_path, selected_classes, num_classes, datasize, is_training=False, up_sampling=True, **kwargs):
        """
        Create PKL file for either training or test dataset

            output_pkl_path (str): Path where the output PKL file
        Args:will be saved
            selected_classes (list): List of class IDs to include. If None, include all classes
            is_training (bool): Whether this is training data
            **kwargs: Additional arguments:
                For training: root_dir
                For test: csv_path, image_dir

        Returns:
            dict: Dictionary containing image_paths and labels
        """
        if 1:
            # Collect data based on dataset type

            if is_training:
                if 'root_dir' not in kwargs:
                    raise ValueError("root_dir is required for training dataset")
                samples, class_counts = self.collect_training_data(
                    kwargs['root_dir'], selected_classes, num_classes, datasize)
            else:
                if 'csv_path' not in kwargs:
                    raise ValueError("csv_path is required for test dataset")
                samples, class_counts = self.collect_test_data(
                    kwargs['csv_path'],
                    kwargs.get('image_dir', ''),
                    selected_classes,
                    num_classes,
                    datasize)

            if len(samples) == 0:
                raise ValueError("No samples found")

            # Perform upsampling for training data if needed
            if is_training and up_sampling:
                samples = self.upsample_minority_classes(samples, class_counts)

            # Create dictionaries
            image_paths = {}
            labels = {}

            # Fill dictionaries
            for idx, (image_path, class_id) in enumerate(samples):
                image_paths[idx] = image_path
                if selected_classes is not None:
                    labels[idx] = MAP_ID[class_id]
                else:
                    labels[idx] = class_id

            # Create final data structure
            data = {
                'image_paths': image_paths,
                'labels': labels
            }

            # Save to PKL file
            with open(output_pkl_path, 'wb') as f:
                pickle.dump(data, f)

            # Print statistics
            print(f"\nDataset statistics for {'training' if is_training else 'test'} set:")
            print(f"Total samples: {len(image_paths)}")
            print("\nClass distribution:")
            final_class_counts = Counter(labels.values())
            for class_id, count in sorted(final_class_counts.items()):
                print(f"Class {class_id}: {count} samples")

            return data

        # except Exception as e:
        #     print(f"Error occurred: {str(e)}")
        #     return None


if __name__ == "__main__":
    # Example usage
    generator = DatasetPklGenerator()

    # Parameters
    trainset_size = 100  # 0 for default, replace with int numbers
    testset_size = 100  # 0 for default, replace with int numbers
    train_root = "/home/jamie/Works/Waterloo/ECE730/ECE730Project/Data/GTSRB/Final_Training/Images"  # Contains class folders (00000, 00001, etc.)
    test_csv = "/home/jamie/Works/Waterloo/ECE730/ECE730Project/Data/GTSRB/GT-final_test.csv"
    test_image_dir = "/home/jamie/Works/Waterloo/ECE730/ECE730Project/Data/GTSRB/Final_Test/Images"
    selected_classes = None # Use None for all classes, or specify classes using [0,1,2,5,27]
    num_classes = 43  # must match with selected_classes
    if selected_classes is not None:
        MAP_ID = {}
        for i, selected_class in enumerate(selected_classes):
            MAP_ID[selected_class] = i

    # # # Create training dataset
    # train_data = generator.create_dataset_pkl(
    #     output_pkl_path="pkls/train_dataset_43cls_uw_100.pkl",
    #     selected_classes=selected_classes,
    #     num_classes=num_classes,
    #     datasize=trainset_size,
    #     is_training=True,
    #     up_sampling=False,
    #     root_dir=train_root,
    # )

    # Create test dataset
    test_data = generator.create_dataset_pkl(
        output_pkl_path="pkls/test_dataset_43cls_uw_100.pkl",
        selected_classes=selected_classes,
        num_classes=num_classes,
        datasize=testset_size,
        is_training=False,
        csv_path=test_csv,
        image_dir=test_image_dir,
    )