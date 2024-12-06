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
        all_samples = []
        class_counts = {}

        class_folders = sorted(glob(os.path.join(root_dir, '*')))

        for folder in class_folders:
            class_id = int(os.path.basename(folder))

            if selected_classes is not None and class_id not in selected_classes:
                continue

            if class_id == 0:
                samples = []
                image_paths = glob(os.path.join(folder, '*.jpg'))
                if datasize != 0:
                    num_pick = int(datasize / num_classes)
                    current_size = len(image_paths)

                    if current_size >= num_pick:
                        sampled_paths = random.sample(image_paths, num_pick)
                    else:
                        repeat_times = num_pick // current_size + 1
                        extended_paths = image_paths * repeat_times
                        sampled_paths = random.sample(extended_paths, num_pick)

                    for image_path in sampled_paths:
                        samples.append((image_path, class_id))
                else:
                    for image_path in image_paths:
                        samples.append((image_path, class_id))

            else:
                csv_files = glob(os.path.join(folder, '*.csv'))
                if not csv_files:
                    print(f"Warning: No CSV file found in {folder}")
                    continue

                df = pd.read_csv(csv_files[0], sep=';')
                if datasize != 0:
                    num_diff = df.Height.size - int(datasize / num_classes)
                    if num_diff >= 0:
                        num_pick = int(datasize / num_classes)
                        df = df.sample(n=num_pick).reset_index(drop=True)
                    else:
                        num_pick = int(datasize / num_classes)
                        current_size = df.Height.size
                        repeat_times = num_pick // current_size + 1
                        df_repeated = pd.concat([df] * repeat_times, ignore_index=True)
                        df = df_repeated.sample(n=num_pick).reset_index(drop=True)

                samples = []
                for _, row in df.iterrows():
                    image_path = os.path.join(folder, row['Filename'])
                    samples.append((image_path, class_id))

            all_samples.extend(samples)
            class_counts[class_id] = len(samples)

        return all_samples, class_counts

    def collect_test_data(self, csv_path, image_dir, selected_classes, num_classes, datasize=0):
        all_samples = []
        class_counts = {}

        df = pd.read_csv(csv_path, sep=';')

        if selected_classes is not None:
            df = df[df['ClassId'].isin(selected_classes)]

        if datasize !=0:
            df = df.groupby('ClassId').apply(lambda x: x.sample(n=min(int(datasize/num_classes), len(x)), random_state=42)).reset_index(
                drop=True)

        for _, row in df.iterrows():
            class_id = int(row['ClassId'])
            image_path = os.path.join(image_dir, row['Filename']) if image_dir else row['Filename']
            all_samples.append((image_path, class_id))

            if class_id not in class_counts:
                class_counts[class_id] = 0
            class_counts[class_id] += 1

        return all_samples, class_counts

    def upsample_minority_classes(self, samples, class_counts):
        if self.class_distribution is None:
            self.analyze_class_distribution(class_counts)

        max_samples = self.class_distribution['max_samples']
        upsampled_samples = []

        class_samples = {}
        for sample in samples:
            class_id = sample[1]
            if class_id not in class_samples:
                class_samples[class_id] = []
            class_samples[class_id].append(sample)

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

        if is_training and up_sampling:
            samples = self.upsample_minority_classes(samples, class_counts)

        image_paths = {}
        labels = {}

        for idx, (image_path, class_id) in enumerate(samples):
            image_paths[idx] = image_path
            if selected_classes is not None:
                labels[idx] = MAP_ID[class_id]
            else:
                labels[idx] = class_id

        data = {
            'image_paths': image_paths,
            'labels': labels
        }

        with open(output_pkl_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"\nDataset statistics for {'training' if is_training else 'test'} set:")
        print(f"Total samples: {len(image_paths)}")
        print("\nClass distribution:")
        final_class_counts = Counter(labels.values())
        for class_id, count in sorted(final_class_counts.items()):
            print(f"Class {class_id}: {count} samples")

        return data




if __name__ == "__main__":
    generator = DatasetPklGenerator()

    trainset_size = 0 #10000  # 0 for default, replace with int numbers
    testset_size = 0 #200  # 0 for default, replace with int numbers
    train_root = "GTSRB/Final_Training/Images"
    test_csv = "GTSRB/GT-final_test.csv"
    test_image_dir = "GTSRB/Final_Test/Images"
    selected_classes = None #[0,1,2,4,5,10,14,17,27,34] # Use None for all classes, or specify classes using [0,1,2,5,27]
    num_classes = 43  # must match with selected_classes
    if selected_classes is not None:
        MAP_ID = {}
        for i, selected_class in enumerate(selected_classes):
            MAP_ID[selected_class] = i

    train_data = generator.create_dataset_pkl(
        output_pkl_path="pkls/train_dataset_{}cls_{}.pkl".format(num_classes, trainset_size),
        selected_classes=selected_classes,
        num_classes=num_classes,
        datasize=trainset_size,
        is_training=True,
        up_sampling=True,
        root_dir=train_root,
    )

    test_data = generator.create_dataset_pkl(
        output_pkl_path="pkls/test_dataset_{}cls_{}.pkl".format(num_classes, testset_size),
        selected_classes=selected_classes,
        num_classes=num_classes,
        datasize=testset_size,
        is_training=False,
        csv_path=test_csv,
        image_dir=test_image_dir,
    )