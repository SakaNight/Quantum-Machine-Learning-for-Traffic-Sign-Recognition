import pickle
import pandas as pd


def read_pkl_file(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error reading PKL file: {str(e)}")
        return None


def display_pkl_info(pkl_data):
    if pkl_data is None:
        return

    print(f"\nTotal number of entries: {len(pkl_data)}")
    print("\nFirst 5 entries:")
    for i, (image_name, class_id) in enumerate(pkl_data.items()):
        if i >= 5:
            break
        print(f"Image: {image_name}, Class ID: {class_id}")

    class_ids = list(pkl_data.values())
    unique_classes = set(class_ids)
    print(f"\nNumber of unique classes: {len(unique_classes)}")
    print("Class distribution:")
    for class_id in sorted(unique_classes):
        count = class_ids.count(class_id)
        print(f"Class {class_id}: {count} images")


def convert_to_dataframe(pkl_data):
    df = pd.DataFrame.from_dict(pkl_data.items())
    df.columns = ['Image_Name', 'Class_ID']
    return df


if __name__ == "__main__":
    pkl_path = "pkls/test_dataset_100.pkl"

    data = read_pkl_file(pkl_path)

    if data:
        print("Successfully read PKL file!")
        display_pkl_info(data)

        df = convert_to_dataframe(data)
        print("\nDataFrame head:")
        print(df.head())

        image_name = list(data.keys())[0]
        print(f"\nClass ID for image {image_name}: {data[image_name]}")