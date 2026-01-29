import os 
import json 
import pandas as pd 

def dataset_preparation(captions_path, meta_data, output_path, filtered_images):
    dataset = []
    images = []

    meta_data_df = pd.read_csv(meta_data)

    with open(captions_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            img_name = data['image']
            meta_row = meta_data_df[meta_data_df['file_name'] == img_name]

            row = meta_row.iloc[0]
            prompt = f"""
You are an Insurance Adjuster. Evaluate the car damage shown in the image.

Metadata:
- Shooting Angle: {row['shooting angle']}
- View: {row['complete or partial ']}
- Color: {row['color']}
- Damage Category: {row['category_name']}
- Area: {row['area']}
- BBox: {row['bbox']}
- Segmentation Length: {len(json.loads(row['segmentation']) if isinstance(row['segmentation'], str) else row['segmentation'])}\n"
- IsCrowd: {row['iscrowd']}
"""
            dataset.append({
                "id": data['image'].replace(".jpg", ""),
                "image": data['image'],
                "conversations": [
                    {
                        "role": "user", 
                        "content": prompt
                    },
                    {
                        "role": "assistant", 
                        "content": data['caption']
                    }
                ]
            })
            images.append(data['image'])

    with open(output_path, 'w', encoding='utf-8') as w:
        json.dump(dataset, w, ensure_ascii=False, indent=2)

    with open(filtered_images, 'w', encoding='utf-8') as m:
        json.dump(images, m, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    captions_path = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/filtered_captions.jsonl"
    meta_data = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/metadata/enriched_trained_meta.csv"
    output_path = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/dataset/dataset.json"
    filtered_images = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data/gpt/filtered_images_only.json"
    dataset_preparation(
        captions_path=captions_path,
        meta_data=meta_data,
        output_path=output_path,
        filtered_images=filtered_images
    )
