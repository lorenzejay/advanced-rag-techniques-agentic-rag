import os
from datasets import load_dataset
from PIL import Image


# Used to download the invoices dataset from the Hugging Face Hub: source: https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1
def load_invoices():
    ds = load_dataset("katanaml-org/invoices-donut-data-v1")
    test_ds = ds.get("test")
    print("ds", test_ds)
    print("typeof", type(test_ds))

    # Create knowledge directory if it doesn't exist
    knowledge_dir = "knowledge"
    if not os.path.exists(knowledge_dir):
        os.makedirs(knowledge_dir)
        print(f"Created directory: {knowledge_dir}")

    # Iterate through the first 5 images in the dataset
    for i, example in enumerate(test_ds):
        if i >= 5:  # Only process the first 5 images
            break

        # Access the image (this is a PIL Image object)
        image = example["image"]
        ground_truth = example["ground_truth"]

        # Check if the dataset has a filename field
        if "filename" in example:
            filename = example["filename"]
            print(f"Image {i}: {filename}")
        else:
            # Create a filename based on the index
            filename = f"invoice_{i}.png"
            print(f"Image {i}: (Generated filename: {filename})")

        # Save the image to the knowledge directory
        image_path = os.path.join(knowledge_dir, "invoices", filename)
        image.save(image_path)
        print(f"  Saved to: {image_path}")
        print(f"  Size: {image.size}")
        print(f"  Mode: {image.mode}")
        print(
            f"  Ground truth: {ground_truth[:100]}..."
        )  # Print first 100 chars of ground truth


if __name__ == "__main__":
    load_invoices()
