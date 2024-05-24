import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(224, 224)):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Function to resize images
    def resize_image(input_path, output_path, size):
        with Image.open(input_path) as img:
            img_resized = img.resize(size, Image.Resampling.LANCZOS)  # Use LANCZOS instead of ANTIALIAS
            img_resized.save(output_path)

    # Process all images in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):  # Add more extensions if needed
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            resize_image(input_path, output_path, size)

    print("All images have been resized and saved.")

# Example usage
input_directory = 'path/to/directory/images/to/be/resized'
output_directory = 'path/to/directory/images/after/resize/for/saving'
resize_images(input_directory, output_directory, size=(224, 224))
