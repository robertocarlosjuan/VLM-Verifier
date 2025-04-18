from PIL import Image, ImageDraw

def combine_images_side_by_side(image1_path, image2_path, output_path, border_width=10, border_color=(0, 0, 0), scale_factor=0.5):
    # Open the two images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    width1, height1 = img1.size
    width2, height2 = img2.size
    combined_width = width1 + width2 + border_width
    combined_height = max(height1, height2)
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (width1 + border_width, 0))
    draw = ImageDraw.Draw(combined_image)
    draw.rectangle([width1, 0, width1 + border_width, combined_height], fill=border_color)
    new_width = int(combined_width * scale_factor)
    new_height = int(combined_height * scale_factor)
    resized_image = combined_image.resize((new_width, new_height))
    combined_image.save(output_path)

def get_template(file_path):
    with open(file_path) as f:
        return f.read()