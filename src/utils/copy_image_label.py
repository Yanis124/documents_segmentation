import os
import shutil


def copy_labels_images(file_name: str, label_src_dir: str, label_dest_dir: str, image_src_dir: str, image_dest_dir: str):
    """Copy label and image files to their respective destination directories.

    Args:
        file_name (str): The name of the file (without extension).
        label_dest (str): The destination directory for label files.
        image_dest (str): The destination directory for image files.
    """
    image_file = os.path.splitext(file_name)[0] + ".jpg"  # assumes .jpg images
    src_label = os.path.join(label_src_dir, file_name)
    src_image = os.path.join(image_src_dir, image_file)

    dst_label = os.path.join(label_dest_dir, file_name)
    dst_image = os.path.join(image_dest_dir, image_file)

    shutil.copy(src_label, dst_label)
    if os.path.exists(src_image):  # copy image only if it exists
        shutil.copy(src_image, dst_image)