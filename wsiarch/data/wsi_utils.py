import openslide

def get_thumbnail(wsi_path):
    """Open the wsi at wsi_path and return the highest level view of the images as a PIL image which is the lowest resolution image."""

    # Open the wsi
    wsi = openslide.OpenSlide(wsi_path)

    # Get the thumbnail
    thumbnail = wsi.get_thumbnail(wsi.level_dimensions[0])

    return thumbnail
