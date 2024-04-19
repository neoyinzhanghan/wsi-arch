import openslide

def get_thumbnail(wsi_path):
    """Open the WSI at wsi_path and return the lowest resolution view of the images as a PIL image."""

    # Open the WSI
    wsi = openslide.OpenSlide(wsi_path)

    # Get the thumbnail
    thumbnail = wsi.get_thumbnail(wsi.level_dimensions[-1])

    return thumbnail
