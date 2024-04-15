import h5py
import numpy as np
import torch


def find_minimum_separation(coords):
    # Ensure coords is a numpy array
    coords = np.asarray(coords)
    # Calculate all pairwise differences
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # Calculate the Euclidean distances
    distances = np.sqrt(np.sum(diffs**2, axis=2))
    # Set the diagonal (self-comparisons) to infinity to ignore them
    np.fill_diagonal(distances, np.inf)
    # Find the minimum distance
    min_distance = np.min(distances)
    return min_distance


def create_feature_image(pix_coord, features):

    feature_dim = np.prod(features[0].shape)

    # Convert pix_coord to a numpy array if it isn't already
    pix_coord = np.array(pix_coord)

    # Determine the dimensions of the image
    height = pix_coord[:, 0].max() + 1  # Assuming first column is the y-coordinate
    width = pix_coord[:, 1].max() + 1  # Assuming second column is the x-coordinate

    # Initialize the tensor for the image
    image = torch.zeros((int(height), int(width), feature_dim), dtype=torch.float32)

    # Fill the tensor with the features
    for coord, feature in zip(pix_coord, features):
        # Ensure the feature is a tensor and has the correct shape
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        # reshape the feature tensor to a 1D tensor
        feature_tensor = feature_tensor.view(-1)

        # Assign the feature vector to the correct location
        image[int(coord[0]), int(coord[1])] = feature_tensor

    # reshape the image to (C, H, W)
    image = image.permute(2, 0, 1)

    return image


def h5_to_feature_image(h5_file):
    # Extract the coordinates and features
    coords = h5_file["coords"][:]
    features = h5_file["features"][:]

    # Calculate the minimum separation
    min_sep = find_minimum_separation(coords)

    # Convert the coordinates to pixel coordinates
    pix_coords = coords // min_sep

    # Create the feature image
    feature_image = create_feature_image(pix_coords, features)

    return feature_image, pix_coords


def h5_to_standard_format(h5_file):
    """Return a triples of (coords, pix_coords, feature_dim) from the h5 file."""

    # Extract the coordinates and features
    coords = h5_file["coords"][:]
    features = h5_file["features"][:]

    feature_image, pix_coords = h5_to_feature_image(h5_file)

    return coords, pix_coords, feature_image


if __name__ == "__main__":
    h5_path = "/Users/neo/Documents/MODS/wsi-arch/examples/23.CFNA.1 A1 H&E _164024-patch_features.h5"

    # open the h5 file
    h5_file = h5py.File(h5_path, "r")

    feature_image, _ = h5_to_feature_image(h5_file)

    print(feature_image.shape)
