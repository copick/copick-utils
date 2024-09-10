import numpy as np
from skimage.feature import multiscale_basic_features
import copick
import zarr
from numcodecs import Blosc
import os

def load_copick_root(copick_config_path):
    """
    Loads the Copick root from the given configuration file path.
    """
    print(f"Loading Copick root configuration from: {copick_config_path}")
    root = copick.from_file(copick_config_path)
    print("Copick root loaded successfully")
    return root

def get_tomogram(root, run_name, voxel_spacing, tomo_type):
    """
    Fetches the tomogram data from the Copick root using the run name, voxel spacing, and tomogram type.
    """
    run = root.get_run(run_name)
    if run is None:
        raise ValueError(f"Run with name '{run_name}' not found.")
    
    voxel_spacing_obj = run.get_voxel_spacing(voxel_spacing)
    if voxel_spacing_obj is None:
        raise ValueError(f"Voxel spacing '{voxel_spacing}' not found in run '{run_name}'.")

    tomogram = voxel_spacing_obj.get_tomogram(tomo_type)
    if tomogram is None:
        raise ValueError(f"Tomogram type '{tomo_type}' not found for voxel spacing '{voxel_spacing}'.")
    
    return tomogram

def process_tomogram(tomogram, feature_type, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16.0):
    """
    Processes the tomogram chunkwise and computes the multiscale basic features.
    """
    image = zarr.open(tomogram.zarr(), mode='r')['0']
    input_chunk_size = image.chunks
    chunk_size = input_chunk_size if len(input_chunk_size) == 3 else input_chunk_size[1:]
    
    overlap = int(3 * sigma_max)
    
    print(f"Processing image with shape {image.shape}")
    print(f"Using chunk size: {chunk_size}, overlap: {overlap}")

    # Determine number of features by running on a small test array
    test_chunk = np.zeros((10, 10, 10), dtype=image.dtype)
    test_features = multiscale_basic_features(
        test_chunk,
        intensity=intensity,
        edges=edges,
        texture=texture,
        sigma_min=sigma_min,
        sigma_max=sigma_max
    )
    num_features = test_features.shape[-1]

    # Prepare output Zarr array directly in the tomogram store
    print(f"Creating new feature store with {num_features} features...")
    copick_features = tomogram.new_features(feature_type)
    feature_store = copick_features.zarr()

    out_array = zarr.create(
        shape=(num_features, *image.shape),
        chunks=(num_features, *chunk_size),
        dtype='float32',
        compressor=Blosc(cname='zstd', clevel=3, shuffle=2),
        store=feature_store,
        overwrite=True
    )

    # Process each chunk
    for z in range(0, image.shape[0], chunk_size[0]):
        for y in range(0, image.shape[1], chunk_size[1]):
            for x in range(0, image.shape[2], chunk_size[2]):
                z_start = max(z - overlap, 0)
                z_end = min(z + chunk_size[0] + overlap, image.shape[0])
                y_start = max(y - overlap, 0)
                y_end = min(y + chunk_size[1] + overlap, image.shape[1])
                x_start = max(x - overlap, 0)
                x_end = min(x + chunk_size[2] + overlap, image.shape[2])

                chunk = image[z_start:z_end, y_start:y_end, x_start:x_end]
                chunk_features = multiscale_basic_features(
                    chunk,
                    intensity=intensity,
                    edges=edges,
                    texture=texture,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max
                )

                # Adjust indices for overlap
                z_slice = slice(overlap if z_start > 0 else 0, None if z_end == image.shape[0] else -overlap)
                y_slice = slice(overlap if y_start > 0 else 0, None if y_end == image.shape[1] else -overlap)
                x_slice = slice(overlap if x_start > 0 else 0, None if x_end == image.shape[2] else -overlap)

                # Ensure contiguous array and correct slicing
                contiguous_chunk = np.ascontiguousarray(chunk_features[z_slice, y_slice, x_slice].transpose(3, 0, 1, 2))

                out_array[0:num_features, z:z + chunk_size[0], y:y + chunk_size[1], x:x + chunk_size[2]] = contiguous_chunk

    print(f"Features saved under feature type '{feature_type}'")
    
def generate_skimage_features(copick_config_path, run_name, voxel_spacing, tomo_type, feature_type, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16.0):
    """
    Generates multiscale features using scikit-image from a Copick run.
    """
    # Load Copick root configuration
    root = load_copick_root(copick_config_path)
    
    # Get the tomogram
    tomogram = get_tomogram(root, run_name, voxel_spacing, tomo_type)
    
    # Process the image and generate features
    process_image(tomogram, feature_type, intensity, edges, texture, sigma_min, sigma_max)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Multiscale Basic Features using Scikit-Image and Copick API.")
    parser.add_argument("copick_config_path", type=str, help="Path to the Copick configuration JSON file.")
    parser.add_argument("run_name", type=str, help="Name of the Copick run to process.")
    parser.add_argument("voxel_spacing", type=float, help="Voxel spacing to be used.")
    parser.add_argument("tomo_type", type=str, help="Type of tomogram to process.")
    parser.add_argument("feature_type", type=str, help="Name for the feature type to be saved.")
    parser.add_argument("--intensity", type=bool, default=True, help="Include intensity features.")
    parser.add_argument("--edges", type=bool, default=True, help="Include edge features.")
    parser.add_argument("--texture", type=bool, default=True, help="Include texture features.")
    parser.add_argument("--sigma_min", type=float, default=0.5, help="Minimum sigma for Gaussian blurring.")
    parser.add_argument("--sigma_max", type=float, default=16.0, help="Maximum sigma for Gaussian blurring.")

    args = parser.parse_args()

    generate_skimage_features(
        copick_config_path=args.copick_config_path,
        run_name=args.run_name,
        voxel_spacing=args.voxel_spacing,
        tomo_type=args.tomo_type,
        feature_type=args.feature_type,
        intensity=args.intensity,
        edges=args.edges,
        texture=args.texture,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max
    )
