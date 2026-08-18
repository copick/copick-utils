import tempfile

import numpy as np
from skimage.feature import multiscale_basic_features

from copick_utils.io.zarr import get_level_array


def _axis_slices(origin, chunk_size, overlap, image_size):
    target_end = min(origin + chunk_size, image_size)
    read_start = max(origin - overlap, 0)
    read_end = min(target_end + overlap, image_size)
    crop_start = origin - read_start
    crop_end = target_end - read_start

    return (
        slice(read_start, read_end),
        slice(crop_start, crop_end),
        slice(origin, target_end),
    )


def compute_skimage_features(
    tomogram,
    feature_type,
    copick_root,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16.0,
    feature_chunk_size=None,
    *,
    chunks=None,
    shards=None,
):
    """
    Processes the tomogram chunkwise and computes the multiscale basic features.
    Allows for optional feature chunk size.
    """
    image = get_level_array(tomogram)
    input_chunk_size = feature_chunk_size if feature_chunk_size else image.chunks
    chunk_size = input_chunk_size if len(input_chunk_size) == 3 else input_chunk_size[1:]

    overlap = int(chunk_size[0] / 2)

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
        sigma_max=sigma_max,
    )
    num_features = test_features.shape[-1]

    # Preserve the existing entity-creation timing, but defer all persistence
    # until the complete feature tensor has been assembled on disk.
    print(f"Creating new feature store with {num_features} features...")
    copick_features = tomogram.new_features(feature_type)
    with tempfile.TemporaryDirectory(prefix="copick-utils-skimage-") as directory:
        out_array = np.memmap(
            f"{directory}/features.dat",
            dtype=np.float32,
            mode="w+",
            shape=(num_features, *image.shape),
        )
        try:
            # Process each chunk
            for z in range(0, image.shape[0], chunk_size[0]):
                for y in range(0, image.shape[1], chunk_size[1]):
                    for x in range(0, image.shape[2], chunk_size[2]):
                        z_read, z_crop, z_output = _axis_slices(z, chunk_size[0], overlap, image.shape[0])
                        y_read, y_crop, y_output = _axis_slices(y, chunk_size[1], overlap, image.shape[1])
                        x_read, x_crop, x_output = _axis_slices(x, chunk_size[2], overlap, image.shape[2])

                        chunk = image[z_read, y_read, x_read]
                        chunk_features = multiscale_basic_features(
                            chunk,
                            intensity=intensity,
                            edges=edges,
                            texture=texture,
                            sigma_min=sigma_min,
                            sigma_max=sigma_max,
                        )

                        contiguous_chunk = np.ascontiguousarray(
                            chunk_features[z_crop, y_crop, x_crop].transpose(3, 0, 1, 2),
                        )
                        out_array[:, z_output, y_output, x_output] = contiguous_chunk

            storage_chunks = chunks
            if storage_chunks is None and feature_chunk_size is not None:
                storage_chunks = feature_chunk_size

            out_array.flush()
            copick_features.from_numpy(
                out_array,
                chunks=storage_chunks,
                shards=shards,
                dtype=np.float32,
                overwrite=True,
            )
        finally:
            del out_array

    print(f"Features saved under feature type '{feature_type}'")
    return copick_features


if __name__ == "__main__":
    root = None  # copick.from_file
    tomo = None  # get a tomogram from root
    compute_skimage_features(
        tomogram=tomo,
        feature_type="skimageFeatures",
        copick_root=root,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=0.5,
        sigma_max=16.0,
        feature_chunk_size=None,  # Default to detected chunk size
    )
