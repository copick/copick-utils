"""3D spline fitting to skeleton volumes for pick generation with orientations."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from copick.util.uri import get_copick_objects_by_type
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:
    from copick.models import CopickPicks, CopickRoot, CopickRun, CopickSegmentation


class SkeletonSplineFitter:
    """3D spline fitting to skeleton coordinates with point sampling and orientation computation."""

    def __init__(self):
        self.skeleton_coords = None
        self.ordered_path = None
        self.spline_functions = None
        self.regularized_points = None
        self.t_sampled = None

    def extract_skeleton_coordinates(self, binary_volume: np.ndarray) -> np.ndarray:
        """Extract skeleton coordinates from binary volume."""
        self.skeleton_coords = np.array(np.where(binary_volume)).T
        return self.skeleton_coords

    def order_skeleton_points_longest_path(self, coords: np.ndarray, connectivity_radius: float = 2.0) -> np.ndarray:
        """Order skeleton points by finding the longest path through the skeleton."""
        if len(coords) <= 2:
            self.ordered_path = coords
            return coords

        # Build adjacency matrix
        distances = squareform(pdist(coords))
        adjacency = (distances <= connectivity_radius) & (distances > 0)

        # Create NetworkX graph
        G = nx.from_numpy_array(adjacency)

        # Find endpoints (degree 1 nodes)
        endpoints = [node for node, degree in G.degree() if degree == 1]

        if len(endpoints) < 2:
            # If no clear endpoints, use the two points that are farthest apart
            max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
            endpoints = [max_dist_idx[0], max_dist_idx[1]]

        # Find the longest path between any two endpoints
        longest_path = []
        max_length = 0

        for i, start in enumerate(endpoints):
            for end in endpoints[i + 1 :]:
                try:
                    path = nx.shortest_path(G, start, end)
                    if len(path) > max_length:
                        max_length = len(path)
                        longest_path = path
                except nx.NetworkXNoPath:
                    continue

        # If no path found, try all pairs of nodes
        if not longest_path:
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    try:
                        path = nx.shortest_path(G, i, j)
                        if len(path) > max_length:
                            max_length = len(path)
                            longest_path = path
                    except nx.NetworkXNoPath:
                        continue

        # Return ordered coordinates
        if longest_path:
            self.ordered_path = coords[longest_path]
        else:
            # Fallback: order by distance from first point
            self.ordered_path = self._order_by_nearest_neighbor(coords)

        return self.ordered_path

    def _order_by_nearest_neighbor(self, coords: np.ndarray) -> np.ndarray:
        """Fallback method to order points by nearest neighbor traversal."""
        if len(coords) == 0:
            return coords

        ordered = [0]  # Start with first point
        remaining = list(range(1, len(coords)))

        while remaining:
            current_point = coords[ordered[-1]]
            distances = [np.linalg.norm(coords[i] - current_point) for i in remaining]
            next_idx = remaining[np.argmin(distances)]
            ordered.append(next_idx)
            remaining.remove(next_idx)

        return coords[ordered]

    def fit_regularized_spline(
        self,
        coords: np.ndarray,
        smoothing_factor: Optional[float] = None,
        degree: int = 3,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Fit regularized 3D parametric spline using scipy.interpolate.splprep."""
        if len(coords) < degree + 1:
            raise ValueError(f"Need at least {degree + 1} points for degree {degree} spline")

        # Use splprep for parametric 3D spline fitting
        from scipy.interpolate import splprep

        # Determine spline degree
        k = min(degree, len(coords) - 1)
        if k <= 0:
            k = 1

        # coords should be transposed for splprep: [x_coords, y_coords, z_coords]
        tck, u = splprep([coords[:, 2], coords[:, 1], coords[:, 0]], s=smoothing_factor, k=k)
        print(f"Successfully fitted parametric spline with degree {k}, smoothing {smoothing_factor}")

        self.spline_functions = {"tck": tck, "u_original": u, "degree": k}

        return u, self.spline_functions

    def sample_points_along_spline(self, spacing_distance: float) -> np.ndarray:
        """Sample points along the spline at regular intervals using arc-length parameterization."""
        if self.spline_functions is None:
            raise ValueError("Spline not fitted yet. Call fit_regularized_spline first.")

        # Get the spline representation
        from scipy.interpolate import splev

        tck = self.spline_functions["tck"]

        # Estimate total length by sampling densely along the parameter
        u_dense = np.linspace(0, 1, 1000)
        points_dense = np.column_stack(splev(u_dense, tck))

        # Calculate cumulative arc lengths
        distances = np.zeros(len(points_dense))
        distances[1:] = np.cumsum(np.linalg.norm(np.diff(points_dense, axis=0), axis=1))

        total_length = distances[-1]
        if total_length == 0:
            # Fallback: just return the first point
            self.regularized_points = np.array([points_dense[0]])
            self.t_sampled = np.array([0.0])
            return self.regularized_points

        # Calculate number of points needed for desired spacing
        n_points = max(2, int(np.ceil(total_length / spacing_distance)) + 1)

        # Sample at regular arc length intervals
        target_distances = np.linspace(0, total_length, n_points)
        u_sampled = np.interp(target_distances, distances, u_dense)

        # Evaluate spline at sampled parameter values
        sampled_points = np.column_stack(splev(u_sampled, tck))

        self.regularized_points = sampled_points
        self.t_sampled = u_sampled  # Store parameter values for transform calculation
        return sampled_points

    def compute_transforms(self) -> np.ndarray:
        """
        Compute 4x4 transformation matrices for each sampled point.
        Follows the ArtiaX pattern: z_align(last_pos, curr_pos).zero_translation().inverse()
        The rotation from particle i-1 to particle i is applied to particle i-1.

        Returns:
            np.ndarray: [N, 4, 4] array of transformation matrices
        """
        if self.regularized_points is None:
            raise ValueError("Must sample points first before computing transforms")

        n_points = len(self.regularized_points)
        transforms = np.zeros((n_points, 4, 4))

        print(f"Computing transforms for {n_points} points using ArtiaX z_align pattern...")

        # Initialize all transforms as identity
        for i in range(n_points):
            transforms[i] = np.eye(4)

        # ArtiaX pattern: for each particle i, compute rotation from i-1 to i, apply to i-1
        for i in range(1, n_points):
            curr_pos = self.regularized_points[i]
            last_pos = self.regularized_points[i - 1]

            # z_align(last_pos, curr_pos).zero_translation().inverse()
            rotation_matrix = self._z_align_inverse(last_pos, curr_pos)

            # Apply rotation to the PREVIOUS particle (i-1)
            transforms[i - 1][:3, :3] = rotation_matrix

        # Handle the last particle - use the same rotation as the previous particle
        if n_points > 1:
            transforms[n_points - 1][:3, :3] = transforms[n_points - 2][:3, :3]

        return transforms

    def _z_align_inverse(self, pt1: np.ndarray, pt2: np.ndarray) -> np.ndarray:
        """
        Create the inverse of the z_align transformation.
        This rotates the z-axis to align with the pt1->pt2 direction.
        Based on the z_align algorithm but returns the inverse matrix.

        Args:
            pt1: Two 3D points defining the direction vector
            pt2: Two 3D points defining the direction vector

        Returns:
            np.ndarray: 3x3 rotation matrix (inverse of z_align)
        """
        a, b, c = pt2 - pt1
        l = a * a + c * c  # noqa
        d = l + b * b
        epsilon = 1e-10

        if abs(d) < epsilon:
            # Fallback to identity matrix
            return np.eye(3)

        l = np.sqrt(l)  # noqa
        d = np.sqrt(d)
        # Create the z_align rotation matrix
        xf = np.zeros((3, 3), dtype=np.float64)
        xf[1][1] = l / d

        if abs(l) < epsilon:
            xf[0][0] = 1.0
            xf[2][1] = -b / d
        else:
            xf[0][0] = c / l
            xf[2][0] = -a / l
            xf[0][1] = -(a * b) / (l * d)
            xf[2][1] = -(b * c) / (l * d)

        xf[0][2] = a / d
        xf[1][2] = b / d
        xf[2][2] = c / d

        return xf

    def get_spline_properties(self) -> Dict[str, Any]:
        """Get properties of the fitted spline."""
        if self.regularized_points is None:
            return {}

        # Calculate total length
        distances = np.linalg.norm(np.diff(self.regularized_points, axis=0), axis=1)
        total_length = np.sum(distances)

        # Calculate curvature at sampled points
        curvatures = []
        if len(self.regularized_points) >= 3:
            for i in range(1, len(self.regularized_points) - 1):
                p1, p2, p3 = self.regularized_points[i - 1 : i + 2]
                v1 = p2 - p1
                v2 = p3 - p2
                # Approximate curvature
                cross_prod = np.linalg.norm(np.cross(v1, v2))
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    curvature = cross_prod / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    curvatures.append(curvature)

        return {
            "n_points": len(self.regularized_points),
            "total_length": total_length,
            "average_spacing": (
                total_length / (len(self.regularized_points) - 1) if len(self.regularized_points) > 1 else 0
            ),
            "mean_curvature": np.mean(curvatures) if curvatures else 0,
            "max_curvature": np.max(curvatures) if curvatures else 0,
            "curvatures": curvatures,
        }

    def detect_high_curvature_outliers(self, coords: np.ndarray, curvature_threshold: float = 0.2) -> np.ndarray:
        """Detect points that contribute to high curvature."""
        if len(coords) < 4:
            return coords

        # Calculate curvature at each point
        curvatures = []
        for i in range(1, len(coords) - 1):
            p1, p2, p3 = coords[i - 1 : i + 2]
            v1 = p2 - p1
            v2 = p3 - p2
            cross_prod = np.linalg.norm(np.cross(v1, v2))
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                curvature = cross_prod / (np.linalg.norm(v1) * np.linalg.norm(v2))
                curvatures.append(curvature)
            else:
                curvatures.append(0)

        # Find points with high curvature
        curvatures = np.array(curvatures)
        high_curvature_mask = curvatures > curvature_threshold

        if not np.any(high_curvature_mask):
            return coords

        # Remove points contributing to high curvature (keep first and last)
        points_to_keep = [True] + [not high_curvature_mask[i] for i in range(len(high_curvature_mask))] + [True]

        filtered_coords = coords[points_to_keep]
        removed_count = len(coords) - len(filtered_coords)

        print(f"Removed {removed_count} outlier points with high curvature")
        return filtered_coords


def fit_spline_to_skeleton(
    binary_volume: np.ndarray,
    spacing_distance: float,
    smoothing_factor: Optional[float] = None,
    degree: int = 3,
    connectivity_radius: float = 2.0,
    compute_transforms: bool = True,
    curvature_threshold: float = 0.2,
    max_iterations: int = 5,
) -> Tuple[np.ndarray, Optional[np.ndarray], SkeletonSplineFitter, Dict[str, Any]]:
    """
    Main function to fit a regularized 3D spline to a skeleton and sample points.

    Args:
        binary_volume: 3D binary volume where skeleton is True/1
        spacing_distance: Distance between consecutive sampled points along the spline
        smoothing_factor: Smoothing parameter for spline fitting (auto if None)
        degree: Degree of the spline (1-5)
        connectivity_radius: Maximum distance to consider skeleton points as connected
        compute_transforms: Whether to compute 4x4 transformation matrices for each point
        curvature_threshold: Maximum allowed curvature before outlier removal (default 0.2)
        max_iterations: Maximum number of outlier removal iterations (default 5)

    Returns:
        Tuple of (sampled_points, transforms, spline_fitter, properties):
            - sampled_points: Nx3 array of evenly spaced points along spline
            - transforms: [N, 4, 4] array of transformation matrices (or None if compute_transforms=False)
            - spline_fitter: SkeletonSplineFitter object for further analysis
            - properties: dict with spline properties
    """
    # Initialize fitter
    fitter = SkeletonSplineFitter()

    # Extract skeleton coordinates
    coords = fitter.extract_skeleton_coordinates(binary_volume)

    if len(coords) == 0:
        raise ValueError("No skeleton points found in binary volume")

    # Order skeleton points
    ordered_coords = fitter.order_skeleton_points_longest_path(coords, connectivity_radius=connectivity_radius)

    if len(ordered_coords) < 2:
        raise ValueError("Not enough ordered points for spline fitting")

    # Iterative fitting with outlier removal
    current_coords = ordered_coords.copy()
    iteration = 0

    while iteration < max_iterations:
        # Fit regularized spline
        fitter.fit_regularized_spline(current_coords, smoothing_factor=smoothing_factor, degree=degree)

        # Sample points along spline
        sampled_points = fitter.sample_points_along_spline(spacing_distance)

        # Get properties to check curvature
        properties = fitter.get_spline_properties()
        max_curvature = properties.get("max_curvature", 0)

        print(f"Iteration {iteration + 1}: Max curvature = {max_curvature:.4f}")

        # If curvature is acceptable, break
        if max_curvature <= curvature_threshold:
            print(f"Curvature acceptable after {iteration + 1} iterations")
            break

        # Remove outliers and try again
        print(f"Max curvature {max_curvature:.4f} > {curvature_threshold}, removing outliers...")
        filtered_coords = fitter.detect_high_curvature_outliers(current_coords, curvature_threshold)

        # If no points were removed, break to avoid infinite loop
        if len(filtered_coords) == len(current_coords):
            print("No outliers found to remove, stopping iterations")
            break

        # If too few points remain, break
        if len(filtered_coords) < degree + 1:
            print(f"Too few points remaining ({len(filtered_coords)}), stopping iterations")
            break

        current_coords = filtered_coords
        iteration += 1

    if iteration >= max_iterations:
        print(f"Reached maximum iterations ({max_iterations}), final curvature: {max_curvature:.4f}")

    # Compute transformation matrices if requested
    transforms = None
    if compute_transforms:
        transforms = fitter.compute_transforms()

    # Get final properties
    properties = fitter.get_spline_properties()

    return sampled_points, transforms, fitter, properties


def fit_spline_to_segmentation(
    segmentation: "CopickSegmentation",
    spacing_distance: float,
    smoothing_factor: Optional[float] = None,
    degree: int = 3,
    connectivity_radius: float = 2.0,
    compute_transforms: bool = True,
    curvature_threshold: float = 0.2,
    max_iterations: int = 5,
    output_session_id: Optional[str] = None,
    output_user_id: str = "spline",
    voxel_spacing: float = 1.0,
) -> Optional["CopickPicks"]:
    """
    Fit a spline to a segmentation (skeleton) volume and create picks with orientations.

    Args:
        segmentation: Input segmentation containing skeleton to fit spline to
        spacing_distance: Distance between consecutive sampled points along the spline
        smoothing_factor: Smoothing parameter for spline fitting (auto if None)
        degree: Degree of the spline (1-5)
        connectivity_radius: Maximum distance to consider skeleton points as connected
        compute_transforms: Whether to compute orientations for picks
        curvature_threshold: Maximum allowed curvature before outlier removal
        max_iterations: Maximum number of outlier removal iterations
        output_session_id: Session ID for output picks (default: same as input)
        output_user_id: User ID for output picks
        voxel_spacing: Voxel spacing for coordinate scaling

    Returns:
        Created picks object or None if failed
    """
    # Get the segmentation volume
    volume = segmentation.numpy()
    if volume is None:
        print(f"Error: Could not load segmentation data for {segmentation.run.name}")
        return None

    run = segmentation.run
    name = segmentation.name

    # Use input session_id if no output session_id specified
    if output_session_id is None:
        output_session_id = segmentation.session_id

    print(f"Fitting spline to segmentation {segmentation.session_id} in run {run.name}")

    try:
        # Fit spline to skeleton
        sampled_points, transforms, fitter, properties = fit_spline_to_skeleton(
            binary_volume=volume.astype(bool),
            spacing_distance=spacing_distance,
            smoothing_factor=smoothing_factor,
            degree=degree,
            connectivity_radius=connectivity_radius,
            compute_transforms=compute_transforms,
            curvature_threshold=curvature_threshold,
            max_iterations=max_iterations,
        )

        # Scale points to physical coordinates
        scaled_points = sampled_points * voxel_spacing

        print(f"Spline properties: {properties}")

        # Create output picks
        output_picks = run.new_picks(
            object_name=name,
            session_id=output_session_id,
            user_id=output_user_id,
            exist_ok=True,
        )

        # Store the picks with transformations
        if compute_transforms and transforms is not None:
            output_picks.from_numpy(scaled_points, transforms)
        else:
            output_picks.from_numpy(scaled_points)

        print(f"Created {len(scaled_points)} picks with session_id: {output_session_id}")
        return output_picks

    except Exception as e:
        print(f"Error fitting spline to segmentation: {e}")
        return None


def _fit_spline_worker(
    run: "CopickRun",
    segmentation_name: str,
    segmentation_user_id: str,
    session_id_pattern: str,
    spacing_distance: float,
    smoothing_factor: Optional[float],
    degree: int,
    connectivity_radius: float,
    compute_transforms: bool,
    curvature_threshold: float,
    max_iterations: int,
    output_session_id_template: Optional[str],
    output_user_id: str,
    voxel_spacing: float,
    root: "CopickRoot",
) -> Dict[str, Any]:
    """Worker function for batch spline fitting."""
    try:
        # Find matching segmentations using copick's official URI resolution
        matching_segmentations = get_copick_objects_by_type(
            root=run.root,
            object_type="segmentation",
            run_name=run.name,
            name=segmentation_name,
            user_id=segmentation_user_id,
            session_id=session_id_pattern,
            pattern_type="glob",
        )

        if not matching_segmentations:
            return {
                "processed": 0,
                "errors": [f"No segmentations found matching pattern '{session_id_pattern}' in {run.name}"],
                "picks_created": 0,
            }

        picks_created = 0
        errors = []

        for segmentation in matching_segmentations:
            # Determine output session ID
            if output_session_id_template:
                # Replace placeholders in template
                output_session_id = output_session_id_template.replace("{input_session_id}", segmentation.session_id)
            else:
                output_session_id = segmentation.session_id

            # Fit spline
            picks = fit_spline_to_segmentation(
                segmentation=segmentation,
                spacing_distance=spacing_distance,
                smoothing_factor=smoothing_factor,
                degree=degree,
                connectivity_radius=connectivity_radius,
                compute_transforms=compute_transforms,
                curvature_threshold=curvature_threshold,
                max_iterations=max_iterations,
                output_session_id=output_session_id,
                output_user_id=output_user_id,
                voxel_spacing=voxel_spacing,
            )

            if picks:
                picks_created += 1
            else:
                errors.append(f"Failed to fit spline to {segmentation.session_id}")

        return {
            "processed": 1,
            "errors": errors,
            "picks_created": picks_created,
            "segmentations_processed": len(matching_segmentations),
        }

    except Exception as e:
        return {"processed": 0, "errors": [f"Error processing {run.name}: {e}"], "picks_created": 0}


def fit_spline_batch(
    root: "CopickRoot",
    segmentation_name: str,
    segmentation_user_id: str,
    session_id_pattern: str,
    spacing_distance: float,
    smoothing_factor: Optional[float] = None,
    degree: int = 3,
    connectivity_radius: float = 2.0,
    compute_transforms: bool = True,
    curvature_threshold: float = 0.2,
    max_iterations: int = 5,
    output_session_id_template: Optional[str] = None,
    output_user_id: str = "spline",
    voxel_spacing: float = 1.0,
    run_names: Optional[List[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """
    Batch fit splines to segmentations across multiple runs.

    Args:
        root: The copick root containing runs to process
        segmentation_name: Name of the segmentations to process
        segmentation_user_id: User ID of the segmentations to process
        session_id_pattern: Regex pattern or exact session ID to match segmentations
        spacing_distance: Distance between consecutive sampled points along the spline
        smoothing_factor: Smoothing parameter for spline fitting (auto if None)
        degree: Degree of the spline (1-5). Default is 3.
        connectivity_radius: Maximum distance to consider skeleton points as connected. Default is 2.0.
        compute_transforms: Whether to compute orientations for picks. Default is True.
        curvature_threshold: Maximum allowed curvature before outlier removal. Default is 0.2.
        max_iterations: Maximum number of outlier removal iterations. Default is 5.
        output_session_id_template: Template for output session IDs. Use {input_session_id} as placeholder.
            If None, uses the same session ID as input.
        output_user_id: User ID for output picks. Default is "spline".
        voxel_spacing: Voxel spacing for coordinate scaling. Default is 1.0.
        run_names: List of run names to process. If None, processes all runs.
        workers: Number of worker processes. Default is 8.

    Returns:
        Dictionary with processing results and statistics
    """
    from copick.ops.run import map_runs

    runs_to_process = [run.name for run in root.runs] if run_names is None else run_names

    results = map_runs(
        callback=_fit_spline_worker,
        root=root,
        runs=runs_to_process,
        workers=workers,
        task_desc="Fitting splines to segmentations",
        segmentation_name=segmentation_name,
        segmentation_user_id=segmentation_user_id,
        session_id_pattern=session_id_pattern,
        spacing_distance=spacing_distance,
        smoothing_factor=smoothing_factor,
        degree=degree,
        connectivity_radius=connectivity_radius,
        compute_transforms=compute_transforms,
        curvature_threshold=curvature_threshold,
        max_iterations=max_iterations,
        output_session_id_template=output_session_id_template,
        output_user_id=output_user_id,
        voxel_spacing=voxel_spacing,
    )

    return results
