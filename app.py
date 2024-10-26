import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters
from skimage import data, color, util
from skimage.util import view_as_windows
from skimage.io import imread
from typing import Tuple, Optional
import logging
import multiprocessing as mp
from dataclasses import dataclass
import concurrent.futures
import time
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DenoiseParams:
    patch_size: int = 64
    overlap: int = 16
    noise_variance: float = 0.01
    heat_tau: float = 10
    filter_method: str = 'exact'
    num_processes: int = max(1, mp.cpu_count() - 1)

def create_patch_graph(patch_size: int) -> graphs.Grid2d:
    graph = graphs.Grid2d(patch_size, patch_size)
    graph.compute_fourier_basis()
    return graph

class PatchResult:
    def __init__(self, filtered_patch: np.ndarray, position: Tuple[int, int, int, int]):
        self.filtered_patch = filtered_patch
        self.position = position

def process_single_patch(args: Tuple[np.ndarray, DenoiseParams, Tuple[int, int], graphs.Grid2d]) -> PatchResult:
    patch, params, (i, j), patch_graph = args
    low_pass_filter = filters.Heat(patch_graph, tau=params.heat_tau)
    filtered_patch = low_pass_filter.filter(patch.flatten(), method=params.filter_method)
    filtered_patch = filtered_patch.reshape(patch.shape)
    start_i = i * (params.patch_size - params.overlap)
    end_i = start_i + params.patch_size
    start_j = j * (params.patch_size - params.overlap)
    end_j = start_j + params.patch_size
    return PatchResult(filtered_patch, (start_i, end_i, start_j, end_j))

class ImageDenoiser:
    def __init__(self, params: Optional[DenoiseParams] = None):
        self.params = params or DenoiseParams()
        self._validate_params()
        self.patch_graph = create_patch_graph(self.params.patch_size)

    def _validate_params(self) -> None:
        if self.params.patch_size <= 0:
            raise ValueError("Patch size must be positive")
        if self.params.overlap >= self.params.patch_size:
            raise ValueError("Overlap must be smaller than patch size")
        if self.params.noise_variance < 0:
            raise ValueError("Noise variance must be non-negative")
        if self.params.heat_tau <= 0:
            raise ValueError("Heat tau must be positive")
        if self.params.num_processes <= 0:
            raise ValueError("Number of processes must be positive")

    def load_image(self, image) -> np.ndarray:
        if isinstance(image, str):
            try:
                image = imread(image)
            except Exception as e:
                logger.error(f"Error loading image: {e}")
                raise
        elif image is None:
            image = data.astronaut()
        if len(image.shape) == 3:
            image = color.rgb2gray(image)
        return util.img_as_float(image)

    def add_noise(self, image: np.ndarray) -> np.ndarray:
        return util.random_noise(image, mode='gaussian', var=self.params.noise_variance)

    def denoise(self, image: np.ndarray, progress_bar) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Starting parallel denoising process with {self.params.num_processes} processes...")
        noisy_image = self.add_noise(image)
        patches = view_as_windows(
            noisy_image,
            (self.params.patch_size, self.params.patch_size),
            step=self.params.patch_size - self.params.overlap
        )
        process_args = []
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                process_args.append((patches[i, j], self.params, (i, j), self.patch_graph))

        denoised_image = np.zeros_like(noisy_image)
        weight_mask = np.zeros_like(noisy_image)
        total_patches = len(process_args)
        progress_bar.progress(0)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.params.num_processes) as executor:
            futures = {executor.submit(process_single_patch, args): idx for idx, args in enumerate(process_args)}
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    start_i, end_i, start_j, end_j = result.position
                    denoised_image[start_i:end_i, start_j:end_j] += result.filtered_patch
                    weight_mask[start_i:end_i, start_j:end_j] += 1
                    progress_bar.progress((idx + 1) / total_patches)
                except Exception as e:
                    logger.error(f"Error processing patch {idx}: {e}")

        denoised_image /= np.maximum(weight_mask, 1e-10)
        logger.info("Parallel denoising completed successfully")
        return noisy_image, denoised_image

def main():
    st.title("Image Denoising with Graph Signal Processing")
    st.write("Upload an image and customize the denoising parameters.")
    st.sidebar.header("Denoising Parameters")
    patch_size = st.sidebar.slider("Patch Size", 32, 128, 64, 16)
    overlap = st.sidebar.slider("Overlap", 8, patch_size-1, 16, 8)
    noise_variance = st.sidebar.slider("Noise Variance", 0.01, 0.5, 0.1, 0.01)
    heat_tau = st.sidebar.slider("Heat Tau", 1, 50, 10, 1)
    num_processes = st.sidebar.slider("Number of Processes", 1, mp.cpu_count(), max(1, (mp.cpu_count()-1)//2))
    uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None or st.button("Run with sample image"):
        try:
            params = DenoiseParams(
                patch_size=patch_size,
                overlap=overlap,
                noise_variance=noise_variance,
                heat_tau=heat_tau,
                num_processes=num_processes
            )
            denoiser = ImageDenoiser(params)
            if uploaded_file is not None:
                image = imread(io.BytesIO(uploaded_file.read()))
            else:
                image = data.astronaut()
            image = denoiser.load_image(image)
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("Adding noise to image...")
            noisy_image = denoiser.add_noise(image)
            status_text.text("Denoising image...")
            start_time = time.time()
            noisy_image, denoised_image = denoiser.denoise(image, progress_bar)
            processing_time = time.time() - start_time
            status_text.text(f"Processing completed in {processing_time:.2f} seconds!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Original")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("Noisy")
                st.image(noisy_image, use_column_width=True)
            with col3:
                st.subheader("Denoised")
                st.image(denoised_image, use_column_width=True)

        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            logger.error(f"Error during denoising: {e}")

if __name__ == "__main__":
    main()
