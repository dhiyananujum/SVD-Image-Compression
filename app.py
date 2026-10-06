import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SVD Image Compression",
    page_icon="🖼️",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🖼️ SVD-Based RGB Image Compression")
st.subheader("Application of Linear Algebra in Image Processing")

st.write(
    """
    This application demonstrates how Singular Value Decomposition (SVD)
    can be used to compress a color image using low-rank matrix approximation.
    Each RGB channel is independently decomposed using SVD.
    """
)

# --------------------------------------------------
# MATHEMATICAL CONCEPT
# --------------------------------------------------

with st.expander("📐 Mathematical Concept — SVD"):

    st.markdown(
        """
        A color image consists of three matrices:

        - **Red channel (R)**
        - **Green channel (G)**
        - **Blue channel (B)**

        Each channel is decomposed using:

        **A = UΣVᵀ**

        For compression, only the first **k** singular values are retained:

        **Aₖ = UₖΣₖVₖᵀ**

        This is performed independently for the R, G and B channels.

        The three reconstructed channels are then combined to produce
        the compressed RGB image.
        """
    )

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a color image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # --------------------------------------------------
    # LOAD RGB IMAGE
    # --------------------------------------------------

    original_image = Image.open(uploaded_file).convert("RGB")

    image_array = np.array(
        original_image,
        dtype=np.float64
    )

    height, width, channels = image_array.shape

    st.success(
        f"Image loaded successfully! "
        f"Image size: {width} × {height} pixels | "
        f"Channels: RGB"
    )

    # --------------------------------------------------
    # DISPLAY ORIGINAL
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original RGB Image")

        st.image(
            original_image,
            width="stretch"
        )

    # --------------------------------------------------
    # SPLIT RGB CHANNELS
    # --------------------------------------------------

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    # --------------------------------------------------
    # PERFORM SVD ON EACH CHANNEL
    # --------------------------------------------------

    st.markdown("### Performing SVD on RGB Channels...")

    with st.spinner("Computing SVD..."):

        U_r, S_r, VT_r = np.linalg.svd(
            red,
            full_matrices=False
        )

        U_g, S_g, VT_g = np.linalg.svd(
            green,
            full_matrices=False
        )

        U_b, S_b, VT_b = np.linalg.svd(
            blue,
            full_matrices=False
        )

    st.success(
        "SVD successfully computed for Red, Green and Blue channels!"
    )

    # --------------------------------------------------
    # MAXIMUM RANK
    # --------------------------------------------------

    max_rank = min(height, width)

    # --------------------------------------------------
    # SINGULAR VALUE GRAPHS
    # --------------------------------------------------

    st.markdown("### Singular Value Distribution")

    fig, ax = plt.subplots()

    ax.plot(
        S_r,
        label="Red Channel"
    )

    ax.plot(
        S_g,
        label="Green Channel"
    )

    ax.plot(
        S_b,
        label="Blue Channel"
    )

    ax.set_xlabel("Singular Value Index")
    ax.set_ylabel("Singular Value Magnitude")
    ax.set_title("Singular Values of RGB Channels")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
    # RANK SELECTION
    # --------------------------------------------------

    st.markdown("### Choose Compression Rank")

    k = st.slider(
        "Number of singular values to retain (k)",
        min_value=1,
        max_value=max_rank,
        value=min(50, max_rank),
        step=1
    )

    # --------------------------------------------------
    # RGB LOW-RANK RECONSTRUCTION
    # --------------------------------------------------

    red_compressed = (
        U_r[:, :k]
        @ np.diag(S_r[:k])
        @ VT_r[:k, :]
    )

    green_compressed = (
        U_g[:, :k]
        @ np.diag(S_g[:k])
        @ VT_g[:k, :]
    )

    blue_compressed = (
        U_b[:, :k]
        @ np.diag(S_b[:k])
        @ VT_b[:k, :]
    )

    # --------------------------------------------------
    # CLIP PIXEL VALUES
    # --------------------------------------------------

    red_compressed = np.clip(
        red_compressed,
        0,
        255
    )

    green_compressed = np.clip(
        green_compressed,
        0,
        255
    )

    blue_compressed = np.clip(
        blue_compressed,
        0,
        255
    )

    # --------------------------------------------------
    # COMBINE RGB CHANNELS
    # --------------------------------------------------

    compressed_array = np.stack(
        [
            red_compressed,
            green_compressed,
            blue_compressed
        ],
        axis=2
    )

    compressed_array = compressed_array.astype(
        np.uint8
    )

    compressed_image = Image.fromarray(
        compressed_array,
        mode="RGB"
    )

    # --------------------------------------------------
    # DISPLAY COMPRESSED IMAGE
    # --------------------------------------------------

    with col2:

        st.markdown(
            f"### Reconstructed Image (k = {k})"
        )

        st.image(
            compressed_image,
            width="stretch"
        )

    # --------------------------------------------------
    # STORAGE CALCULATION
    # --------------------------------------------------

    # Original RGB image:
    # height × width × 3

    original_storage = (
        height *
        width *
        3
    )

    # For each RGB channel:
    # U_k = height × k
    # Sigma_k = k
    # V_k = k × width
    #
    # Multiply by 3 because there are 3 channels.

    compressed_storage = 3 * (
        height * k +
        k +
        width * k
    )

    compression_ratio = (
        original_storage /
        compressed_storage
    )

    storage_reduction = (
        1 -
        compressed_storage /
        original_storage
    ) * 100

    # --------------------------------------------------
    # RECONSTRUCTION ERROR
    # --------------------------------------------------

    original_float = image_array

    reconstructed_float = compressed_array.astype(
        np.float64
    )

    reconstruction_error = (
        np.linalg.norm(
            original_float -
            reconstructed_float
        )
        /
        np.linalg.norm(
            original_float
        )
    ) * 100

    # --------------------------------------------------
    # ENERGY RETENTION
    # --------------------------------------------------

    total_energy = (
        np.sum(S_r ** 2) +
        np.sum(S_g ** 2) +
        np.sum(S_b ** 2)
    )

    retained_energy = (
        np.sum(S_r[:k] ** 2) +
        np.sum(S_g[:k] ** 2) +
        np.sum(S_b[:k] ** 2)
    )

    energy_retained = (
        retained_energy /
        total_energy
    ) * 100

    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.markdown("## 📊 Compression Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Compression Ratio",
            f"{compression_ratio:.2f} : 1"
        )

    with c2:

        st.metric(
            "Storage Reduction",
            f"{storage_reduction:.2f}%"
        )

    with c3:

        st.metric(
            "Energy Retained",
            f"{energy_retained:.2f}%"
        )

    with c4:

        st.metric(
            "Reconstruction Error",
            f"{reconstruction_error:.2f}%"
        )

    # --------------------------------------------------
    # MATRIX INFORMATION
    # --------------------------------------------------

    st.markdown("## 🔢 Matrix Information")

    info1, info2, info3 = st.columns(3)

    with info1:

        st.write(
            f"**Image Dimensions:** "
            f"{height} × {width}"
        )

        st.write(
            f"**Color Channels:** 3 (RGB)"
        )

        st.write(
            f"**Original Values:** "
            f"{original_storage:,}"
        )

    with info2:

        st.write(
            f"**Selected Rank:** {k}"
        )

        st.write(
            f"**Singular Values per Channel:** {k}"
        )

        st.write(
            "**SVD Operations:** 3"
        )

    with info3:

        st.write(
            f"**Compressed Values:** "
            f"{compressed_storage:,}"
        )

        st.write(
            f"**Storage Reduction:** "
            f"{storage_reduction:.2f}%"
        )

    # --------------------------------------------------
    # ENERGY CURVE
    # --------------------------------------------------

    st.markdown("## 📈 Information Retained")

    energy_r = np.cumsum(S_r ** 2) / np.sum(S_r ** 2)
    energy_g = np.cumsum(S_g ** 2) / np.sum(S_g ** 2)
    energy_b = np.cumsum(S_b ** 2) / np.sum(S_b ** 2)

    fig2, ax2 = plt.subplots()

    ax2.plot(
        energy_r * 100,
        label="Red"
    )

    ax2.plot(
        energy_g * 100,
        label="Green"
    )

    ax2.plot(
        energy_b * 100,
        label="Blue"
    )

    ax2.axvline(
        k - 1,
        linestyle="--",
        label=f"Selected k = {k}"
    )

    ax2.set_xlabel("Rank k")
    ax2.set_ylabel("Energy Retained (%)")
    ax2.set_title("Cumulative Energy Retained")
    ax2.legend()

    st.pyplot(fig2)

    # --------------------------------------------------
    # CHANNEL PREVIEW
    # --------------------------------------------------

    with st.expander("🔴🟢🔵 View Individual RGB Channels"):

        channel1, channel2, channel3 = st.columns(3)

        with channel1:

            st.markdown("### Red Channel")

            st.image(
                red.astype(np.uint8),
                width="stretch"
            )

        with channel2:

            st.markdown("### Green Channel")

            st.image(
                green.astype(np.uint8),
                width="stretch"
            )

        with channel3:

            st.markdown("### Blue Channel")

            st.image(
                blue.astype(np.uint8),
                width="stretch"
            )

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------

    buffer = io.BytesIO()

    compressed_image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Reconstructed RGB Image",
        data=buffer.getvalue(),
        file_name=f"svd_rgb_compressed_k{k}.png",
        mime="image/png"
    )

else:

    st.info(
        "👆 Upload a JPG or PNG image to begin RGB SVD compression."
    )