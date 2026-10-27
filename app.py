"""Local Streamlit demo; all prediction logic is shared with the CLI."""
from io import BytesIO
from pathlib import Path
import os

import streamlit as st
import torch
from PIL import Image

from petseg.inference import load_model, load_upload, predict_mask, render_outputs, validate_deployment
from petseg.utils import read_json, sha256


@st.cache_resource
def cached_model(path, file_digest):
    return load_model(path)


def png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    st.set_page_config(page_title="Pet foreground studio", page_icon="🐾", layout="wide")
    torch.set_num_threads(1)
    st.title("Pet foreground studio")
    st.write("Separate a pet from its background, inspect the mask, and try a new background color.")
    st.caption("Binary semantic segmentation. Fur edges, clutter, multiple pets, and images without pets can fail. Boundary pixels are ignored in reported evaluation metrics; this is not alpha matting.")
    checkpoint = os.environ.get("PETSEG_CHECKPOINT", "artifacts/final/selected.pt")
    decision_path = os.environ.get("PETSEG_DECISION", "artifacts/final/decision.json")
    try:
        if not Path(checkpoint).is_file():
            raise FileNotFoundError("Trained checkpoint missing. Run the Colab training notebook, download artifacts, then follow README steps select and benchmark.")
        decision = read_json(decision_path)
        digest = sha256(checkpoint)
        model, state = cached_model(checkpoint, digest)
        size = validate_deployment(decision, digest, state)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        st.error(str(exc))
        st.info("See README.md for model preparation. No placeholder model is used.")
        st.stop()
    st.caption(f"Experiment {state['config']['experiment']} · {size} × {size} input · CPU")
    color_hex = st.color_picker("Replacement background", "#50A0F0")
    uploaded = st.file_uploader("Choose a JPEG or PNG (up to 10 MiB, 20 megapixels)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload a cat or dog photo to begin.")
        return
    try:
        image = load_upload(uploaded.getvalue())
        with st.spinner("Segmenting pet…"):
            mask = predict_mask(model, image, size)
            color = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
            overlay, composite = render_outputs(image, mask, color)
        columns = st.columns(4)
        for col, label, view in zip(columns, ["Original", "Pet mask", "Overlay", "New background"],
                                    [image, Image.fromarray(mask * 255), overlay, composite]):
            col.image(view, caption=label, width="stretch")
        st.download_button("Download mask PNG", png_bytes(Image.fromarray(mask * 255)), "pet_mask.png", "image/png")
        st.caption("Mask encoding: 0 = background, 255 = pet.")
        st.download_button("Download composite PNG", png_bytes(composite), "pet_composite.png", "image/png")
        st.download_button("Download overlay PNG", png_bytes(overlay), "pet_overlay.png", "image/png")
    except (ValueError, OSError, RuntimeError) as exc:
        st.error(f"Could not process image: {exc}")


if __name__ == "__main__":
    main()
