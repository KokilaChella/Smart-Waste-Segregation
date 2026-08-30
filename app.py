"""
app.py

Streamlit web app for the Smart Waste Segregation System.

Upload a photo of a waste item -> YOLOv8 classification model predicts
the specific item (one of 30 classes) -> app looks up the disposal
category (recyclable / organic / non-recyclable / hazardous) and shows
both, along with a disposal tip.

Run with:  streamlit run app.py
"""

import streamlit as st
from PIL import Image
from ultralytics import YOLO

from category_mapping import get_category, get_tip, CATEGORY_COLORS

MODEL_PATH = "runs/waste_cls/weights/best.pt"  # produced by train.py


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def format_class_name(raw_name: str) -> str:
    """'plastic_water_bottles' -> 'Plastic Water Bottles'"""
    return raw_name.replace("_", " ").title()


def main():
    st.set_page_config(page_title="Smart Waste Segregation", page_icon="♻️", layout="centered")

    st.title("♻️ Smart Waste Segregation System")
    st.write(
        "Upload a photo of a waste item and the model will identify it "
        "and tell you how to dispose of it."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Classifying..."):
            results = model.predict(image, verbose=False)
            result = results[0]

            top1_idx = result.probs.top1
            top1_conf = float(result.probs.top1conf)
            predicted_class = result.names[top1_idx]

            top5_idx = result.probs.top5
            top5_conf = result.probs.top5conf.tolist()

        category = get_category(predicted_class)
        tip = get_tip(predicted_class)
        color = CATEGORY_COLORS.get(category, "#616161")

        with col2:
            st.subheader(format_class_name(predicted_class))
            st.caption(f"Confidence: {top1_conf:.1%}")

            st.markdown(
                f"""
                <div style="padding: 12px; border-radius: 8px; background-color:{color}22; border: 1px solid {color};">
                    <span style="color:{color}; font-weight:600; text-transform:uppercase; font-size:0.85em;">
                        {category}
                    </span>
                    <p style="margin-top:6px; margin-bottom:0;">{tip}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("See top 5 predictions"):
            for idx, conf in zip(top5_idx, top5_conf):
                cls_name = result.names[idx]
                st.write(f"{format_class_name(cls_name)}: {conf:.1%}")

    else:
        st.info("Upload an image to get started.")

    st.divider()
    st.caption(
        "Note: disposal categories follow general recycling guidance and may "
        "vary by local municipality. Always check your local rules."
    )


if __name__ == "__main__":
    main()
