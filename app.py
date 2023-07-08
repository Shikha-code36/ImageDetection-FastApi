import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import ImageEnhance, Image

# Streamlit UI
def main():
    st.title("Image Recognition App")

    # Upload image file
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Perform object detection on the uploaded image
        if st.button("Detect Objects"):
            # Make a request to the FastAPI endpoint
            files = {"file": uploaded_file}
            response = requests.post("http://127.0.0.1:8000/detect_objects/", files=files)

            # Process the response
            if response.status_code == 200:
                detection_results = response.json()
                boxes = detection_results["boxes"]
                scores = detection_results["scores"]
                classes = detection_results["classes"]

                # Convert the PIL image to matplotlib format
                image = plt.imread(uploaded_file)
                fig, ax = plt.subplots()
                ax.imshow(image)

                # Adjust brightness of the image
                brightness = ImageEnhance.Brightness(Image.fromarray((image * 255).astype('uint8')))
                enhanced_image = brightness.enhance(1.5)
                ax.imshow(enhanced_image)

                # Draw bounding boxes on the image with confidence threshold
                confidence_threshold = 0.5
                for box, score, cls in zip(boxes, scores, classes):
                    if score >= confidence_threshold:
                        xmin, ymin, xmax, ymax = box
                        width = xmax - xmin
                        height = ymax - ymin
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor="r", facecolor="none")
                        ax.add_patch(rect)
                        ax.text(xmin, ymin, f"{cls} ({score:.2f})", fontsize=8, color="r")

                # Display the image with bounding boxes
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.error("Error performing object detection.")

if __name__ == "__main__":
    main()
