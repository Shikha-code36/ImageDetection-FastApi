# Image Recognition App

This is a simple image recognition app built using FastAPI and Streamlit. It allows users to upload an image, perform object detection using a pre-trained model, and visualize the detection results with bounding boxes.

## Demo

Check out the [demo video](https://www.loom.com/share/3850c640639f4344b076fa457b7052b2?sid=8ad0bb29-5fbc-45cb-bdd1-e3e3becba627) to see the app in action.


## Requirements

- Python 3.7 or higher
- FastAPI
- Streamlit
- Torch
- torchvision
- Pillow
- Matplotlib
- Requests

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/Shikha-code36/ImageDetection-FastApi.git
   ```

2. Navigate to the project directory:

   ```
   cd ImageDetection-FastApi
   ```

3. Install the required packages using pip:

    ```
    pip install -r requirements.txt
    ```

## Usage

1. Start the FastAPI server:

    ```
    uvicorn app.main:app --reload 
    # for development environment with auto reload on code changes
    ```

    The server will be running at http://127.0.0.1:8000.

2. Run the Streamlit app:

   ```
   streamlit run app.py
   ```

   The app will be accessible at http://localhost:8501 in your web browser.

3. Use the Image Recognition App:
    - Upload an image by clicking on the "Choose an image" button.
    - The uploaded image will be displayed.
    - Click on the "Detect Objects" button to perform object detection.
    - Bounding boxes will be drawn around the detected objects with their corresponding class labels and confidence scores.

Note: Make sure the FastAPI server is running before using the Streamlit app.

## Model and Classes

- The app uses the `ssdlite320_mobilenet_v3_large model` from the torchvision library for object detection.
- The available classes for object detection are defined in the `CLASSES` list in `main.py`. Modify the list as per your requirements.

## License

This project is licensed under the [MIT License](LICENSE).