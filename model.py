import gradio as gr
from PIL import Image
from ultralytics import YOLO
import tempfile

# Load your trained YOLO model
model = YOLO("best.pt")

def detect_objects(image):
    # Save the uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_filename = temp_file.name
        image.save(temp_filename)
    
    # Perform inference on the image and get results
    results = model(temp_filename)
    
    # Extract the first (and only) result
    result = results[0]
    
    # Count the number of detected objects (bounding boxes)
    detected_objects_count = len(result.boxes)
    
    # Create a result image with detections
    result_image = result.show()
    
    return image, result_image, detected_objects_count

# Create the Gradio interface with updated API
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Uploaded Image"),
        gr.Image(type="pil", label="Result Image with Detections"),
        gr.Textbox(label="Total number of detected objects")
    ],
    title="Log Counting by RV",
    description="Upload an image to detect and count objects."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch(share=True)
