import torch
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import os
from tkinter import Tk, filedialog

def predict_damage(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reconstruct architecture shell
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Match the validation tensor transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        
    classes = ['damaged', 'undamaged']  # Inferred alphabetically from folder names
    return classes[preds[0].item()]

if __name__ == "__main__":
    # 1. Hide the ugly blank Tkinter root window
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True) # Bring the file dialog to the front of your screen

    # 2. Define your project's data directory as the starting point
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

    print("Opening file selector... Please choose an image to analyze.")

    # 3. Pop open the Windows File Explorer dialog
    selected_image_path = filedialog.askopenfilename(
        initialdir=DEFAULT_DATA_DIR,
        title="Select a Post-Storm Aerial Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif *.tiff *.webp")]
    )

    # 4. Run prediction only if the user actually picked a file
    if selected_image_path:
        MODEL_PATH = "hurricane_damage_resnet50.pth"
        
        # Verify the model weights exist before running inference
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Could not find '{MODEL_PATH}' in your folder. Did you run train.py first?")
        else:
            print(f"\nAnalyzing: {os.path.basename(selected_image_path)}")
            result = predict_damage(selected_image_path, MODEL_PATH)
            print(f"Prediction Result: {result.upper()}")
    else:
        print("Selection canceled. No image analyzed.")
















        