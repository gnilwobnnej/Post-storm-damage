import os #handles file paths and checks for file existence
import tkinter as tk #GUI library for building the user interface
from tkinter import filedialog, messagebox #pop up screens
from PIL import Image, ImageTk #opening files and such
import torch #the core PyTorch library for tensor operations and model handling
import torch.nn as nn #for defining the neural network architecture and layers
from torchvision import models, transforms #pre-trained model architectures and image transformation utilities


# 1. CORE AI INFERENCE FUNCTION
# checks hardware, if cuda is availabe it will use that otherwise cpu.
def predict_damage(image_path, model_weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reconstruct architecture shell
    #rips off the default 1000 class output layer and replaces it with 2node output layer
    #to match damaged/undamaged config.
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    # pours trained matrix math weights from the .pth file into the structure.
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    # puts the model in evaluation mode which turns off dropout and batchnorm updates so 
    # we get consistent results during inference.
    model.eval()
    
    # Processing pipeling. Raw drone or satelite photos can be massive and different
    # sizes. We need to resize and crop them to 224x224 pixels which is what 
    # ResNet50 expects. Converts the pixels into mathematical tensors
    # between 0.0 and 1.0 and normalizes the colors. 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Open the image file, convert to RGB (in case it's grayscale or has an alpha 
    # channel)changes the data shape from a single image [3, 224, 224] to a 
    # mini-batch of size one [1, 3, 224, 224]. PyTorch models always expect a batch array,
    #  even if you are only passing a single image.
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    

    # Run the image through the model and get the predicted class index. 
    # The output is a tensor of raw scores for each class, so we use torch.max to 
    # find the index of the highest score which corresponds to the predicted class.
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)

    # Map the predicted index to a human-readable label. In our case, 0 = damaged, 
    # 1 = undamaged.    
    classes = ['damaged', 'undamaged']
    return classes[preds[0].item()]


# 2. GUI INTERACTION LOGIC
# This function is called when the user clicks the "Upload & Analyze Patch" button. 
# It opens a file dialog for the user to select an image, displays the selected image 
# in the GUI, runs the AI prediction on it, and updates the result label with the 
# prediction outcome.
def select_and_analyze_image():
    MODEL_PATH = "hurricane_damage_resnet50.pth"
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", f"Could not find '{MODEL_PATH}'.\nDid you run train.py first?")
        return

    # Open the file picker starting in the 'data' directory where we expect users 
    # to have their post-storm images.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, 'data')
    # If the data directory doesn't exist, start in the script directory instead
    selected_path = filedialog.askopenfilename(
        initialdir=default_data_dir,
        title="Select a Post-Storm Aerial Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.tif *.tiff *.webp")]
    )
    
    if not selected_path:
        return # User cancelled selection

    try:
        # Display the Picture
        # Open and resize the image so it fits neatly inside our GUI display box
        display_img = Image.open(selected_path)
        display_img.thumbnail((380, 380)) # Scale down proportionally to max 380x380 px
        
        # Convert PIL image to a format Tkinter can draw
        tk_photo = ImageTk.PhotoImage(display_img)
        image_panel.config(image=tk_photo)
        image_panel.image = tk_photo # CRITICAL: Keep a memory reference or image disappears
        
        # Run Model Prediction
        result_label.config(text="Analyzing image...", fg="blue")
        root.update() # Refresh screen immediately to show loading text
        
        prediction = predict_damage(selected_path, MODEL_PATH)
        
        # Update Results UI
        # Dynamic color-coding: Red for danger/damage, Green for normal/clear
        if prediction == "damaged":
            result_label.config(text="RESULT: DAMAGED", fg="#d9534f") # Crimson Red
        else:
            result_label.config(text="RESULT: UNDAMAGED", fg="#5cb85c") # Emerald Green
            
    except Exception as e:
        messagebox.showerror("Processing Error", f"An error occurred while loading the image:\n{str(e)}")


# 3. INTERFACE LAYOUT SPECIFICATION
if __name__ == "__main__":
    # Initialize the primary window
    root = tk.Tk()
    root.title("Post-Storm Damage Assessment AI")
    root.geometry("450x600")
    root.configure(bg="#f5f5f5") # Clean neutral background light gray

    # Header title
    title_lbl = tk.Label(root, text="Hurricane Damage Detector", font=("Arial", 16, "bold"), bg="#f5f5f5", fg="#333333")
    title_lbl.pack(pady=15)

    # Image placeholder box frame
    image_frame = tk.LabelFrame(root, text=" Aerial Image Viewport ", width=400, height=400, bg="#ffffff", bd=2, relief="groove")
    image_frame.pack_propagate(False) # Keep frame rigid regardless of image size
    image_frame.pack(pady=10)

    # Label inside the frame that actually holds the pixel graphic
    image_panel = tk.Label(image_frame, text="No Image Selected", font=("Arial", 10, "italic"), bg="#ffffff", fg="#888888")
    image_panel.pack(fill="both", expand=True)

    # Prediction Output Text Box
    result_label = tk.Label(root, text="Select an image below to analyze.", font=("Arial", 11, "bold"), bg="#f5f5f5", fg="#666666")
    result_label.pack(pady=15)

    # Main Action Button
    select_btn = tk.Button(root, text="Upload & Analyze Patch", font=("Arial", 11, "bold"), bg="#0275d8", fg="#ffffff", activebackground="#025aa5", activeforeground="#ffffff", padx=15, pady=8, cursor="hand2", command=select_and_analyze_image)
    select_btn.pack(pady=5)

    # Launch loop window thread execution
    root.mainloop()