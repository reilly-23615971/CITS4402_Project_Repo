# CITS4402 Group Project
# Group Members:
# Felix Mavrodoglu (23720305)
# Jalil Inayat-Hussain (22751096)
# Reilly Evans (23615971)
# Code for GUI

# imports
import os
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox
import joblib
from PIL import Image, ImageTk
import pandas as pd
from projectFunctions import computeHOGFeatures, trainAndSaveModel



# Constants defining model parameters for when model is generated live
# Parameters were selected through our ablation studies
bin_count = 12
cell_dimensions = (8, 8)
block_dimensions = (2, 2)
norm_technique = 'L2-Hys'



class HumanDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection GUI")

        self.image_list = []
        self.image_index = 0
        self.predictions = []

        self.model = joblib.load(args.model_path)

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.filename_label = tk.Label(root, text="")
        self.filename_label.pack()

        self.prediction_label = tk.Label(root, text="", font=("Arial", 14))
        self.prediction_label.pack()

        # Buttons
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        tk.Button(
            self.btn_frame, text="Load Image Folder", command=self.load_folder
        ).grid(row=0, column=0)
        tk.Button(
            self.btn_frame, text="Previous", command=self.show_previous
        ).grid(row=0, column=1)
        tk.Button(
            self.btn_frame, text="Next", command=self.show_next
        ).grid(row=0, column=2)
        tk.Button(
            self.btn_frame, text="Export to predictions.xlsx", 
            command=self.export_predictions
        ).grid(row=0, column=3)

    # Load folder with images
    def load_folder(self):
        folder_path = filedialog.askdirectory(initialdir=os.getcwd())
        if not folder_path:
            return

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg', 'pgm'))])
        if len(files) == 0:
            messagebox.showerror("Error", "No images found in folder.")
            return

        self.image_list = [(os.path.join(folder_path, f), f) for f in files]
        self.predictions = [None] * len(self.image_list)
        self.image_index = 0
        self.show_image()

    # Display the current image and its filename
    def show_image(self):
        path, fname = self.image_list[self.image_index]
        image = Image.open(path).resize((256, 512))
        photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=photo)
        self.image_label.image = photo

        self.filename_label.config(text=f"Filename: {fname}")

        if self.predictions[self.image_index] is None:
            features = computeHOGFeatures(
                path, numberOfBins=bin_count, cellDimensions=cell_dimensions, 
                blockDimensions=block_dimensions, normalisationTechnique=norm_technique
            )
            self.predictions[self.image_index] = self.model.predict([features])[0]

        label = "Human" if self.predictions[self.image_index] == 1 else "Non-Human"
        self.prediction_label.config(text=f"Prediction: {label}")

    # Show the next image in the list
    def show_next(self):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.show_image()

    # Show the previous image in the list
    def show_previous(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.show_image()

    # Export predictions to an Excel file
    # Each row contains the filename and the prediction (0 or 1)
    def export_predictions(self):
        # Ensure images have been loaded
        if not self.image_list: messagebox.showerror(
            "Error", 
            "No images loaded. Please load an image folder before exporting predictions."
        )
        else:
            # Make sure features have been calculated for each image
            for i, (path, _) in enumerate(self.image_list):
                if self.predictions[i] is None:
                    features = computeHOGFeatures(
                        path, numberOfBins=bin_count, 
                        cellDimensions=cell_dimensions, 
                        blockDimensions=block_dimensions, 
                        normalisationTechnique=norm_technique
                    )
                    self.predictions[i] = self.model.predict([features])[0]
            # Save data as Excel file
            data = [(fname, int(pred)) for (_, fname), pred in zip(self.image_list, self.predictions)]
            df = pd.DataFrame(data, columns=["Filename", "Prediction"])
            df.to_excel("predictions.xlsx", index=False)
            messagebox.showinfo("Exported", "Saved as predictions.xlsx")



# Initialize argument parser
parser = argparse.ArgumentParser(
    description="Opens a GUI for predicting whether images contain humans using HOG features."
)

# Specify parameters, ensuring existing model and dataset to create 
# model are mutually exclusive to one another
parser.add_argument(
    "-m", "--model_path", default='svm_model.joblib', 
    help = "Path to trained SVM model as a .joblib file"
)
parser.add_argument(
    "-d", "--dataset_path", 
    help = ((
        "Path to dataset to train the model on as a .tar.gz file. "
        "If included, the code will attempt to train an SVM model on this dataset "
        "and save it to model_path before opening the GUI."
    ))
)

if __name__ == "__main__":
    # Get arguments from command line
    args = parser.parse_args()
    # Create model if dataset to use was specified
    if args.dataset_path:
        if not os.path.isfile(args.dataset_path):
            raise FileNotFoundError(
                f'The specified dataset zip file at {args.dataset_path} does not exist.'
            )
        print(f"Training model with data from {args.dataset_path}...")
        trainAndSaveModel(
            args.dataset_path, outputFile = args.model_path,
            numberOfBins=bin_count, cellDimensions=cell_dimensions, 
            blockDimensions=block_dimensions, 
            normalisationTechnique=norm_technique
        )
        print(f"Model saved to {args.model_path}.")
    else:
        if not os.path.isfile(args.model_path):
            raise FileNotFoundError(
                f'The SVM model file at {args.model_path} does not exist.'
            )
    # Run the GUI
    print("Opening GUI...")
    root = tk.Tk()
    app = HumanDetectionGUI(root)
    root.mainloop()

    print("The GUI has been closed.")
    # Delete the saved model after GUI closes if it was created just now
    if args.dataset_path and os.path.exists(args.model_path):
        os.remove(args.model_path)
        print(f"Deleted model file: {args.model_path}")
