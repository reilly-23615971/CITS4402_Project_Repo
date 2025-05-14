# CITS4402 Group Project
# Group Members:
# Felix Mavrodoglu (23720305)
# Jalil Inayat-Hussain (22751096)
# Reilly Evans (23615971)
# Code for GUI

# imports
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pandas as pd
import os
import random
from projectFunctions import computeHOGFeatures
import joblib
from train_model import train_and_save_model


class HumanDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection GUI")

        self.image_list = []
        self.image_index = 0
        self.predictions = []

        self.model = joblib.load("svm_model.joblib")

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

        tk.Button(self.btn_frame, text="Load Image Folder", command=self.load_folder).grid(row=0, column=0)
        tk.Button(self.btn_frame, text="Previous", command=self.show_previous).grid(row=0, column=1)
        tk.Button(self.btn_frame, text="Next", command=self.show_next).grid(row=0, column=2)
        tk.Button(self.btn_frame, text="Export to predictions.xlsx", command=self.export_predictions).grid(row=0, column=3)

    # Load folder with images
    # Ensure the folder contains exactly 20 images
    def load_folder(self):
        folder_path = filedialog.askdirectory()
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
    # Randomly assign a prediction (0 or 1) if not already assigned
    def show_image(self):
        path, fname = self.image_list[self.image_index]
        image = Image.open(path).resize((256, 512))
        photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=photo)
        self.image_label.image = photo

        self.filename_label.config(text=f"Filename: {fname}")

        if self.predictions[self.image_index] is None:
            features = computeHOGFeatures(path)
            self.predictions[self.image_index] = self.model.predict([features])[0]

        label = "Human" if self.predictions[self.image_index] == 1 else "Non-Human"
        self.prediction_label.config(text=f"Prediction: {label}")

    # Show the next image in the list
    # If at the end, loop back to the start
    def show_next(self):
        if self.image_index < len(self.image_list) - 1:
            self.image_index += 1
            self.show_image()

    # Show the previous image in the list
    # If at the start, loop back to the end
    def show_previous(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.show_image()

    # Export predictions to an Excel file
    # Each row contains the filename and the prediction (0 or 1)
    def export_predictions(self):
        for i, (path, _) in enumerate(self.image_list):
            if self.predictions[i] is None:
                features = computeHOGFeatures(path)
                self.predictions[i] = self.model.predict([features])[0]



        data = [(fname, int(pred)) for (_, fname), pred in zip(self.image_list, self.predictions)]
        df = pd.DataFrame(data, columns=["Filename", "Prediction"])
        df.to_excel("predictions.xlsx", index=False)
        messagebox.showinfo("Exported", "Saved as predictions.xlsx")

if __name__ == "__main__":
    train_and_save_model()
    root = tk.Tk()
    app = HumanDetectionGUI(root)
    root.mainloop()



    # Delete the saved model after GUI closes
    model_path = "svm_model.joblib"
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"🧹 Deleted model file: {model_path}")
