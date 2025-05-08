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

class HumanDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Human Detection GUI")

        self.image_list = []
        self.image_index = 0
        self.predictions = []

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

        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        if len(files) != 20:
            messagebox.showerror("Error", "Select a folder with exactly 20 images.")
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
            self.predictions[self.image_index] = random.randint(0, 1)

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
        data = [(fname, int(pred)) for (_, fname), pred in zip(self.image_list, self.predictions)]
        df = pd.DataFrame(data, columns=["Filename", "Prediction"])
        df.to_excel("predictions.xlsx", index=False)
        messagebox.showinfo("Exported", "Saved as predictions.xlsx")

if __name__ == "__main__":
    root = tk.Tk()
    app = HumanDetectionGUI(root)
    root.mainloop()
