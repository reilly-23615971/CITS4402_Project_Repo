# CITS4402 Project: Histograms of Oriented Gradients for Human Detection

Group Members:
- Felix Mavrodoglu (23720305)
- Jalil Inayat-Hussain (22751096)
- Reilly Evans (23615971)

This project uses the following third-party Python libraries: 

- NumPy
- Pandas
- Joblib
- Pillow
- openpyxl
- scikit-learn
- scikit-image

Ensure that these libraries are installed on your Python distribution before running the GUI, or it may not function correctly.

The GUI can be opened by simply running the `GUIcode.py` script. The trained model that the GUI uses can be found at `Others/svm_model.joblib`. If a different model needs to be used and has been saved to a `.joblib` file, specify the model file path with `-m path_to_model`. If you wish to train a model directly from one of our dataset tarballs and use it for the GUI, specify the dataset path with `-d path_to_dataset`; note that this will create a folder with the expanded dataset contents in the current directory.

If the dataset used for training/testing the model is required, it and other files used during the project's lifespan can be found at our project's GitHub repository: https://github.com/reilly-23615971/CITS4402_Project_Repo

References for datasets used over the course of this project:

Daimler: S. Munder and D. M. Gavrila. "An Experimental Study on Pedestrian Classification". IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 28, no. 11, pp.1863-1868, November 2006.

INRIA: Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7263–7271, 2005.