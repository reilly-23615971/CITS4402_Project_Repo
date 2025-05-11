File Names and Purposes:
- `INRIA.tar` and `DC-ped-dataset_base.tar` are the tarfiles storing the INRIA and Daimler datasets respectively; use them with createDataset to generate training, testing and GUI images from the examples within.
- Within the `Daimler` folder, `DaimlerTrain.tar.gz`, `DaimlerTest.tar.gz` and `DaimlerTestingImages` are the Daimler training and testing sets as well as the unzipped images for demonstrating the GUI with. There are 500 images in the training set, 200 in the testing set and 20 in the GUI folder, each split evenly between positive and negative instances.
- Within the `INRIASmallDataset` folder, `INRIASmallTrain.tar.gz`, `INRIASmallTest.tar.gz` and `INRIASmallTestingImages` are the INRIA training/testing sets and GUI folder generated with the same number of instances as the Daimler set. While their higher-detail images should be more useful than Daimler, these datasets are still pretty small. They're probably fine to use for the ablation studies if the bigger dataset takes too long to compute or something.
- Within the `INRIAFullDataset` folder, `INRIAFullTrain.tar.gz`, `INRIAFullTest.tar.gz` and the unlabelled `Testing Images` are the INRIA training/testing sets and GUI folder generated with the majority of the instances in the dataset. In total, there are 3600 images in the training set, 800 in the testing set and 20 in the GUI folder, following the standard 80/20 split typically used for machine learning. Use this for Phase 2/the final model parameters, and anywhere else where it's feasible.





The train_set and test_set files are compressed sets made with the Daimler dataset and the latest version of the createDataset function, with 200 test images and 500 training images. Since this will likely be replaced with another dataset, be ready to change file names in your notebooks and similar.