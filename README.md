# CorrSave User Manual
## Corrosion Images Classification to Improve Industrial Corrosion Management
This application use machine learning (convolutional neural network & image processing features) to classify the material corrosiveness.

### Option 1: Use our pre-trained model to test your data.
1. Download pre-trained model from https://github.com/nurafiqah78/CorrSave/blob/master/pre-trained_model%202019-05-16%2010.37.57
2. skip Tab 1 as you already has the scale attributes ready from the trained model.
3. Download scale attributes file from https://github.com/nurafiqah78/CorrSave
4. Tab 2 (Pre-Train Model): upload (1.) Model File and (2.) ScaleAttribute File
5. Tab 3 (Test Model): Upload your test data (1 image, URL or multiple images in older)

### Option 2: Train your own datasets.
1. Separate your datasets into 2 folders (0: corroded , 1: non corroded)
2. Tab 1 (Model Initialization): set your own parameters
3. Tab 2 (Train Model): upload class 0 and class 1 folders
4. click 'Calculate features' and 'start training'
5. Tab 3 (Test Model): Upload your test data (1 image, URL or multiple images in older)

You may refer to our 'Corrosion_Classification_Documentation.pdf' for more info.
