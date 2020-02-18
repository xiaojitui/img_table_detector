# img_table_detector

This is a CNN-RNN model to detect table areas in an image or a sacnned pdf file. 

*note: models in 'object_dection' use more advanced algorithms to do the detection. 
<br><br>
The algorithm is: 

(1) cut the image into several horizontal pieces (row like)

(2) for each horizontal piece, get its CNN features first, then use Bi_LSTM to connect them and get the row's horizontal features

(3) in vertical direction, use another Bi_LSTM to connect horizontal pieces, and get the row's vertical features

(4) finally, classify if the horizoanl piece (row like) is within or out of the table boundary box. 
<br><br>
The images for training should be put into 'train_images' folder. 

The ground truth is recorded in 'annotate.txt'. The format is: 'image path, table_boundary_x1, table_boundary_y1, table_boundary_x2, table_boundary_y2, table'

(optional) if the file is a scanned pdf, use "pdf_to_img" function in "preprocess.py" to convert pdf to image first, and saved the image in 'train_images' folder. The example code is: pdf_to_img(mypdffile, 'train_images', zoom = 2). 

<br><br>
To train the model, run: python img_table_detector.py

The trained model will be saved in 'saved_model' folder. 

