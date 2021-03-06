If find the host ip of computer then write command on `cmd` 
 &emsp; &emsp; `ipconfig` for window
 &emsp; &emsp;  `ifconfig` for Linux/mac

### Setting up environment with Anaconda
  1. Create `auto-emergencybreak` environment with all necessary libraries for this project  
     ```conda env create -f environment.yml```
     
  2. Activate `auto-emergencybreak` environment  
     ```source activate auto-emergencybreak```
  
  
### About the files
**test/**  
  &emsp; &emsp; `rc_control_test.py`: RC car control with keyboard  
  &emsp; &emsp;  `stream_server_test.py`: video streaming from Pi to computer  
  &emsp; &emsp;  `ultrasonic_server_test.py`: sensor data streaming from Pi to computer  
  &emsp; &emsp;  **model_train_test/**  
      &emsp; &emsp;  &emsp; &emsp; `data_test.npz`: sample data  
      &emsp; &emsp;  &emsp; &emsp; `train_predict_test.ipynb`: a jupyter notebook that goes through neural network model in OpenCV 
  
**raspberryPi/**    
  &emsp; &emsp;  `stream_client.py`:        stream video frames in jpeg format to the host computer  
  &emsp; &emsp;  `ultrasonic_client.py`:    send distance data measured by sensor to the host computer  
  
**arduino/**  
  &emsp; &emsp;  `rc_keyboard_control.ino`: control RC car controller  
  
**computer/**    
  &emsp; &emsp;  **cascade_xml/**  
      &emsp; &emsp;  &emsp; &emsp;  trained cascade classifiers  
  &emsp; &emsp;  **chess_board/**   
      &emsp; &emsp;  &emsp; &emsp;  images for calibration, captured by pi camera  
      
  &emsp; &emsp;  `picam_calibration.py`:     pi camera calibration  
  &emsp; &emsp;  `collect_training_data.py`: collect images in grayscale, data saved as `*.npz`  
  &emsp; &emsp;  `model.py`:                 neural network model  
  &emsp; &emsp;  `model_training.py`:        model training and validation  
  &emsp; &emsp;  `rc_driver_helper.py`:      helper classes/functions for `rc_driver.py`  
  &emsp; &emsp;  `rc_driver.py`:             receive data from raspberry pi and drive the RC car based on model prediction  
  &emsp; &emsp;  `rc_driver_nn_only.py`:     simplified `rc_driver.py` without object detection  
  


### How to drive
1. **Testing:** Flash `rc_keyboard_control.ino` to Arduino and run `rc_control_test.py` to drive the RC car with keyboard. Run `stream_server_test.py` on computer and then run `stream_client.py` on raspberry pi to test video streaming. Similarly, `ultrasonic_server_test.py` and `ultrasonic_client.py` can be used for sensor data streaming testing.   

2. **Pi Camera calibration (optional):** Take multiple chess board images using pi camera module at various angles and put them into **`chess_board`** folder, run `picam_calibration.py` and returned parameters from the camera matrix will be used in `rc_driver.py`.

3. **Collect training/validation data:** First run `collect_training_data.py` and then run `stream_client.py` on raspberry pi. Press arrow keys to drive the RC car, press `q` to exit. Frames are saved only when there is a key press action. Once exit, data will be saved into newly created **`training_data`** folder.

4. **Neural network training:** Run `model_training.py` to train a neural network model. Please feel free to tune the model architecture/parameters to achieve a better result. After training, model will be saved into newly created **`saved_model`** folder.

5. **Cascade classifiers training (optional):** Trained stop sign and traffic light classifiers are included in the **`cascade_xml`** folder, if you are interested in training your own classifiers, please refer to [OpenCV doc](http://docs.opencv.org/doc/user_guide/ug_traincascade.html) and this great [tutorial](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html).

6. **Self-driving in action**: First run `rc_driver.py` to start the server on the computer (for simplified no object detection version, run `rc_driver_nn_only.py` instead), and then run `stream_client.py` and `ultrasonic_client.py` on raspberry pi. 
