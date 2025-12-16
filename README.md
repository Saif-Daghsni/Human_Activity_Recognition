# 🏃‍♂️Human Activity Recognition with Flutter & TensorFlow Lite

A complete end-to-end machine learning mobile application that recognizes human activities in real-time using smartphone sensors.

## 🎯 Features

- **Real-time Activity Detection**: Classifies 4 activities (sitting, walking, running, jumping) 
- **Custom Dataset**: Built from scratch using accelerometer and gyroscope data
- **On-Device ML**: TensorFlow Lite model runs entirely on-device with no internet required
- **100 Hz Sampling**: Precise sensor data collection at 100 Hz with non-overlapping windows
- **Beautiful UI**: Modern Flutter interface with live predictions and confidence scores

## 📊 Technical Highlights

- **Model Architecture**: 1D Convolutional Neural Network (CNN)
- **Input**: 300 samples × 6 features (3-axis accelerometer + 3-axis gyroscope)
- **Inference Time**: Real-time predictions every 3 seconds
- **Data Augmentation**: 20× augmentation with noise injection, scaling, time warping, and rotation

## 🚀 Tech Stack

- **Mobile App**: Flutter, Dart, TFLite Flutter plugin
- **ML Framework**: TensorFlow, Keras
- **Data Processing**: Python, NumPy, Pandas, Scikit-learn
- **Sensors**: Accelerometer, Gyroscope (sensors_plus package)

## 📁 Project Structure
```
├── flutter_app/           # Flutter mobile application
│   ├── lib/
│   │   └── main.dart     # Main app with activity detection
│   └── assets/models/    # TFLite model
│
├── ml_training/          # Machine learning pipeline
│   ├── data_collection/  # Data collection Flutter app
│   ├── prepare_data.py   # Data preprocessing with augmentation
│   ├── train_model.py    # Model training script
│   └── convert_to_tflite.py  # TFLite conversion
│
└── README.md
```

## 🎓 What I Learned

- End-to-end ML pipeline from data collection to deployment
- Mobile sensor data processing and signal normalization
- Data augmentation techniques for small datasets
- Converting Keras models to TensorFlow Lite
- Real-time inference on mobile devices
- Flutter sensor integration and state management



---

## 👨‍💻 Author

Saif Eddine Daghsni
 LinkedIn: https://www.linkedin.com/in/saif-eddine-daghsni/

---

## 📱 Demo

(https://drive.google.com/file/d/1Uf-iulvo0rSxP4Vsbkg5Kjf3COpkCFy8/view?usp=sharing)



