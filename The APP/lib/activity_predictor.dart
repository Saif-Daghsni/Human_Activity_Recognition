// lib/activity_predictor.dart
import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math' as math;

class ActivityPredictor {
  Interpreter? _interpreter;
  final List<String> _labels = ['sit', 'jum', 'walk', 'run'];

  static const int BUFFER_SIZE = 300;
  static const int FEATURES = 6;
  List<List<double>> _sensorBuffer = [];

  // Track timestamps for sampling rate
  List<DateTime> _timestamps = [];

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/model.tflite');
      print('✓ Model loaded');
      print('Input: ${_interpreter!.getInputTensor(0).shape}');
      print('Output: ${_interpreter!.getOutputTensor(0).shape}');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  void addSensorReading(
    double accelX,
    double accelY,
    double accelZ,
    double gyroX,
    double gyroY,
    double gyroZ,
  ) {
    _sensorBuffer.add([accelX, accelY, accelZ, gyroX, gyroY, gyroZ]);

    _timestamps.add(DateTime.now());

    if (_sensorBuffer.length > BUFFER_SIZE) {
      _sensorBuffer.removeAt(0);
      _timestamps.removeAt(0);
    }
  }

  bool isBufferReady() => _sensorBuffer.length == BUFFER_SIZE;

  double getActualSamplingRate() {
    if (_timestamps.length < 2) return 0.0;

    var duration = _timestamps.last.difference(_timestamps.first);
    var samples = _timestamps.length - 1;

    return samples / (duration.inMilliseconds / 1000.0);
  }

  Map<String, dynamic>? predict() {
    if (_interpreter == null || !isBufferReady()) return null;

    try {
      double actualRate = getActualSamplingRate();

      // Create input tensor (1, 300, 6)
      var input = [
        List.generate(BUFFER_SIZE, (i) => List<double>.from(_sensorBuffer[i])),
      ];

      // Normalize per-sample (exactly like training)
      input = _normalizePerSample(input);

      // Output buffer
      var output = [List<double>.filled(4, 0.0)];

      // Run inference
      _interpreter!.run(input, output);

      int predictedClass = _argMax(output[0]);
      double confidence = output[0][predictedClass];

      return {
        'activity': _labels[predictedClass],
        'confidence': confidence,
        'probabilities': {for (var i = 0; i < 4; i++) _labels[i]: output[0][i]},
        'samplingRate': actualRate,
      };
    } catch (e) {
      print('Prediction error: $e');
      return null;
    }
  }

  List<List<List<double>>> _normalizePerSample(List<List<List<double>>> input) {
    for (int sampleIdx = 0; sampleIdx < input.length; sampleIdx++) {
      // Find max absolute value across ALL timesteps and ALL features
      double maxAbs = 0.0;

      for (var timestep in input[sampleIdx]) {
        for (var feature in timestep) {
          double absVal = feature.abs();
          if (absVal > maxAbs) maxAbs = absVal;
        }
      }

      // Normalize by max absolute value
      if (maxAbs > 0) {
        for (int t = 0; t < input[sampleIdx].length; t++) {
          for (int f = 0; f < input[sampleIdx][t].length; f++) {
            input[sampleIdx][t][f] /= maxAbs;
          }
        }
      }
    }

    return input;
  }

  void _printInputStats(List<List<double>> sample) {
    // Print statistics for debugging
    for (int featureIdx = 0; featureIdx < FEATURES; featureIdx++) {
      List<double> featureValues = [];
      for (var timestep in sample) {
        featureValues.add(timestep[featureIdx]);
      }

      double min = featureValues.reduce(math.min);
      double max = featureValues.reduce(math.max);
      double mean =
          featureValues.reduce((a, b) => a + b) / featureValues.length;

      String featureName = [
        'AccX',
        'AccY',
        'AccZ',
        'GyroX',
        'GyroY',
        'GyroZ',
      ][featureIdx];
      print(
        '$featureName: min=${min.toStringAsFixed(3)}, max=${max.toStringAsFixed(3)}, mean=${mean.toStringAsFixed(3)}',
      );
    }
  }

  int _argMax(List<double> list) {
    int maxIdx = 0;
    for (int i = 1; i < list.length; i++) {
      if (list[i] > list[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }

  int get bufferSize => _sensorBuffer.length;

  void clearBuffer() {
    _sensorBuffer.clear();
    _timestamps.clear();
  }

  void close() => _interpreter?.close();
}
