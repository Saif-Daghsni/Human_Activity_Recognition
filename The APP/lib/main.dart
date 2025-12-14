// lib/main.dart
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:async';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:math' as math;

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Activity Detection',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        textTheme: GoogleFonts.poppinsTextTheme(),
      ),
      home: ActivityDetectionScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class ActivityDetectionScreen extends StatefulWidget {
  @override
  _ActivityDetectionScreenState createState() => _ActivityDetectionScreenState();
}

class _ActivityDetectionScreenState extends State<ActivityDetectionScreen> {
  Interpreter? _interpreter;
  final List<String> _labels = ['sit', 'jum', 'walk', 'run'];
  
  String _currentActivity = 'Press Start';
  double _confidence = 0.0;
  Map<String, double> _probabilities = {};
  int _bufferSize = 0;
  double _actualSamplingRate = 0.0;
  
  StreamSubscription? _accelSubscription;
  StreamSubscription? _gyroSubscription;
  
  List<double> _rawAccel = [0, 0, 0];
  List<double> _rawGyro = [0, 0, 0];
  
  // For downsampling to 100 Hz
  List<double>? _lastAccel;
  List<double>? _lastGyro;
  List<List<double>> _sensorBuffer = [];
  List<DateTime> _timestamps = [];
  
  // Counter-based decimation
  int _sensorEventCounter = 0;
  int _decimationFactor = 5; // 500Hz / 5 = 100Hz
  
  bool _isRunning = false;
  bool _modelLoaded = false;
  
  static const int BUFFER_SIZE = 300;
  
  @override
  void initState() {
    super.initState();
    _loadModel();
  }
  
  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/model.tflite',
        options: InterpreterOptions()..threads = 2,
      );
      print('✓ Model loaded');
      setState(() {
        _modelLoaded = true;
        _currentActivity = 'Ready';
      });
    } catch (e) {
      print('Error loading model: $e');
      setState(() {
        _currentActivity = 'Model Error';
      });
    }
  }
  
  void _startDetection() {
    if (!_modelLoaded || _isRunning) return;
    
    setState(() {
      _isRunning = true;
      _sensorBuffer.clear();
      _timestamps.clear();
      _bufferSize = 0;
      _currentActivity = 'Collecting...';
      _confidence = 0.0;
      _probabilities = {};
    });
    
    // Listen to accelerometer at fastest rate
    _accelSubscription = accelerometerEventStream(
      samplingPeriod: SensorInterval.fastestInterval,
    ).listen((event) {
      _lastAccel = [event.x, event.y, event.z];
      
      // Only count and sample on accelerometer events
      _tryAddSample();
    });
    
    // Listen to gyroscope at fastest rate (just update values, don't sample)
    _gyroSubscription = gyroscopeEventStream(
      samplingPeriod: SensorInterval.fastestInterval,
    ).listen((event) {
      _lastGyro = [event.x, event.y, event.z];
    });
    
    // Update UI periodically (separate from sampling)
    Timer.periodic(Duration(milliseconds: 100), (timer) {
      if (!_isRunning) {
        timer.cancel();
        return;
      }
      if (mounted) {
        setState(() {
          _rawAccel = _lastAccel ?? [0, 0, 0];
          _rawGyro = _lastGyro ?? [0, 0, 0];
        });
      }
    });
  }
  
  void _tryAddSample() {
    if (!_isRunning) return;
    
    // Counter-based decimation: only add every Nth sample
    _sensorEventCounter++;
    
    if (_sensorEventCounter >= _decimationFactor && 
        _lastAccel != null && 
        _lastGyro != null) {
      _sensorEventCounter = 0;
      
      // Add raw sensor values
      _sensorBuffer.add([
        _lastAccel![0], _lastAccel![1], _lastAccel![2],
        _lastGyro![0], _lastGyro![1], _lastGyro![2],
      ]);
      
      _timestamps.add(DateTime.now());
      
      if (_sensorBuffer.length > BUFFER_SIZE) {
        _sensorBuffer.removeAt(0);
        _timestamps.removeAt(0);
      }
      
      setState(() {
        _bufferSize = _sensorBuffer.length;
      });
      
      // Predict when buffer is full
      if (_sensorBuffer.length == BUFFER_SIZE) {
        _predict();
        
        // Clear buffer for non-overlapping windows
        _sensorBuffer.clear();
        _timestamps.clear();
        _bufferSize = 0;
      }
    }
  }
  
  void _predict() {
    if (_interpreter == null || _sensorBuffer.length != BUFFER_SIZE) return;
    
    try {
      // Calculate actual sampling rate
      if (_timestamps.length >= 2) {
        var duration = _timestamps.last.difference(_timestamps.first);
        var samples = _timestamps.length - 1;
        _actualSamplingRate = samples / (duration.inMilliseconds / 1000.0);
      }
      
      // Create input tensor (1, 300, 6)
      var input = [
        List.generate(
          BUFFER_SIZE,
          (i) => List<double>.from(_sensorBuffer[i])
        )
      ];
      
      // Normalize per-sample (exactly like training)
      input = _normalizePerSample(input);
      
      // Output buffer
      var output = [List<double>.filled(4, 0.0)];
      
      // Run inference
      _interpreter!.run(input, output);
      
      int predictedClass = _argMax(output[0]);
      double confidence = output[0][predictedClass];
      
      if (mounted) {
        setState(() {
          _currentActivity = _labels[predictedClass];
          _confidence = confidence;
          _probabilities = {
            for (var i = 0; i < 4; i++) _labels[i]: output[0][i]
          };
        });
      }
    } catch (e) {
      print('Prediction error: $e');
    }
  }
  
  List<List<List<double>>> _normalizePerSample(
    List<List<List<double>>> input
  ) {
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
  
  int _argMax(List<double> list) {
    int maxIdx = 0;
    for (int i = 1; i < list.length; i++) {
      if (list[i] > list[maxIdx]) maxIdx = i;
    }
    return maxIdx;
  }
  
  void _stopDetection() {
    setState(() {
      _isRunning = false;
      _currentActivity = 'Stopped';
      _confidence = 0.0;
      _probabilities = {};
      _bufferSize = 0;
    });
    
    _accelSubscription?.cancel();
    _gyroSubscription?.cancel();
    _sensorBuffer.clear();
    _timestamps.clear();
  }
  
  @override
  void dispose() {
    _accelSubscription?.cancel();
    _gyroSubscription?.cancel();
    _interpreter?.close();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    double progress = _bufferSize / 300.0;
    
    return Scaffold(
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Colors.blue.shade400, Colors.indigo.shade600],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              // Header
              Padding(
                padding: EdgeInsets.all(20),
                child: Column(
                  children: [
                    Text(
                      'Activity Detection',
                      style: TextStyle(
                        fontSize: 28,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                    SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          _modelLoaded ? Icons.check_circle : Icons.error,
                          color: _modelLoaded ? Colors.green : Colors.orange,
                          size: 16,
                        ),
                        SizedBox(width: 8),
                        Text(
                          _modelLoaded ? 'Model Ready' : 'Loading...',
                          style: TextStyle(fontSize: 14, color: Colors.white70),
                        ),
                        SizedBox(width: 20),
                        Text(
                          '${_actualSamplingRate.toStringAsFixed(1)} Hz',
                          style: TextStyle(fontSize: 14, color: Colors.white70),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              
              // Main content
              Expanded(
                child: SingleChildScrollView(
                  child: Container(
                    margin: EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(30),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black26,
                          blurRadius: 20,
                          offset: Offset(0, 10),
                        ),
                      ],
                    ),
                    child: Padding(
                      padding: EdgeInsets.all(24),
                      child: Column(
                        children: [
                          // Activity Icon
                          Container(
                            width: 100,
                            height: 100,
                            decoration: BoxDecoration(
                              color: _isRunning ? Colors.blue.shade50 : Colors.grey.shade200,
                              shape: BoxShape.circle,
                            ),
                            child: Icon(
                              _getActivityIcon(_currentActivity),
                              size: 50,
                              color: _isRunning ? Colors.blue.shade600 : Colors.grey.shade400,
                            ),
                          ),
                          SizedBox(height: 20),
                          
                          // Activity Name
                          Text(
                            _currentActivity.toUpperCase(),
                            style: TextStyle(
                              fontSize: 32,
                              fontWeight: FontWeight.bold,
                              color: Colors.grey.shade800,
                            ),
                          ),
                          SizedBox(height: 8),
                          
                          // Confidence
                          if (_confidence > 0) ...[
                            Text(
                              '${(_confidence * 100).toStringAsFixed(1)}%',
                              style: TextStyle(
                                fontSize: 20,
                                color: Colors.blue.shade600,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ],
                          
                          SizedBox(height: 20),
                          
                          // Progress Bar
                          if (_isRunning) ...[
                            ClipRRect(
                              borderRadius: BorderRadius.circular(10),
                              child: LinearProgressIndicator(
                                value: progress,
                                minHeight: 10,
                                backgroundColor: Colors.grey.shade200,
                                valueColor: AlwaysStoppedAnimation(Colors.blue.shade600),
                              ),
                            ),
                            SizedBox(height: 6),
                            Text(
                              'Buffer: $_bufferSize / 300',
                              style: TextStyle(fontSize: 11, color: Colors.grey.shade600),
                            ),
                          ],
                          
                          SizedBox(height: 24),
                          
                          // Probabilities
                          if (_probabilities.isNotEmpty) ...[
                            Divider(),
                            SizedBox(height: 12),
                            Text(
                              'Probabilities',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.grey.shade700,
                              ),
                            ),
                            SizedBox(height: 12),
                            ..._probabilities.entries.map((entry) {
                              return Padding(
                                padding: EdgeInsets.symmetric(vertical: 6),
                                child: Row(
                                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                  children: [
                                    Row(
                                      children: [
                                        Icon(
                                          _getActivityIcon(entry.key),
                                          size: 18,
                                          color: Colors.grey.shade600,
                                        ),
                                        SizedBox(width: 8),
                                        Text(
                                          entry.key.toUpperCase(),
                                          style: TextStyle(
                                            fontSize: 14,
                                            fontWeight: FontWeight.w500,
                                            color: Colors.grey.shade700,
                                          ),
                                        ),
                                      ],
                                    ),
                                    Text(
                                      '${(entry.value * 100).toStringAsFixed(1)}%',
                                      style: TextStyle(
                                        fontSize: 14,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.blue.shade600,
                                      ),
                                    ),
                                  ],
                                ),
                              );
                            }).toList(),
                          ],
                          
                          // Raw sensor values
                          if (_isRunning) ...[
                            SizedBox(height: 24),
                            Divider(),
                            SizedBox(height: 12),
                            Text(
                              'Sensor Values',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.bold,
                                color: Colors.grey.shade700,
                              ),
                            ),
                            SizedBox(height: 8),
                            _buildSensorRow('Accel', _rawAccel),
                            _buildSensorRow('Gyro', _rawGyro),
                          ],
                        ],
                      ),
                    ),
                  ),
                ),
              ),
              
              // Control Buttons
              Padding(
                padding: EdgeInsets.all(20),
                child: Row(
                  children: [
                    Expanded(
                      child: ElevatedButton(
                        onPressed: (!_isRunning && _modelLoaded) ? _startDetection : null,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green,
                          padding: EdgeInsets.symmetric(vertical: 18),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.play_arrow, color: Colors.white),
                            SizedBox(width: 8),
                            Text(
                              'START',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    SizedBox(width: 16),
                    Expanded(
                      child: ElevatedButton(
                        onPressed: _isRunning ? _stopDetection : null,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.red,
                          padding: EdgeInsets.symmetric(vertical: 18),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.stop, color: Colors.white),
                            SizedBox(width: 8),
                            Text(
                              'STOP',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
  
  Widget _buildSensorRow(String label, List<double> values) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
          ),
          Text(
            'X: ${values[0].toStringAsFixed(2)} Y: ${values[1].toStringAsFixed(2)} Z: ${values[2].toStringAsFixed(2)}',
            style: TextStyle(
              fontSize: 11,
              fontFamily: 'Courier',
              color: Colors.grey.shade700,
            ),
          ),
        ],
      ),
    );
  }
  
  IconData _getActivityIcon(String activity) {
    switch (activity.toLowerCase()) {
      case 'sit':
        return Icons.event_seat;
      case 'walk':
        return Icons.directions_walk;
      case 'run':
        return Icons.directions_run;
      case 'jum':
        return Icons.fitness_center;
      default:
        return Icons.sensors;
    }
  }
}