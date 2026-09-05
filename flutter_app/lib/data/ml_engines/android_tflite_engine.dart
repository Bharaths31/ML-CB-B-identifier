import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import '../../domain/entities/prediction_result.dart';
import '../../domain/services/i_model_service.dart';
import '../utils/image_preprocessor.dart';

class TfliteEngine implements IModelService {
  TfliteEngine({ImagePreprocessor? preprocessor})
      : _preprocessor = preprocessor ?? ImagePreprocessor();

  static const int _inputSize = 260;

  final ImagePreprocessor _preprocessor;
  tfl.Interpreter? _interpreter;
  List<String> _binaryLabels = const [];
  List<String> _cattleLabels = const [];
  List<String> _buffaloLabels = const [];

  @override
  Future<void> initialize() async {
    final data = await rootBundle.load('assets/models/model.tflite');
    _interpreter = tfl.Interpreter.fromBuffer(data.buffer);
    _binaryLabels = await _loadLabels('assets/models/labels_binary.txt');
    _cattleLabels = await _loadLabels('assets/models/labels_cattle.txt');
    _buffaloLabels = await _loadLabels('assets/models/labels_buffalo.txt');
  }

  Future<List<String>> _loadLabels(String path) async {
    final raw = await rootBundle.loadString(path);
    return raw
        .trim()
        .split('\n')
        .map((s) => s.trim())
        .where((s) => s.isNotEmpty)
        .toList();
  }

  @override
  Future<PredictionResult> predict(Uint8List imageBytes) async {
    final interpreter = _interpreter;
    if (interpreter == null) {
      throw StateError('Model not initialized. Call initialize() first.');
    }

    final input = _preprocessor.preprocess(imageBytes);
    final inputTensor =
        tfl.Tensor.fromList(input, [1, 3, _inputSize, _inputSize]);

    final outBinary = List<double>.filled(2, 0.0);
    final outCattle = List<double>.filled(_cattleLabels.length, 0.0);
    final outBuffalo = List<double>.filled(_buffaloLabels.length, 0.0);

    final sw = Stopwatch()..start();
    interpreter.runForMultipleInputsOutputs(
      [inputTensor],
      {0: outBinary, 1: outCattle, 2: outBuffalo},
    );
    sw.stop();

    return _postprocess(outBinary, outCattle, outBuffalo, sw.elapsedMilliseconds);
  }

  PredictionResult _postprocess(
      List<double> binary,
      List<double> cattle,
      List<double> buffalo,
      int latencyMs) {
    final binaryProbs = _softmax(binary);
    final speciesIndex = _argmax(binary);
    final isCattle = speciesIndex == 0;
    final logits = isCattle ? cattle : buffalo;
    final labels = isCattle ? _cattleLabels : _buffaloLabels;
    final probs = _softmax(logits);

    final order = List<int>.generate(probs.length, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));
    final top3 = order
        .take(3)
        .map((i) => BreedScore(
              i,
              i < labels.length ? labels[i] : 'class_$i',
              probs[i],
            ))
        .toList();
    final top = top3.first;

    return PredictionResult(
      species: isCattle ? 'cattle' : 'buffalo',
      speciesIndex: speciesIndex,
      speciesConfidence: binaryProbs[speciesIndex],
      breed: top.label,
      breedIndex: top.index,
      breedConfidence: top.confidence,
      top3: top3,
      latencyMs: latencyMs,
    );
  }

  List<double> _softmax(List<double> logits) {
    final max = logits.reduce((a, b) => a > b ? a : b);
    final exps = logits.map((x) => _exp(x - max)).toList();
    final sum = exps.fold(0.0, (a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  double _exp(double x) {
    const e = 2.718281828459045;
    var result = 1.0;
    var term = 1.0;
    for (var i = 1; i < 32; i++) {
      term *= x / i;
      result += term;
    }
    return result;
  }

  int _argmax(List<double> values) {
    var best = 0;
    for (var i = 1; i < values.length; i++) {
      if (values[i] > values[best]) best = i;
    }
    return best;
  }

  @override
  Future<void> dispose() async {
    await _interpreter?.close();
    _interpreter = null;
  }
}