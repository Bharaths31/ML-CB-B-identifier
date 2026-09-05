import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;

import '../../domain/entities/prediction_result.dart';
import '../../domain/services/i_model_service.dart';
import '../utils/image_preprocessor.dart';

@JS('tf')
external JSObject get tf;

class TfJsEngine implements IModelService {
  TfJsEngine({ImagePreprocessor? preprocessor})
      : _preprocessor = preprocessor ?? ImagePreprocessor();

  static const int _inputSize = 260;

  final ImagePreprocessor _preprocessor;
  JSObject? _model;
  List<String> _binaryLabels = const [];
  List<String> _cattleLabels = const [];
  List<String> _buffaloLabels = const [];

  @override
  Future<void> initialize() async {
    _binaryLabels = await _loadLabels('assets/models/labels_binary.txt');
    _cattleLabels = await _loadLabels('assets/models/labels_cattle.txt');
    _buffaloLabels = await _loadLabels('assets/models/labels_buffalo.txt');

    final loader = tf.callMethod(
      'loadGraphModel',
      ['assets/models/model_web.json'.toJS],
    );
    if (loader is! JSPromise) {
      throw StateError(
        'TensorFlow.js is not available. Add the tfjs script tag to '
        'web/index.html before the app starts.',
      );
    }
    _model = (await (loader as JSPromise<JSAny?>).toDart) as JSObject;
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
    final model = _model;
    if (model == null) {
      throw StateError('Model not initialized. Call initialize() first.');
    }

    final input = _preprocessor.preprocess(imageBytes);
    final data = input.toList().toJS;
    final shape = [1, 3, _inputSize, _inputSize].toJS;
    final tensor = tf.callMethod('tensor4d', [data, shape]);

    final sw = Stopwatch()..start();
    final results = await _execute(model, tensor);
    sw.stop();

    final binary = results[0];
    final cattle = results[1];
    final buffalo = results[2];

    tensor.callMethod('dispose', const []);

    return _postprocess(binary, cattle, buffalo, sw.elapsedMilliseconds);
  }

  Future<List<List<double>>> _execute(JSObject model, JSAny tensor) async {
    JSAny? out;
    try {
      final outputNames = ['binary', 'cattle', 'buffalo'].toJS;
      out = model.callMethod('execute', [tensor, outputNames]);
    } catch (_) {
      out = model.callMethod('predict', [tensor]);
    }
    final arr = (out as JSArray<JSAny?>).toDart;
    final results = <List<double>>[];
    for (final t in arr) {
      final raw = (t as JSObject).callMethod('dataSync', const []);
      results.add(_toList(raw));
      (t as JSObject).callMethod('dispose', const []);
    }
    return results;
  }

  List<double> _toList(JSAny? value) {
    if (value == null) return const [];
    if (value is JSFloat32Array) return value.toDart;
    if (value is JSFloat64Array) return value.toDart.toList();
    if (value is JSArray) {
      return value.toDart
          .map((e) => (e as JSNumber).toDartDouble)
          .toList();
    }
    return const [];
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
    _model?.callMethod('dispose', const []);
    _model = null;
  }
}