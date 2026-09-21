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
    final cattleProbs = _softmax(cattle);
    final buffaloProbs = _softmax(buffalo);

    // Soft species routing: score every breed as p(species) * softmax(head).
    // A hard binary argmax would discard the other head entirely and
    // propagate ~5% routing errors; the mixture lets a confident breed head
    // win even when the binary head is ambiguous.
    final scored = <_ScoredBreed>[];
    for (var i = 0; i < cattleProbs.length; i++) {
      scored.add(_ScoredBreed(binaryProbs[0] * cattleProbs[i], 0, i,
          i < _cattleLabels.length ? _cattleLabels[i] : 'class_$i'));
    }
    for (var i = 0; i < buffaloProbs.length; i++) {
      scored.add(_ScoredBreed(binaryProbs[1] * buffaloProbs[i], 1, i,
          i < _buffaloLabels.length ? _buffaloLabels[i] : 'class_$i'));
    }
    scored.sort((a, b) => b.score.compareTo(a.score));

    final top3 = scored
        .take(3)
        .map((s) => BreedScore(s.index, s.label, s.score))
        .toList();
    final top = scored.first;

    return PredictionResult(
      species: top.speciesIndex == 0 ? 'cattle' : 'buffalo',
      speciesIndex: top.speciesIndex,
      speciesConfidence: binaryProbs[top.speciesIndex],
      breed: top.label,
      breedIndex: top.index,
      breedConfidence: top.score,
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

  @override
  Future<void> dispose() async {
    _model?.callMethod('dispose', const []);
    _model = null;
  }
}

class _ScoredBreed {
  final double score;
  final int speciesIndex;
  final int index;
  final String label;

  const _ScoredBreed(this.score, this.speciesIndex, this.index, this.label);
}