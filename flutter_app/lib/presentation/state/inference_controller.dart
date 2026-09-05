import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import '../../domain/entities/prediction_result.dart';
import '../../domain/services/i_model_service.dart';

class InferenceController extends ChangeNotifier {
  InferenceController(this._service);

  final IModelService _service;

  PredictionResult? _result;
  bool _busy = false;
  Object? _error;
  bool _initialized = false;

  PredictionResult? get result => _result;
  bool get busy => _busy;
  Object? get error => _error;
  bool get initialized => _initialized;

  Future<void> initialize() async {
    try {
      await _service.initialize();
      _initialized = true;
      _error = null;
    } catch (e) {
      _error = e;
    }
    notifyListeners();
  }

  Future<void> predictImage(Uint8List imageBytes) async {
    _busy = true;
    _error = null;
    _result = null;
    notifyListeners();
    try {
      _result = await _service.predict(imageBytes);
    } catch (e) {
      _error = e;
    } finally {
      _busy = false;
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _service.dispose();
    super.dispose();
  }
}