import 'dart:typed_data';

import '../entities/prediction_result.dart';

abstract class IModelService {
  Future<void> initialize();

  Future<PredictionResult> predict(Uint8List imageBytes);

  Future<void> dispose();
}