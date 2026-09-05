import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'data/ml_engines/android_tflite_engine.dart';
import 'data/ml_engines/web_tfjs_engine.dart';
import 'domain/services/i_model_service.dart';
import 'presentation/screens/camera_screen.dart';
import 'presentation/state/inference_controller.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  final IModelService service =
      kIsWeb ? TfJsEngine() : TfliteEngine();
  final controller = InferenceController(service);
  await controller.initialize();

  runApp(
    ChangeNotifierProvider.value(
      value: controller,
      child: const CattleBreedApp(),
    ),
  );
}

class CattleBreedApp extends StatelessWidget {
  const CattleBreedApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Cattle & Buffalo Breed',
      theme: ThemeData(colorSchemeSeed: Colors.green, useMaterial3: true),
      home: const CameraScreen(),
    );
  }
}