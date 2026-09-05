import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../../domain/entities/prediction_result.dart';
import '../state/inference_controller.dart';

class CameraScreen extends StatelessWidget {
  const CameraScreen({super.key});

  Future<void> _pickAndPredict(
      BuildContext context, ImageSource source) async {
    final controller = context.read<InferenceController>();
    final picker = ImagePicker();
    final XFile? file = await picker.pickImage(
      source: source,
      maxWidth: 1600,
      maxHeight: 1600,
      imageQuality: 92,
    );
    if (file == null) return;
    await controller.predictImage(await file.readAsBytes());
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Cattle & Buffalo Breed')),
      body: Consumer<InferenceController>(
        builder: (context, controller, _) {
          return SafeArea(
            child: Column(
              children: [
                Expanded(child: _buildBody(context, controller)),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      FilledButton.icon(
                        onPressed: controller.busy
                            ? null
                            : () =>
                                _pickAndPredict(context, ImageSource.camera),
                        icon: const Icon(Icons.photo_camera),
                        label: const Text('Capture'),
                      ),
                      FilledButton.tonalIcon(
                        onPressed: controller.busy
                            ? null
                            : () =>
                                _pickAndPredict(context, ImageSource.gallery),
                        icon: const Icon(Icons.photo_library),
                        label: const Text('Gallery'),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  Widget _buildBody(BuildContext context, InferenceController controller) {
    if (controller.error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            'Error: ${controller.error}',
            style: Theme.of(context).textTheme.bodyMedium,
            textAlign: TextAlign.center,
          ),
        ),
      );
    }
    if (controller.busy) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Running inference...'),
          ],
        ),
      );
    }
    final result = controller.result;
    if (result == null) {
      return const Center(
        child: Text('Capture or select a photo to identify the breed'),
      );
    }
    return _ResultCard(result: result);
  }
}

class _ResultCard extends StatelessWidget {
  const _ResultCard({required this.result});

  final PredictionResult result;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      margin: const EdgeInsets.all(16),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Chip(label: Text(result.species)),
                const Spacer(),
                Text('${result.latencyMs} ms',
                    style: theme.textTheme.bodySmall),
              ],
            ),
            const SizedBox(height: 12),
            Text('Breed: ${result.breed}', style: theme.textTheme.titleLarge),
            const SizedBox(height: 4),
            Text(
              'Confidence: ${(result.breedConfidence * 100).toStringAsFixed(1)}%',
              style: theme.textTheme.bodyMedium,
            ),
            Text(
              'Species confidence: '
              '${(result.speciesConfidence * 100).toStringAsFixed(1)}%',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            Text('Top 3', style: theme.textTheme.labelLarge),
            ...result.top3.map((s) => Padding(
                  padding: const EdgeInsets.only(top: 2),
                  child: Text(
                    '  ${s.confidence * 100 >= 1 ? '${(s.confidence * 100).toStringAsFixed(1)}%' : '<1%'}  ${s.label}',
                    style: theme.textTheme.bodySmall,
                  ),
                )),
          ],
        ),
      ),
    );
  }
}