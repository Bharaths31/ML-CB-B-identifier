import 'dart:typed_data';

import 'package:image/image.dart' as img;

class ImagePreprocessor {
  ImagePreprocessor({this.size = 260});

  final int size;

  Float32List preprocess(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) {
      throw ArgumentError('Could not decode the image bytes');
    }
    final square = _resizeThenCenterCrop(image, size);

    final out = Float32List(3 * size * size);
    int idx = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
          final p = square.getPixel(x, y);
          final value = c == 0 ? p.r : (c == 1 ? p.g : p.b);
          out[idx++] = value / 255.0;
        }
      }
    }
    return out;
  }

  img.Image _resizeThenCenterCrop(img.Image image, int target) {
    final w = image.width;
    final h = image.height;
    final scale = target / (w < h ? w : h);
    final nw = (w * scale).round();
    final nh = (h * scale).round();
    final resized = img.copyResize(image, width: nw, height: nh);
    final x0 = ((nw - target) / 2).floor();
    final y0 = ((nh - target) / 2).floor();
    return img.copyCrop(resized,
        x: x0, y: y0, width: target, height: target);
  }
}