class BreedScore {
  final int index;
  final String label;
  final double confidence;

  const BreedScore(this.index, this.label, this.confidence);

  Map<String, dynamic> toJson() =>
      {'index': index, 'label': label, 'confidence': confidence};
}

class PredictionResult {
  final String species;
  final int speciesIndex;
  final double speciesConfidence;
  final String breed;
  final int breedIndex;
  final double breedConfidence;
  final List<BreedScore> top3;
  final int latencyMs;

  const PredictionResult({
    required this.species,
    required this.speciesIndex,
    required this.speciesConfidence,
    required this.breed,
    required this.breedIndex,
    required this.breedConfidence,
    required this.top3,
    required this.latencyMs,
  });

  String get text =>
      '$species: $breed (${(breedConfidence * 100).toStringAsFixed(1)}%)';

  Map<String, dynamic> toJson() => {
        'species': species,
        'speciesIndex': speciesIndex,
        'speciesConfidence': speciesConfidence,
        'breed': breed,
        'breedIndex': breedIndex,
        'breedConfidence': breedConfidence,
        'top3': top3.map((e) => e.toJson()).toList(),
        'latencyMs': latencyMs,
      };
}