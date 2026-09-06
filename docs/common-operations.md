# 12. Common Operations

### Setup
```bash
python setup_venv.py          # Create venv + install deps
source .venv/bin/activate
```

### Verify Architecture
```bash
python -m src.verify          # Check backbone loading + forward pass shapes
```

### Prepare Data Splits
```bash
python -m src.data_pipeline   # Scan raw/ → generate splits/*.csv
```

### Train (Full)
```bash
python -m src.train --backbone lite2
```

### Train (Smoke Test)
```bash
python -m src.train --smoke-test --skip-qat
```

### Evaluate
```bash
python -m src.evaluate --backbone lite2
```

### Export
```bash
python -m src.export --mode portable --backbone lite2
python -m src.export --mode onnx --backbone lite2
```

### Create Standalone Training Zip
```bash
python create_training_zip.py # Creates training_package.zip (excludes webapp/ and memory/)
```

### Run Webapp
```bash
python webapp/server.py       # → http://localhost:8000
```

---