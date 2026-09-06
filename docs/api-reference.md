# 11. API Reference

### Training Arguments (`python -m src.train`)

```
--backbone {lite2,lite4}     Backbone architecture (default: lite2)
--weights PATH               Pretrained weights path
--attention {cbam,se}        Attention module (default: cbam)
--data PATH                  Raw data root
--split-dir PATH             Split CSV output directory
--batch-size N               Batch size (default: 32)
--num-workers N              DataLoader workers (default: 4)
--device DEVICE              Force device (auto-detects cuda/cpu)
--no-mix                     Disable CutMix/MixUp
--phase{1,2,3}-epochs N      Override epoch count
--skip-qat                   Skip phase 3 (QAT)
--smoke-test                 Use mini-dataset (5 imgs/breed, 1 epoch)
--seed N                     Random seed (default: 42)
--export-dir PATH            Portable export destination
--no-export                  Skip auto-export after training
```

### Webapp API Payload Formats

**POST /api/train**:
```json
{
  "backbone": "lite2",
  "smoke_test": true,
  "skip_qat": false,
  "phase1_epochs": 5,
  "phase2_epochs": 30,
  "phase3_epochs": 10,
  "num_workers": 4
}
```

**POST /api/export**:
```json
{"backbone": "lite2", "mode": "portable"}
```

---