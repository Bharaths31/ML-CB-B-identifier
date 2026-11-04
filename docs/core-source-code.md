# Core Source Code

BreedID Livestock Vision

## 1. Model Architecture

### 1.1 Cosine classification head

```python
class CosineHead(nn.Module):

    def __init__(self, in_features, out_features, scale=COSINE_SCALE):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = float(scale)

    def forward(self, x):
        cos = F.linear(F.normalize(x, dim=1), F.normalize(self.weight, dim=1))
        return self.scale * cos
```

### 1.2 Breed head builder

```python
def _make_breed_head(feature_dim, num_classes, dropout, cosine=False,
                     scale=COSINE_SCALE, breed_dim=BREED_DIM):
    layers = [
        nn.Linear(feature_dim, breed_dim),
        nn.BatchNorm1d(breed_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(breed_dim, breed_dim // 2),
        nn.BatchNorm1d(breed_dim // 2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
    ]
    layers.append(CosineHead(breed_dim // 2, num_classes, scale) if cosine
                  else nn.Linear(breed_dim // 2, num_classes))
    return nn.Sequential(*layers)
```

### 1.3 Multi-head breed classifier

```python
class BreedClassifier(nn.Module):
    def __init__(self, backbone="lite2", num_cattle=NUM_CATTLE_BREEDS,
                 num_buffalo=NUM_BUFFALO_BREEDS, cbam_stage=CBAM_AFTER_STAGE,
                 attention="cbam", activation="relu6", pretrained_path=None,
                 dropout=DROPOUT, cosine_head=False, cosine_scale=COSINE_SCALE,
                 binary_dim=BINARY_DIM, breed_dim=BREED_DIM):
        super().__init__()
        self.cosine_head = bool(cosine_head)
        self.cosine_scale = float(cosine_scale)
        self.backbone = EfficientNetLite(arch=backbone, activation=activation)
        self.cbam_stage = cbam_stage
        stage_channels = self.backbone.stage_channels[cbam_stage]
        self.attention = build_attention(attention, stage_channels)
        feature_dim = self.backbone.head_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.binary_head = nn.Sequential(
            nn.Linear(feature_dim, binary_dim),
            nn.BatchNorm1d(binary_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(binary_dim, binary_dim // 2),
            nn.BatchNorm1d(binary_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(binary_dim // 2, 2),
        )
        self.cattle_head = _make_breed_head(feature_dim, num_cattle, dropout,
                                            self.cosine_head, self.cosine_scale,
                                            breed_dim=breed_dim)
        self.buffalo_head = _make_breed_head(feature_dim, num_buffalo, dropout,
                                             self.cosine_head, self.cosine_scale,
                                             breed_dim=breed_dim)

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, PROJECTION_DIM),
        )

        if pretrained_path:
            n = load_backbone_weights(self.backbone, pretrained_path)
            print(f"[model] loaded {n} tensors from {pretrained_path}")

    def forward_features(self, x):
        feats = self.backbone.forward_until(x, self.cbam_stage)
        feats = self.attention(feats)
        feats = self.backbone.forward_from(feats, self.cbam_stage + 1)
        return self.avg_pool(feats).flatten(1)

    def forward(self, x):
        pooled = self.forward_features(x)
        return {
            "binary": self.binary_head(pooled),
            "cattle": self.cattle_head(pooled),
            "buffalo": self.buffalo_head(pooled),
            "features": pooled,
            "embedding": self.projection_head(pooled),
        }

    @torch.no_grad()
    def predict(self, x):
        out = self.forward(x)
        p_species = F.softmax(out["binary"], dim=1)
        p_cattle = F.softmax(out["cattle"], dim=1)
        p_buffalo = F.softmax(out["buffalo"], dim=1)
        return torch.cat([p_species[:, 0:1] * p_cattle,
                          p_species[:, 1:2] * p_buffalo], dim=1)

    @torch.no_grad()
    def predict_safe(self, x, ood_detector=None):
        out = self.forward(x)

        if ood_detector is not None:
            is_ood, score, details = ood_detector.is_ood(
                out["binary"], out["cattle"], out["buffalo"]
            )
            if is_ood:

                n_total = out["cattle"].shape[1] + out["buffalo"].shape[1]
                return torch.zeros(x.shape[0], n_total, device=x.device), {
                    "ood": True, "ood_score": score, **details
                }

        p_species = F.softmax(out["binary"], dim=1)
        p_cattle = F.softmax(out["cattle"], dim=1)
        p_buffalo = F.softmax(out["buffalo"], dim=1)
        combined = torch.cat([p_species[:, 0:1] * p_cattle,
                              p_species[:, 1:2] * p_buffalo], dim=1)
        return combined, {"ood": False}

    @torch.no_grad()
    def predict_guarded(self, x, pil_img=None, image_bytes=None,
                        ood_detector=None, guardrail=None):
        out = self.forward(x)
        if guardrail is None:
            from .input_guardrails import get_guardrail_engine
            guardrail = get_guardrail_engine()
        verdict = guardrail.validate_image(
            image_bytes, pil_img, model_outputs=out, ood_detector=ood_detector)
        return out, verdict

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.attention.parameters():
            p.requires_grad = False

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def backbone_eval(self):
        self.backbone.eval()
        self.attention.eval()

    def backbone_train(self):
        self.backbone.train()
        self.attention.train()
```

### 1.4 Channel attention

```python
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        mid = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        self.identity_init = identity_init
        if identity_init:
            _zero_init_last(self.mlp)

    def forward(self, x):
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        att = self.mlp(avg) + self.mlp(mx)
        if self.identity_init:

            gate = 2.0 * self.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.sigmoid(att)
```

### 1.5 Spatial attention

```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.identity_init = identity_init
        if identity_init:
            nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        fused = torch.cat([avg, mx], dim=1)
        att = self.conv(fused)
        if self.identity_init:
            gate = 2.0 * self.sigmoid(att) - 1.0
            return x * (1.0 + gate)
        return x * self.sigmoid(att)
```

### 1.6 CBAM attention block

```python
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7,
                 identity_init=CBAM_IDENTITY_INIT):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction,
                                                  identity_init=identity_init)
        self.spatial_attention = SpatialAttention(kernel_size,
                                                  identity_init=identity_init)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
```

### 1.7 Attention factory

```python
def build_attention(kind, channels):
    kind = (kind or "cbam").lower()
    if kind == "cbam":
        return CBAM(channels)
    if kind == "se":
        return SEBlock(channels)
    raise ValueError(f"Unknown attention kind {kind}")
```

## 2. Image Preprocessing

### 2.1 Pad-to-square transform

```python
class PadToSquare:

    def __init__(self, size=IMAGE_SIZE, fill=None):
        self.size = int(size)
        self.fill = fill or tuple(int(round(m * 255)) for m in IMAGENET_MEAN)

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        resized = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas

    def __repr__(self):
        return f"PadToSquare({self.size}, fill={self.fill})"
```

### 2.2 Inference transform

```python
def inference_transform(pad=None, size=IMAGE_SIZE):
    if pad is None:
        from .config import INFERENCE_PAD_TO_SQUARE
        pad = INFERENCE_PAD_TO_SQUARE

    if pad:
        blocks = [PadToSquare(size)]
    else:
        blocks = [transforms.Resize(size), transforms.CenterCrop(size)]
    blocks += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(blocks)
```

## 3. Multi-Task Training Loss

### 3.1 Soft cross-entropy

```python
def soft_ce(pred, target, label_smoothing=0.0, logit_prior=None, tau=0.0):
    if logit_prior is not None and tau:
        pred = pred + tau * logit_prior.to(pred.device).unsqueeze(0)
    if label_smoothing > 0.0:
        n_classes = pred.size(1)
        target = (1.0 - label_smoothing) * target + label_smoothing / n_classes
    return -(target * F.log_softmax(pred, dim=1)).sum(dim=1)
```

### 3.2 Species-masked breed loss

```python
def masked_loss(out, labels, w_binary, w_cattle, w_buffalo,
                label_smoothing=0.0, logit_priors=None, adjust_tau=0.0,
                contrastive_weight=0.0, contrastive_temp=CONTRASTIVE_TEMPERATURE,
                mixed=False):
    cattle_prior = buffalo_prior = None
    if logit_priors is not None:
        cattle_prior = logit_priors.get("cattle")
        buffalo_prior = logit_priors.get("buffalo")

    ce_binary = soft_ce(out["binary"], labels["binary"], label_smoothing)
    if BALANCE_BINARY_HEAD:
        p_buffalo = labels["binary"][:, 1].clamp(0.0, 1.0)
        mean_buf = p_buffalo.mean().clamp(min=1e-4)
        mean_cat = (1.0 - p_buffalo).mean().clamp(min=1e-4)

        w_cat = 0.5 / mean_cat
        w_buf = 0.5 / mean_buf
        sample_w = w_cat * (1.0 - p_buffalo) + w_buf * p_buffalo
        ce_binary = (ce_binary * sample_w).sum() / sample_w.sum()
    else:
        ce_binary = ce_binary.mean()
    cattle_logits = out["cattle"]
    buffalo_logits = out["buffalo"]
    if _COSINE_SCALE is not None and _COSINE_MARGIN > 0:
        cattle_logits = _apply_cosine_margin(cattle_logits, labels["cattle"],
                                             _COSINE_MARGIN, _COSINE_SCALE)
        buffalo_logits = _apply_cosine_margin(buffalo_logits, labels["buffalo"],
                                              _COSINE_MARGIN, _COSINE_SCALE)
    ce_cattle = (soft_ce(cattle_logits, labels["cattle"], label_smoothing,
                         cattle_prior, adjust_tau) * labels["cattle_mask"])
    ce_buffalo = (soft_ce(buffalo_logits, labels["buffalo"], label_smoothing,
                          buffalo_prior, adjust_tau) * labels["buffalo_mask"])
    denom_c = labels["cattle_mask"].sum().clamp(min=1.0)
    denom_b = labels["buffalo_mask"].sum().clamp(min=1.0)
    ce_cattle = ce_cattle.sum() / denom_c
    ce_buffalo = ce_buffalo.sum() / denom_b
    total = (w_binary * ce_binary + w_cattle * ce_cattle + w_buffalo * ce_buffalo)
    total = total + _contrastive_term(out, labels, contrastive_weight,
                                      contrastive_temp, mixed)
    return total, ce_binary, ce_cattle, ce_buffalo
```

## 4. Evaluation

### 4.1 Per-class confusion matrices

```python
@torch.no_grad()
def full_evaluation(model, loader, device, num_cattle=NUM_CATTLE_BREEDS,
                    num_buffalo=NUM_BUFFALO_BREEDS):
    model.eval()
    cattle_cm = torch.zeros(num_cattle, num_cattle, dtype=torch.long)
    buffalo_cm = torch.zeros(num_buffalo, num_buffalo, dtype=torch.long)
    cattle_total = torch.zeros(num_cattle, dtype=torch.long)
    buffalo_total = torch.zeros(num_buffalo, dtype=torch.long)
    for images, labels in tqdm(loader, desc="eval detail", leave=False):
        images = images.to(device, non_blocking=True)
        is_cuda = (getattr(device, "type", str(device)) == "cuda" or str(device).startswith("cuda"))
        with torch.amp.autocast(device_type="cuda", enabled=is_cuda):
            out = model(images)
        bin_true = labels["binary"].argmax(1)
        cattle_true = labels["cattle"].argmax(1)
        buffalo_true = labels["buffalo"].argmax(1)
        cmask = labels["cattle_mask"] > 0.5
        bmask = labels["buffalo_mask"] > 0.5
        if cmask.any():
            pred = out["cattle"][cmask].argmax(1)
            true = cattle_true[cmask]
            for t, p in zip(true.tolist(), pred.tolist()):
                cattle_cm[t, p] += 1
                cattle_total[t] += 1
        if bmask.any():
            pred = out["buffalo"][bmask].argmax(1)
            true = buffalo_true[bmask]
            for t, p in zip(true.tolist(), pred.tolist()):
                buffalo_cm[t, p] += 1
                buffalo_total[t] += 1

    def per_class(cm, total):
        n_classes = cm.size(0)
        precision = []
        recall = []
        for c in range(n_classes):
            tp = cm[c, c].item()
            col = cm[:, c].sum().item()
            row = cm[c, :].sum().item()
            precision.append(tp / col if col else 0.0)
            recall.append(tp / row if row else 0.0)
        return {
            "precision": precision,
            "recall": recall,
            "counts": total.tolist(),
        }

    return {
        "cattle_cm": cattle_cm.numpy(),
        "buffalo_cm": buffalo_cm.numpy(),
        "cattle_total": cattle_total.numpy(),
        "buffalo_total": buffalo_total.numpy(),
        "cattle_per_class": per_class(cattle_cm, cattle_total),
        "buffalo_per_class": per_class(buffalo_cm, buffalo_total),
    }
```

## 5. Out-of-Distribution Detection

### 5.1 Energy, MSP and decision rule

```python
class OODDetector:
    def __init__(self, temperature=1.0, energy_threshold=0.0, msp_threshold=0.0, binary_confidence_floor=0.0):
        self.T = float(temperature)
        self.energy_threshold = float(energy_threshold)
        self.msp_threshold = float(msp_threshold)
        self.binary_confidence_floor = float(binary_confidence_floor)

    def compute_energy(self, *logits_list):
        is_numpy = not isinstance(logits_list[0], torch.Tensor)
        if is_numpy:
            import numpy as np
            combined_logits = np.concatenate(logits_list, axis=-1)
            max_val = np.max(combined_logits / self.T, axis=-1, keepdims=True)
            sum_exp = np.sum(np.exp(combined_logits / self.T - max_val), axis=-1)
            energy = -self.T * (np.squeeze(max_val, axis=-1) + np.log(sum_exp))
            return energy
        else:
            combined_logits = torch.cat(logits_list, dim=-1)
            energy = -self.T * torch.logsumexp(combined_logits / self.T, dim=-1)
            return energy

    def compute_msp(self, binary_logits):
        is_numpy = not isinstance(binary_logits, torch.Tensor)
        if is_numpy:
            import numpy as np
            exp_logits = np.exp(binary_logits - np.max(binary_logits, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            return np.max(probs, axis=-1)
        else:
            probs = F.softmax(binary_logits, dim=-1)
            return torch.max(probs, dim=-1)[0]

    def is_ood(self, binary_logits, cattle_logits, buffalo_logits):
        energy = self.compute_energy(binary_logits, cattle_logits, buffalo_logits)
        msp = self.compute_msp(binary_logits)

        is_numpy = not isinstance(energy, torch.Tensor)
        if is_numpy:
            import numpy as np
            energy_val = float(np.atleast_1d(energy)[0])
            msp_val = float(np.atleast_1d(msp)[0])
        else:
            energy_val = energy.item() if energy.numel() == 1 else energy[0].item()
            msp_val = msp.item() if msp.numel() == 1 else msp[0].item()

        ood_by_energy = energy_val > self.energy_threshold
        ood_by_msp = msp_val < self.msp_threshold
        ood_by_binary_floor = msp_val < self.binary_confidence_floor
        is_ood_flag = ood_by_energy or ood_by_msp or ood_by_binary_floor

        return is_ood_flag, energy_val, {
            "energy": energy_val,
            "energy_threshold": self.energy_threshold,
            "msp": msp_val,
            "msp_threshold": self.msp_threshold,
            "binary_confidence_floor": self.binary_confidence_floor,
            "ood_by_energy": ood_by_energy,
            "ood_by_msp": ood_by_msp,
            "ood_by_binary_floor": ood_by_binary_floor,
            "source_file": self.source_file,
            "profile_name": self.profile_name,
        }
```

## 6. Input Guardrails

### 6.1 Four-stage validation pipeline

```python
class InputGuardrailEngine:
    def validate_image(self, image_bytes: Optional[bytes], pil_img: Optional[Image.Image],
                       model_outputs: Optional[Dict[str, Any]] = None,
                       ood_detector: Any = None) -> Dict[str, Any]:
        started = time.perf_counter()
        if not self.enabled:
            return self._accepted({}, started, skipped=True)

        image = pil_img
        if image is None and image_bytes:
            try:
                import io as _io
                image = Image.open(_io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                image = None

        stages: Dict[str, Any] = {}
        if image is not None:
            visual = self.validate_visual(image)
            stages["stage1_2"] = visual
            if visual["rejected"]:
                return self._rejected(visual, started, stages)
        else:
            stages["stage1_2"] = {"rejected": False, "details": {"skipped": "no image"}}

        if model_outputs is not None:
            model_verdict = self.validate_model_outputs(model_outputs, ood_detector)
            stages["stage3_4"] = model_verdict
            if model_verdict["rejected"]:
                return self._rejected(model_verdict, started, stages)
            details = model_verdict.get("details", {})
        else:
            details = {}

        return self._accepted(details, started, stages=stages)

    def validate_visual(self, pil_img: Image.Image) -> Dict[str, Any]:
        if not self.enabled:
            return {"rejected": False, "details": {"skipped": "disabled"}}

        started = time.perf_counter()
        if GUARDRAIL_PHOTOREALISM_ENABLED:
            realism = self._check_photorealism(pil_img)
            if realism["rejected"]:
                realism["stage"] = STAGE_NAMES[1]
                realism["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return self._rejected(realism, started, {})

        human = self._check_human_or_selfie(pil_img)
        if human["rejected"]:
            human["stage"] = STAGE_NAMES[2]
            human["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return self._rejected(human, started, {})
        return self._accepted(human.get("details", {}), started)

    def validate_model_outputs(self, model_outputs: Dict[str, Any],
                               ood_detector: Any = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"rejected": False, "details": {"skipped": "disabled"}}
        started = time.perf_counter()
        feature = self._check_feature_and_entropy(model_outputs)
        if feature["rejected"]:
            feature["stage"] = STAGE_NAMES[3]
            feature["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return self._rejected(feature, started, {})

        if ood_detector is not None:
            logit = self._check_logit_ood(model_outputs, ood_detector)
            if logit["rejected"]:
                logit["stage"] = STAGE_NAMES[4]
                logit["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return self._rejected(logit, started, {})

        return self._accepted(feature.get("details", {}), started)
```

## 7. Flask Prediction API

### 7.1 Application factory

```python
def create_app():

    logger = init_logger()
    logger.info("Starting Flask API Server...")
    manager = init_manager()

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'secret!'

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    app.extensions['socketio'] = socketio
    get_shared_config(socketio)

    get_escalation_db()

    app.register_blueprint(models_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(dev_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(config_bp)
    app.register_blueprint(shared_config_bp)
    app.register_blueprint(ood_bp)
    app.register_blueprint(escalations_bp)
    app.register_blueprint(knowledge_bp)

    socketio.on_namespace(SystemEventsNamespace('/ws'))

    @app.route('/api/health', methods=['GET'])
    def health_check():
        guardrail = get_guardrail_engine()
        return jsonify({
            "status": "ok",
            "device": manager.device.type,
            "models_count": len(manager.models),
            "uptime_s": time.time() - logger.start_time,
            "guardrails": {
                "enabled": guardrail.enabled,
                "stages": 4,
                "opencv": guardrail.status().get("opencv"),
                "clip": guardrail.status().get("clip"),
            },
        })

    return app, socketio
```

### 7.2 Prediction endpoint

```python
@predict_bp.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image_bytes = file.read()
    model_name = request.form.get("model", "")
    filename = request.form.get("filename", "unknown")

    logger = get_logger()
    if logger:
        logger.info(f"Model selected: {model_name}, filename: {filename}")

    socketio = current_app.extensions.get('socketio')

    if socketio:
        socketio.emit('prediction_started', {
            'filename': filename,
            'model': model_name,
            'timestamp': time.time()
        }, namespace='/ws')

    try:
        manager = get_manager()
        result = predict_image(manager, image_bytes, model_name, socketio=socketio, filename=filename)
        result["filename"] = filename

        return jsonify(result)
    except Exception as e:
        if socketio:
            socketio.emit('prediction_error', {'error': str(e), 'filename': filename}, namespace='/ws')
        return jsonify({"error": str(e)}), 500
```

## 8. Android On-Device Inference

### 8.1 Tensor preprocessing

```javascript
export function preprocessToTensor(canvasOrImage, normalizationBaked = false) {
  const size = IMAGE_SIZE;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(canvasOrImage, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0, p = 0; i < plane; i += 1, p += 4) {
    const r = data[p] / 255;
    const g = data[p + 1] / 255;
    const b = data[p + 2] / 255;
    if (normalizationBaked) {
      tensor[i] = r;
      tensor[plane + i] = g;
      tensor[2 * plane + i] = b;
    } else {
      tensor[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      tensor[plane + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      tensor[2 * plane + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }
  }
  return tensor;
}
```

### 8.2 Gender and age estimation

```javascript
export function estimateGenderAge({ breedSlug, canvas, imageMeta = {} }) {
  const stats = canvasStats(canvas);
  const aspect = Number(imageMeta.originalAspect || 1);
  const dairy = DAIRY_BREEDS.has(String(breedSlug || "").toLowerCase());

  const femaleScore = (dairy ? 0.68 : 0.56)
    + (aspect < 1 ? 0.03 : 0)
    + (stats.dark_ratio > 0.35 ? 0.02 : 0);
  const female = femaleScore >= 0.5;
  const genderConfidence = Math.round(Math.min(0.78, Math.abs(femaleScore - 0.5) * 2 * 0.4 + 0.52) * 100);

  let ageKey = "adult";
  let ageConfidence = 62;
  if (stats.dark_ratio < 0.18 && stats.mean_luma > 0.58) {
    ageKey = "calf";
    ageConfidence = 61;
  } else if (stats.dark_ratio < 0.32 || stats.green_ratio > 0.22) {
    ageKey = "young";
    ageConfidence = 59;
  }

  return {
    gender: {
      key: female ? "female" : "male",
      label: female ? "Female" : "Male",
      confidence: genderConfidence,
      method: "on-device morphology + breed prior",
    },
    age_group: {
      key: ageKey,
      label: AGE_OPTIONS.find((item) => item.key === ageKey)?.label || ageKey,
      confidence: ageConfidence,
      method: "on-device body-size / coat prior",
    },
    stats,
  };
}
```

### 8.3 ONNX Runtime inference

```javascript
export async function runOnDeviceInference({
  canvas,
  model,
  settings,
  oodProfile = null,
  imageMeta = {},
}) {
  const record = model || (await getModel(settings?.activeModel || DEFAULT_MODEL_NAME));
  if (!record) {
    throw new Error("No on-device model selected. Open Settings and upload an ONNX model.");
  }
  if (record.metadata?.runnable === false) {
    throw new Error(record.metadata?.reason || "This checkpoint is not runnable on-device. Export it to ONNX first.");
  }
  const session = await getSession(record);
  const normalizationBaked = inferNormalizationBaked(record.name, record.metadata || {});
  const inputName = session.inputNames?.[0] || "input";
  const tensorDataArray = preprocessToTensor(canvas, normalizationBaked);
  const input = new ort.Tensor("float32", tensorDataArray, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);

  const started = performance.now();
  const outputs = await session.run({ [inputName]: input });
  const { binary, cattle, buffalo } = resolveOutputs(session, outputs);
  const binaryLogits = tensorData(binary);
  const cattleLogits = tensorData(cattle);
  const buffaloLogits = tensorData(buffalo);
  if (!binaryLogits || !cattleLogits || !buffaloLogits) {
    throw new Error("Model outputs are missing binary/cattle/buffalo heads.");
  }
  const latencyMs = performance.now() - started;

  const binaryProbs = softmax(binaryLogits);
  const cattleProbs = softmax(cattleLogits);
  const buffaloProbs = softmax(buffaloLogits);
  const cattleClasses = classListFromModel(record, "cattleClasses", EXPECTED_CATTLE_BREEDS);
  const buffaloClasses = classListFromModel(record, "buffaloClasses", EXPECTED_BUFFALO_BREEDS);

  const combined = [
    ...cattleProbs.map((p) => binaryProbs[0] * p),
    ...buffaloProbs.map((p) => binaryProbs[1] * p),
  ];
  const ranked = combined
    .map((prob, index) => ({ prob, index }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5)
    .map(({ prob, index }) => {
      const isCattle = index < cattleClasses.length;
      const slug = isCattle ? cattleClasses[index] : buffaloClasses[index - cattleClasses.length];
      return {
        breed: breedDisplayName(slug || "unknown"),
        breed_slug: slug || "unknown",
        species: isCattle ? "Cattle" : "Buffalo",
        confidence: Math.round(prob * 10000) / 100,
        prob,
      };
    });

  const top = ranked[0] || { breed: "Unknown", breed_slug: null, species: "Cattle", confidence: 0, prob: 0 };
  const speciesIndex = binaryProbs[1] > binaryProbs[0] ? 1 : 0;
  const species = speciesIndex === 1 ? "Buffalo" : "Cattle";
  const speciesConfidence = Math.round(binaryProbs[speciesIndex] * 10000) / 100;

  const activeLogits = speciesIndex === 1 ? buffaloLogits : cattleLogits;
  const activeProbs = speciesIndex === 1 ? buffaloProbs : cattleProbs;
  const embeddedOod = record.metadata?.ood && typeof record.metadata.ood === "object"
    ? record.metadata.ood
    : null;
  const activeOod = embeddedOod || oodProfile?.json_data || null;
  const temperature = Number(
    activeOod?.energy_temperature ?? settings?.energyTemperature ?? DEFAULT_OOD.energyTemperature
  ) || 1;
  const energyThreshold = Number(
    activeOod?.energy_threshold ?? settings?.energyThreshold ?? DEFAULT_OOD.energyThreshold
  );
  const mspThreshold = Number(
    activeOod?.msp_threshold ?? settings?.mspThreshold ?? DEFAULT_OOD.mspThreshold
  );
  const binaryFloor = Number(
    activeOod?.binary_confidence_floor
      ?? settings?.binaryConfidenceFloor
      ?? DEFAULT_OOD.binaryConfidenceFloor
  );
  const energy = -temperature * logSumExp(activeLogits.map((v) => v / temperature));
  const msp = Math.max(...activeProbs);
  const binaryConfidence = Math.max(...binaryProbs);
  const hasCalibration = Boolean(embeddedOod || oodProfile)
    || record.metadata?.source === "bundled";
  const requiresCalibration = !hasCalibration && record.metadata?.source === "uploaded";
  const embeddedDisabled = embeddedOod?.enabled === false;
  const oodEnabled = settings?.oodEnabled !== false
    && settings?.activeOodProfile !== "disabled"
    && !embeddedDisabled
    && !requiresCalibration;
  const oodByFloor = binaryConfidence < binaryFloor;
  const oodByEnergy = energy > energyThreshold;
  const oodByMsp = msp < mspThreshold;
  const isOod = Boolean(oodEnabled && (oodByFloor || oodByEnergy || oodByMsp));

  const attrs = estimateGenderAge({ breedSlug: top.breed_slug, canvas, imageMeta });

  return {
    model_name: record.name,
    model_source: record.metadata?.source || "uploaded",
    normalization_baked: normalizationBaked,
    species,
    predicted_species: species,
    species_confidence: speciesConfidence,
    species_prob: Math.max(...binaryProbs),
    breed: top.breed,
    top_breed: top.breed,
    breed_slug: top.breed_slug,
    confidence: top.confidence,
    top_breed_confidence: top.confidence,
    top5_breeds: ranked,
    gender: attrs.gender,
    age_group: attrs.age_group,
    is_ood: isOod,
    ood: isOod,
    ood_score: Number(energy.toFixed(4)),
    ood_details: {
      energy: Number(energy.toFixed(4)),
      energy_threshold: energyThreshold,
      msp: Number(msp.toFixed(4)),
      msp_threshold: mspThreshold,
      binary_confidence: Number(binaryConfidence.toFixed(4)),
      binary_confidence_floor: binaryFloor,
      ood_by_energy: oodByEnergy,
      ood_by_msp: oodByMsp,
      ood_by_binary_floor: oodByFloor,
      temperature,
      profile: embeddedOod ? "embedded" : (oodProfile?.filename || settings?.activeOodProfile || "auto"),
      ood_source: embeddedOod ? "embedded" : (oodProfile?.filename ? "profile" : "defaults"),
      requires_calibration: requiresCalibration,
    },
    inference_time_ms: Math.round(latencyMs * 10) / 10,
    total_time_ms: Math.round((performance.now() - started) * 10) / 10,
    routing: "on-device soft routing",
    image_stats: attrs.stats,
  };
}
```

## 9. Android Image Preparation

### 9.1 Crop and resize pipeline

```javascript
function drawPrepared(image, cropMode, base) {
  const canvas = document.createElement("canvas");
  canvas.width = base;
  canvas.height = base;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  const w = image.naturalWidth || image.width;
  const h = image.naturalHeight || image.height;
  if (!w || !h) throw new Error("Image has no dimensions.");

  if (cropMode === "pad") {
    const scale = base / Math.max(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    ctx.fillStyle = PAD_FILL;
    ctx.fillRect(0, 0, base, base);
    ctx.drawImage(image, Math.round((base - dw) / 2), Math.round((base - dh) / 2), dw, dh);
  } else {

    const scale = base / Math.min(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    ctx.drawImage(image, Math.round((base - dw) / 2), Math.round((base - dh) / 2), dw, dh);
  }
  return canvas;
}
```

### 9.2 Prepare uploaded image

```javascript
export async function prepareImage(file, {
  cropMode = "pad",
  scaleMode = "auto",
  manualCropFile = null,
} = {}) {
  const sourceFile = manualCropFile || file;
  const loaded = await loadImageFile(sourceFile);
  try {
    const base = scaleMode === "train" ? TRAIN_RESIZE : IMAGE_SIZE;
    const effectiveCrop = manualCropFile ? "center" : cropMode;
    const baseCanvas = drawPrepared(loaded.image, effectiveCrop, base);
    const canvas = centerCrop260(baseCanvas);
    const blob = await canvasToBlob(canvas);
    const preparedFile = new File(
      [blob],
      `prepared_${Date.now()}.jpg`,
      { type: "image/jpeg", lastModified: Date.now() },
    );
    return {
      file: preparedFile,
      dataUrl: canvasToDataUrl(canvas),
      canvas,
      meta: {
        cropMode: effectiveCrop,
        scaleMode,
        baseSize: base,
        originalWidth: loaded.width,
        originalHeight: loaded.height,
        originalAspect: loaded.width / Math.max(1, loaded.height),
        outputWidth: IMAGE_SIZE,
        outputHeight: IMAGE_SIZE,
        source: "upload",
      },
    };
  } finally {
    loaded.release?.();
  }
}
```

## 10. Android Result Display Settings

### 10.1 Persisted defaults

```javascript
export const DEFAULT_SETTINGS = {
  activeModel: DEFAULT_MODEL_NAME,
  oodEnabled: true,
  activeOodProfile: "auto",
  onlineMode: false,
  theme: "emerald-obsidian",
  language: "en",
  backendUrl: ENV_BACKEND_URL,
  downscaleMode: "auto",
  showConfidence: true,
  topBreedsCount: 3,
  showPredictionView: true,
  energyThreshold: DEFAULT_OOD.energyThreshold,
  mspThreshold: DEFAULT_OOD.mspThreshold,
  binaryConfidenceFloor: DEFAULT_OOD.binaryConfidenceFloor,
  energyTemperature: DEFAULT_OOD.energyTemperature,
};
```
