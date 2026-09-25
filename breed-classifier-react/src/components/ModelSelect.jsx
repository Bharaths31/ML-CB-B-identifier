export default function ModelSelect({ models, value, onChange, meta }) {
  return (
    <div className="model-selector">
      <label htmlFor="model-select">Model</label>
      <select
        id="model-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {models.map((m) => (
          <option key={m.name} value={m.name}>
            {m.name} ({m.size_mb} MB)
          </option>
        ))}
      </select>
      {meta && (
        <div className="model-meta">
          Backbone: {meta.backbone} · {meta.size_mb} MB
        </div>
      )}
    </div>
  );
}
