import {
  DISPLAY_TOGGLE_OPTIONS,
  USER_SECTION_OPTIONS,
  hiddenUserSections,
} from "../../breed-classifier-react/src/utils/userVisibility.js";

/**
 * Vet-controlled visibility for the farmer-facing app.
 *
 * Writes `display.hidden_user_sections` + the display flags to the shared
 * config; every open user page updates over the /ws `config_changed` event.
 */
export default function UserVisibilityPanel({ config, onUpdateConfig }) {
  const display = config?.display || {};
  const hidden = hiddenUserSections(config);

  const update = async (patch) => {
    try {
      await onUpdateConfig(patch);
    } catch (err) {
      alert("Failed to update user visibility: " + err.message);
    }
  };

  const setSection = (id, visible) => {
    const next = new Set(hidden);
    if (visible) next.delete(id);
    else next.add(id);
    update({ "display.hidden_user_sections": Array.from(next) });
  };

  const setAll = (visible) => {
    update({
      "display.hidden_user_sections": visible
        ? []
        : USER_SECTION_OPTIONS.map((option) => option.id),
    });
  };

  return (
    <section className="expert-visibility">
      <div className="expert-toolbar">
        <h2>User Page Visibility</h2>
        <div className="expert-filters">
          <button type="button" className="preset-chip" onClick={() => setAll(true)}>
            Show all
          </button>
          <button type="button" className="preset-chip" onClick={() => setAll(false)}>
            Hide all sections
          </button>
        </div>
      </div>

      <div className="dev-grid-2col">
        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">Visible Content</h3>
            <span className="badge badge-neutral">
              {USER_SECTION_OPTIONS.length - hidden.length}/{USER_SECTION_OPTIONS.length} shown
            </span>
          </div>
          <p className="slider-hint" style={{ marginBottom: 12 }}>
            Turn a block off to hide it from the farmer-facing page. Changes
            apply instantly to every connected user.
          </p>

          {USER_SECTION_OPTIONS.map((option) => {
            const visible = !hidden.includes(option.id);
            return (
              <div className="control-setting-row" key={option.id}>
                <div className="setting-info">
                  <span className="setting-label">{option.label}</span>
                  <span className="setting-subtext">{option.help}</span>
                </div>
                <label className="toggle-switch">
                  <input
                    type="checkbox"
                    checked={visible}
                    onChange={(e) => setSection(option.id, e.target.checked)}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>
            );
          })}
        </div>

        <div className="dev-card">
          <div className="dev-card-header">
            <h3 className="card-heading">Display Details</h3>
            <span className="badge badge-outline">Shared Config</span>
          </div>
          <p className="slider-hint" style={{ marginBottom: 12 }}>
            These flags were moved out of the Developer Console so clinical
            staff control what farmers see.
          </p>

          {DISPLAY_TOGGLE_OPTIONS.map((option) => (
            <div className="control-setting-row" key={option.key}>
              <div className="setting-info">
                <span className="setting-label">{option.label}</span>
                <span className="setting-subtext">{option.help}</span>
              </div>
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={display[option.key] ?? option.default}
                  onChange={(e) => update({ [`display.${option.key}`]: e.target.checked })}
                />
                <span className="toggle-slider" />
              </label>
            </div>
          ))}

          <div className="dev-subcard" style={{ marginTop: 16 }}>
            <h4 className="card-heading" style={{ fontSize: "0.92rem" }}>
              Applied to the user page
            </h4>
            <p className="slider-hint">
              Hidden sections are removed from the DOM (not merely collapsed), so
              shared kiosk/link deployments stay minimal.
            </p>
            <pre className="json-preview-code" style={{ maxHeight: 160, marginTop: 8 }}>
              {JSON.stringify({
                hidden_user_sections: hidden,
                show_species_badge: display.show_species_badge ?? true,
                show_species_confidence: display.show_species_confidence ?? true,
                show_escalation_warning: display.show_escalation_warning ?? true,
                show_technical_details: display.show_technical_details ?? false,
              }, null, 2)}
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}
