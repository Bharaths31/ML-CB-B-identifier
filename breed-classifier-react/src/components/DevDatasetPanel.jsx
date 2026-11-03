import { useState, useEffect, useMemo } from "react";
import { getDatasetInventory } from "../api.js";
import { BREED_DATASET_COUNTS, DATASET_SUMMARY } from "../data/breedDatasetCounts.js";

export default function DevDatasetPanel() {
  const [breeds, setBreeds] = useState(BREED_DATASET_COUNTS || []);
  const [summary, setSummary] = useState(DATASET_SUMMARY || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Filters & sorting
  const [searchQuery, setSearchQuery] = useState("");
  const [speciesFilter, setSpeciesFilter] = useState("all"); // 'all' | 'cattle' | 'buffalo'
  const [tierFilter, setTierFilter] = useState("all"); // 'all' | 'many_shot' | 'medium_shot' | 'few_shot'
  const [sortBy, setSortBy] = useState("count_desc"); // 'count_desc' | 'count_asc' | 'name_asc' | 'name_desc'

  const fetchInventory = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getDatasetInventory();
      if (data?.breeds && Array.isArray(data.breeds)) {
        setBreeds(data.breeds);
      }
      if (data?.totals || data?.total_images) {
        setSummary({
          datasetName: data.dataset || "Atharvadarpude Indian Cattle & Buffalo Dataset",
          datasetSlugs: data.dataset_slugs || [
            "atharvadarpude/indian-cattle-image-dataset",
            "atharvadarpude/indian-buffalo-dataset",
          ],
          totalImages: data.total_images || data.totals?.images || 11199,
          totalBreeds: data.total_breeds || data.totals?.breeds || 68,
          canonicalBreeds: data.canonical_breeds_count || 68,
          bySpecies: data.by_species || data.totals?.by_species || {},
          shotTiers: data.shot_tiers || {},
        });
      }
    } catch (err) {
      console.warn("Using offline Atharvadarpude dataset cache:", err.message);
      setBreeds(BREED_DATASET_COUNTS || []);
      setSummary(DATASET_SUMMARY || null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchInventory();
  }, []);

  // Filtered and sorted dataset
  const filteredBreeds = useMemo(() => {
    let result = [...breeds];

    // Search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (b) =>
          b.name.toLowerCase().includes(q) ||
          b.key.toLowerCase().includes(q) ||
          (b.origin && b.origin.toLowerCase().includes(q)) ||
          (b.distinguisher && b.distinguisher.toLowerCase().includes(q))
      );
    }

    // Species filter
    if (speciesFilter !== "all") {
      result = result.filter((b) => b.species === speciesFilter);
    }

    // Shot tier filter
    if (tierFilter !== "all") {
      result = result.filter((b) => b.tier_code === tierFilter);
    }

    // Sorting
    result.sort((a, b) => {
      if (sortBy === "count_desc") {
        return b.count - a.count || a.name.localeCompare(b.name);
      }
      if (sortBy === "count_asc") {
        return a.count - b.count || a.name.localeCompare(b.name);
      }
      if (sortBy === "name_asc") {
        return a.name.localeCompare(b.name);
      }
      if (sortBy === "name_desc") {
        return b.name.localeCompare(a.name);
      }
      return 0;
    });

    return result;
  }, [breeds, searchQuery, speciesFilter, tierFilter, sortBy]);

  // Max count for relative visual bar
  const maxCount = useMemo(() => {
    if (!breeds.length) return 1;
    return Math.max(...breeds.map((b) => b.count || 0), 1);
  }, [breeds]);

  // Export CSV
  const handleExportCSV = () => {
    const headers = [
      "Rank",
      "Breed Key",
      "Breed Name",
      "Species",
      "Dataset Images",
      "Dataset Share (%)",
      "Shot Tier",
      "Origin / State",
      "Key Morphological Trait",
      "Kaggle Dataset Source",
    ];

    const rows = filteredBreeds.map((b, idx) => [
      idx + 1,
      b.key,
      `"${(b.name || "").replace(/"/g, '""')}"`,
      b.species,
      b.count,
      b.percentage || 0,
      `"${b.tier}"`,
      `"${(b.origin || "").replace(/"/g, '""')}"`,
      `"${(b.distinguisher || "").replace(/"/g, '""')}"`,
      `"${b.dataset_source || ""}"`,
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute(
      "download",
      `atharvadarpude_breed_dataset_inventory_${new Date().toISOString().slice(0, 10)}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const cattleCount = breeds.filter((b) => b.species === "cattle").length;
  const buffaloCount = breeds.filter((b) => b.species === "buffalo").length;
  const totalImgs = summary?.totalImages || 11199;
  const cattleImgs = summary?.bySpecies?.cattle?.images || 8531;
  const buffaloImgs = summary?.bySpecies?.buffalo?.images || 2668;

  return (
    <div className="dev-panel-content">
      {/* Header Block */}
      <div className="panel-header-block">
        <div className="title-row-between">
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <h2 className="panel-title" style={{ margin: 0 }}>
                Atharvadarpude Dataset Inventory & Breed Distribution
              </h2>
              <span className="badge badge-info" style={{ fontSize: "0.74rem" }}>
                Kaggle Atharvadarpude
              </span>
            </div>
            <p className="panel-description">
              Primary project dataset by <strong>Atharva Darpude</strong> comprising{" "}
              <strong>50 Indian Cattle breeds</strong> (<code>atharvadarpude/indian-cattle-image-dataset</code>) and{" "}
              <strong>18 Indian Buffalo breeds</strong> (<code>atharvadarpude/indian-buffalo-dataset</code>),
              arranged by dataset image volume from highest to lowest.
            </p>
          </div>
          <div className="header-actions-row">
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={fetchInventory}
              disabled={loading}
              title="Refresh inventory from server"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: 6 }}>
                <polyline points="23 4 23 10 17 10"></polyline>
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
              </svg>
              {loading ? "Refreshing..." : "Refresh"}
            </button>
            <button
              type="button"
              className="btn btn-sm btn-primary"
              onClick={handleExportCSV}
              title="Download CSV table of Atharvadarpude breed image counts"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ marginRight: 6 }}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              Export CSV
            </button>
          </div>
        </div>
      </div>

      {/* KPI Summary Cards */}
      <div className="dev-kpi-grid">
        <div className="dev-kpi-card">
          <span className="dev-kpi-label">Atharvadarpude Total Images</span>
          <div className="dev-kpi-value">{totalImgs.toLocaleString()}</div>
          <span className="dev-kpi-sub">
            Cattle: <strong>{cattleImgs.toLocaleString()}</strong> ({summary?.bySpecies?.cattle?.percentage || "76.2"}%) | Buffalo:{" "}
            <strong>{buffaloImgs.toLocaleString()}</strong> ({summary?.bySpecies?.buffalo?.percentage || "23.8"}%)
          </span>
        </div>

        <div className="dev-kpi-card">
          <span className="dev-kpi-label">Registered Indian Breeds</span>
          <div className="dev-kpi-value">{breeds.length}</div>
          <span className="dev-kpi-sub">
            <strong>{cattleCount} Cattle</strong> (8,531 imgs) + <strong>{buffaloCount} Buffalo</strong> (2,668 imgs)
          </span>
        </div>

        <div className="dev-kpi-card">
          <span className="dev-kpi-label">Top Represented Breed</span>
          <div className="dev-kpi-value" style={{ fontSize: "1.28rem", color: "var(--primary, #2563eb)" }}>
            {breeds[0]?.name || "Gir"} (Cattle)
          </div>
          <span className="dev-kpi-sub">
            <strong>{breeds[0]?.count?.toLocaleString() || 686}</strong> images (
            {breeds[0]?.percentage || 6.13}% of total dataset)
          </span>
        </div>

        <div className="dev-kpi-card">
          <span className="dev-kpi-label">Training Imbalance Tiers</span>
          <div className="dev-tier-pills-row">
            <span className="badge badge-success" title="Many-shot (>100 images)">
              Many (&gt;100): {breeds.filter((b) => b.count >= 100).length}
            </span>
            <span className="badge badge-warning" title="Medium-shot (30-100 images)">
              Med (30-100): {breeds.filter((b) => b.count >= 30 && b.count < 100).length}
            </span>
            <span className="badge badge-info" title="Few-shot (<30 images)">
              Few (&lt;30): {breeds.filter((b) => b.count > 0 && b.count < 30).length}
            </span>
          </div>
          <span className="dev-kpi-sub">Managed via Effective-Number Sampler (β=0.99)</span>
        </div>
      </div>

      {/* Filter and Control Toolbar */}
      <div className="dev-card" style={{ marginTop: 16, padding: "14px 18px" }}>
        <div className="dev-toolbar-flex">
          {/* Search Box */}
          <div className="search-input-wrap">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="search-icon">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <input
              type="text"
              className="dev-input search-box-input"
              placeholder="Search Atharvadarpude breeds by name, state, or traits..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
            {searchQuery && (
              <button
                type="button"
                className="search-clear-btn"
                onClick={() => setSearchQuery("")}
                title="Clear search"
              >
                ×
              </button>
            )}
          </div>

          {/* Species Filter Pills */}
          <div className="btn-group-segmented">
            <button
              type="button"
              className={`segmented-btn ${speciesFilter === "all" ? "active" : ""}`}
              onClick={() => setSpeciesFilter("all")}
            >
              All Breeds (68)
            </button>
            <button
              type="button"
              className={`segmented-btn ${speciesFilter === "cattle" ? "active" : ""}`}
              onClick={() => setSpeciesFilter("cattle")}
            >
              🐄 Cattle (50)
            </button>
            <button
              type="button"
              className={`segmented-btn ${speciesFilter === "buffalo" ? "active" : ""}`}
              onClick={() => setSpeciesFilter("buffalo")}
            >
              🐃 Buffalo (18)
            </button>
          </div>

          {/* Shot Tier Filter */}
          <div className="btn-group-segmented">
            <button
              type="button"
              className={`segmented-btn ${tierFilter === "all" ? "active" : ""}`}
              onClick={() => setTierFilter("all")}
            >
              All Tiers
            </button>
            <button
              type="button"
              className={`segmented-btn ${tierFilter === "many_shot" ? "active" : ""}`}
              onClick={() => setTierFilter("many_shot")}
              title=">100 images"
            >
              &gt;100 imgs
            </button>
            <button
              type="button"
              className={`segmented-btn ${tierFilter === "medium_shot" ? "active" : ""}`}
              onClick={() => setTierFilter("medium_shot")}
              title="30-100 images"
            >
              30-100 imgs
            </button>
          </div>

          {/* Sort Selector */}
          <div className="sort-select-wrap">
            <label htmlFor="dev-sort-select" className="text-muted" style={{ fontSize: "0.82rem", marginRight: 6 }}>
              Sort:
            </label>
            <select
              id="dev-sort-select"
              className="dev-input dev-select-input"
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
            >
              <option value="count_desc">Images: High → Low</option>
              <option value="count_asc">Images: Low → High</option>
              <option value="name_asc">Name: A → Z</option>
              <option value="name_desc">Name: Z → A</option>
            </select>
          </div>
        </div>
      </div>

      {/* Showing count indicator */}
      <div className="dev-results-meta">
        <span>
          Showing <strong>{filteredBreeds.length}</strong> of <strong>{breeds.length}</strong> breeds
          {speciesFilter !== "all" && ` (${speciesFilter === "cattle" ? "Cattle Dataset" : "Buffalo Dataset"})`}
          {searchQuery && ` matching "${searchQuery}"`}
        </span>
        {(searchQuery || speciesFilter !== "all" || tierFilter !== "all" || sortBy !== "count_desc") && (
          <button
            type="button"
            className="btn btn-sm btn-link"
            onClick={() => {
              setSearchQuery("");
              setSpeciesFilter("all");
              setTierFilter("all");
              setSortBy("count_desc");
            }}
          >
            Reset Filters
          </button>
        )}
      </div>

      {/* Main Breed Table */}
      <div className="dev-card" style={{ padding: 0, overflow: "hidden", marginTop: 8 }}>
        <div className="table-responsive-wrapper">
          <table className="dev-comp-table breed-inventory-table">
            <thead>
              <tr>
                <th style={{ width: "6%" }}># Rank</th>
                <th style={{ width: "22%" }}>Breed Name & Identifier</th>
                <th style={{ width: "11%" }}>Species</th>
                <th style={{ width: "23%" }}>Dataset Images & Share</th>
                <th style={{ width: "12%" }}>Shot Tier</th>
                <th style={{ width: "12%" }}>Origin / State</th>
                <th style={{ width: "14%" }}>Key Morphological Trait</th>
              </tr>
            </thead>
            <tbody>
              {filteredBreeds.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: "center", padding: "36px 16px", color: "var(--text-muted)" }}>
                    No breeds match the selected search or filter criteria.
                  </td>
                </tr>
              ) : (
                filteredBreeds.map((breed) => {
                  const barWidth = Math.max(Math.round(((breed.count || 0) / maxCount) * 100), 2);
                  const isTop3 = breed.rank <= 3 && sortBy === "count_desc";

                  return (
                    <tr key={`${breed.species}_${breed.key}`} className={isTop3 ? "top-rank-row" : ""}>
                      {/* Rank */}
                      <td>
                        <span
                          className={`rank-badge ${
                            breed.rank === 1
                              ? "rank-gold"
                              : breed.rank === 2
                              ? "rank-silver"
                              : breed.rank === 3
                              ? "rank-bronze"
                              : "rank-normal"
                          }`}
                        >
                          {breed.rank === 1 ? "🥇 1" : breed.rank === 2 ? "🥈 2" : breed.rank === 3 ? "🥉 3" : `#${breed.rank}`}
                        </span>
                      </td>

                      {/* Breed Name & Key */}
                      <td>
                        <div className="breed-title-cell">
                          <span className="breed-name-text">{breed.name}</span>
                          <div className="breed-tags-row">
                            <code className="breed-key-tag">{breed.key}</code>
                            <span className="badge badge-canonical" title="ICAR-NBAGR Registered Canonical Breed">
                              Canonical Class
                            </span>
                          </div>
                        </div>
                      </td>

                      {/* Species */}
                      <td>
                        <span className={`badge ${breed.species === "cattle" ? "badge-cattle" : "badge-buffalo"}`}>
                          {breed.species === "cattle" ? "🐄 Cattle" : "🐃 Buffalo"}
                        </span>
                      </td>

                      {/* Count & Visual Progress Bar */}
                      <td>
                        <div className="count-bar-container">
                          <div className="count-numbers-row">
                            <span className="count-highlight">
                              <strong>{(breed.count || 0).toLocaleString()}</strong>
                              <span className="count-unit"> imgs</span>
                            </span>
                            <span className="percentage-text text-muted">
                              {breed.percentage ? `${breed.percentage}%` : "—"}
                            </span>
                          </div>
                          <div className="dataset-progress-track">
                            <div
                              className={`dataset-progress-bar ${
                                breed.species === "cattle" ? "bar-cattle" : "bar-buffalo"
                              }`}
                              style={{ width: `${barWidth}%` }}
                            />
                          </div>
                        </div>
                      </td>

                      {/* Shot Tier */}
                      <td>
                        <span
                          className={`badge ${
                            breed.tier_code === "many_shot"
                              ? "badge-success"
                              : breed.tier_code === "medium_shot"
                              ? "badge-warning"
                              : "badge-info"
                          }`}
                        >
                          {breed.tier}
                        </span>
                      </td>

                      {/* Origin */}
                      <td>
                        <span className="origin-text text-muted" title={breed.origin}>
                          {breed.origin || "India"}
                        </span>
                      </td>

                      {/* Distinguisher / Trait */}
                      <td>
                        <span className="trait-snippet-text" title={breed.distinguisher}>
                          {breed.distinguisher || "Recognized indigenous breed of India."}
                        </span>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
