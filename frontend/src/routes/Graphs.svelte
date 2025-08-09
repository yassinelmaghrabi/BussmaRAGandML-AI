<script lang="ts">
  import { API_BASE_URL } from "../lib/apiConfig";

  let showRulesConfig = false;
  let showGraphConfig = false;
  let showPredictConfig = false;
  let container: HTMLDivElement;

  // Defaults for Rules
  let rulesConfig = {
    brand_col: "brand_name",
    invoice_col: "invoice",
    min_support: 0.01,
    min_confidence: 0.1,
    min_lift: 1.0,
    max_len: 2,
    top_n_brands: 200,
    min_brand_count: 20,
    sort_by: "lift",
    max_rules_display: 100,
    figure_width: 1400,
    figure_height: 800,
    return_html: true,
    include_metrics_plot: true,
  };

  // Defaults for Graph
  let graphConfig = {
    ingredient_col: "ingredient_name",
    invoice_col: "invoice",
    min_product_count: 50,
    top_n_products: 160,
    cooccurrence_percentile: 75.0,
    lift_threshold: 1.1,
    max_connections: 150,
    clustering_method: "dbscan",
    dbscan_eps_range: "0.3,1",
    dbscan_min_samples: 2,
    kmeans_k_range: "3,12",
    max_clusters_display: 12,
    figure_width: 1500,
    figure_height: 1000,
    node_size_multiplier: 1.0,
    show_legend: true,
    return_html: true,
  };

  // Defaults for Predict Sales
  let predictSalesConfig = {
    target_col: "sales_sheet",
    sequence_length: 12,
    forecast_periods: 30,
    smoothing_span: 5,
    lstm_units: 128,
    lstm_epochs: 150,
    dropout_rate: 0.3,
    early_stopping: true,
    confidence_level: 0.95,
    theme: "dark",
    remove_last_n: 1,
  };

  async function fetchGraph(endpoint: string, params: Record<string, any>) {
    const query = new URLSearchParams(
      Object.fromEntries(
        Object.entries(params).map(([k, v]) => [k, String(v)]),
      ),
    );
    const res = await fetch(`${API_BASE_URL}/${endpoint}?${query}`);
    const html = await res.text();
    container.innerHTML = html;

    container.querySelectorAll("script").forEach((oldScript) => {
      const newScript = document.createElement("script");
      if (oldScript.src) {
        newScript.src = oldScript.src;
      } else {
        newScript.textContent = oldScript.textContent;
      }
      document.body.appendChild(newScript);
      document.body.removeChild(newScript);
    });
  }

  async function fetchPredictSales(config: Record<string, any>) {
    const res = await fetch(`${API_BASE_URL}/predict-sales`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
    const html = await res.text();
    container.innerHTML = html;

    container.querySelectorAll("script").forEach((oldScript) => {
      const newScript = document.createElement("script");
      if (oldScript.src) {
        newScript.src = oldScript.src;
      } else {
        newScript.textContent = oldScript.textContent;
      }
      document.body.appendChild(newScript);
      document.body.removeChild(newScript);
    });
  }
</script>

<h1>AI/ML Graphs</h1>

<!-- Top level buttons -->
<button
  on:click={() => {
    showRulesConfig = !showRulesConfig;
    showGraphConfig = false;
    showPredictConfig = false;
  }}
>
  Configure & Render Association Rules
</button>
<button
  on:click={() => {
    showGraphConfig = !showGraphConfig;
    showRulesConfig = false;
    showPredictConfig = false;
  }}
>
  Configure & Render Association Graph
</button>
<button
  on:click={() => {
    showPredictConfig = !showPredictConfig;
    showRulesConfig = false;
    showGraphConfig = false;
  }}
>
  Configure & Run Sales Prediction
</button>

<!-- Rules Config -->
{#if showRulesConfig}
  <h2>Association Rules Config</h2>
  {#each Object.entries(rulesConfig) as [key, value]}
    <label>{key}: <input bind:value={rulesConfig[key]} /></label><br />
  {/each}
  <button on:click={() => fetchGraph("association-rules", rulesConfig)}>
    Render Graph
  </button>
{/if}

<!-- Graph Config -->
{#if showGraphConfig}
  <h2>Association Graph Config</h2>
  {#each Object.entries(graphConfig) as [key, value]}
    <label>{key}: <input bind:value={graphConfig[key]} /></label><br />
  {/each}
  <button on:click={() => fetchGraph("association-graph", graphConfig)}>
    Render Graph
  </button>
{/if}

<!-- Predict Sales Config -->
{#if showPredictConfig}
  <h2>Predict Sales Config</h2>
  {#each Object.entries(predictSalesConfig) as [key, value]}
    <label>{key}: <input bind:value={predictSalesConfig[key]} /></label><br />
  {/each}
  <button on:click={() => fetchPredictSales(predictSalesConfig)}>
    Run Prediction
  </button>
{/if}

<!-- Output -->
<div bind:this={container}></div>

<style>
  /* Overall background and font */
  .graphs-gradient {
    background: radial-gradient(
      ellipse at 50% 50%,
      #0f766e 0%,
      /* teal-800 */ #164e63 50%,
      /* cyan-900 */ #082f49 100% /* deeper blue-gray */
    );
    color: #e0f2f1; /* light teal text */
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    min-height: auto;
    padding: 20px;
    box-sizing: border-box;
  }

  /* Headers */
  h1,
  h2 {
    color: #a5d6a7; /* lighter teal */
    font-weight: 700;
    margin-bottom: 12px;
  }

  /* Buttons */
  button {
    background-color: #0f766e; /* teal-800 */
    border: none;
    color: #e0f2f1;
    padding: 10px 16px;
    margin: 6px 0;
    font-size: 1rem;
    border-radius: 6px;
    cursor: pointer;
    transition: background-color 0.3s ease;
  }

  button:hover {
    background-color: #164e63; /* cyan-900 */
  }

  /* Input fields and labels */
  label {
    display: flex;
    justify-content: space-between;
    margin: 6px 0;
    font-weight: 600;
  }

  input {
    background-color: #225c55; /* dark teal */
    border: 1px solid #0f766e;
    color: #e0f2f1;
    padding: 6px 10px;
    border-radius: 4px;
    width: 180px;
    transition: border-color 0.3s ease;
  }

  input:focus {
    outline: none;
    border-color: #a5d6a7;
    background-color: #1a4f48;
  }

  /* Output container remains clean for Plotly */
  .graphs-gradient > div {
    margin-top: 20px;
  }
</style>
