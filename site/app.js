const paths = {
  eda: "../reports/eda_overview.json",
  centralized: "../reports/centralized_baseline_results.csv",
  fedavg: "../reports/fedavg_baseline_results.csv",
  fedavgHistory: "../reports/fedavg_training_history.csv",
  factories: "../data/factories/factory_partition_summary.csv",
  centralizedModel: "../data/processed/centralized_logistic_model.json",
  fedavgModels: "../data/processed/fedavg_models.json",
  standardization: "../data/processed/standardization_parameters.csv"
};

const colors = {
  blue: "#2568c4",
  teal: "#177c86",
  green: "#16805d",
  amber: "#b16b00",
  red: "#b74343",
  violet: "#6657a8",
  line: "#d9e2e5",
  muted: "#617073"
};

let factoryRows = [];
let currentStrategy = "iid";
let predictionModels = {};
let standardization = {};

const defaultInput = {
  model: "centralized",
  type: "M",
  airTemp: 300.0,
  processTemp: 310.0,
  rotSpeed: 1538,
  torque: 40.0,
  toolWear: 108
};

const highRiskInput = {
  model: "centralized",
  type: "L",
  airTemp: 302.6,
  processTemp: 310.4,
  rotSpeed: 1320,
  torque: 62.5,
  toolWear: 224
};

const featureLabels = {
  air_temperature_k_z: "Air temperature",
  process_temperature_k_z: "Process temperature",
  rotational_speed_rpm_z: "Rotational speed",
  torque_nm_z: "Torque",
  tool_wear_min_z: "Tool wear",
  temperature_gap_k_z: "Temperature gap",
  power_proxy_z: "Power proxy",
  Type_H: "Type H",
  Type_L: "Type L",
  Type_M: "Type M"
};

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines.shift().split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return headers.reduce((row, header, index) => {
      row[header] = values[index];
      return row;
    }, {});
  });
}

async function loadText(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  return response.text();
}

async function loadCSV(path) {
  return parseCSV(await loadText(path));
}

function pct(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function num(value, digits = 3) {
  return Number(value).toFixed(digits);
}

function maxBy(items, key) {
  return items.reduce((best, item) => Number(item[key]) > Number(best[key]) ? item : best, items[0]);
}

function dominantMode(row) {
  const modes = ["mode_HDF", "mode_PWF", "mode_OSF", "mode_TWF", "mode_RNF"];
  const winner = modes.reduce((best, key) => Number(row[key]) > Number(row[best]) ? key : best, modes[0]);
  return winner.replace("mode_", "");
}

function showError(message) {
  document.querySelector("#snapshot-status").textContent = "Data loading failed";
  const target = document.querySelector("#results");
  const error = document.createElement("div");
  error.className = "load-error";
  error.textContent = message;
  target.prepend(error);
}

function buildStandardization(rows) {
  return rows.reduce((map, row) => {
    map[row.feature] = {
      mean: Number(row.mean),
      std: Number(row.std)
    };
    return map;
  }, {});
}

function buildPredictionModels(centralizedModel, fedavgModels) {
  return {
    centralized: {
      label: "Centralized weighted logistic",
      ...centralizedModel
    },
    fedavg_iid: {
      label: "FedAvg IID",
      ...fedavgModels.iid
    },
    fedavg_moderate_non_iid: {
      label: "FedAvg Moderate Non-IID",
      ...fedavgModels.moderate_non_iid
    },
    fedavg_highly_non_iid: {
      label: "FedAvg Highly Non-IID",
      ...fedavgModels.highly_non_iid
    }
  };
}

function zScore(rawFeature, value) {
  const params = standardization[rawFeature];
  if (!params || !params.std) {
    return 0;
  }
  return (value - params.mean) / params.std;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function readPredictionInput() {
  return {
    model: document.querySelector("#model-select").value,
    type: document.querySelector("#input-type").value,
    airTemp: Number(document.querySelector("#air-temp").value),
    processTemp: Number(document.querySelector("#process-temp").value),
    rotSpeed: Number(document.querySelector("#rot-speed").value),
    torque: Number(document.querySelector("#torque").value),
    toolWear: Number(document.querySelector("#tool-wear").value)
  };
}

function writePredictionInput(values) {
  document.querySelector("#model-select").value = values.model;
  document.querySelector("#input-type").value = values.type;
  document.querySelector("#air-temp").value = values.airTemp;
  document.querySelector("#process-temp").value = values.processTemp;
  document.querySelector("#rot-speed").value = values.rotSpeed;
  document.querySelector("#torque").value = values.torque;
  document.querySelector("#tool-wear").value = values.toolWear;
}

function buildFeatureVector(input) {
  const temperatureGap = input.processTemp - input.airTemp;
  const powerProxy = input.rotSpeed * input.torque;

  return {
    values: {
      air_temperature_k_z: zScore("Air temperature [K]", input.airTemp),
      process_temperature_k_z: zScore("Process temperature [K]", input.processTemp),
      rotational_speed_rpm_z: zScore("Rotational speed [rpm]", input.rotSpeed),
      torque_nm_z: zScore("Torque [Nm]", input.torque),
      tool_wear_min_z: zScore("Tool wear [min]", input.toolWear),
      temperature_gap_k_z: zScore("temperature_gap [K]", temperatureGap),
      power_proxy_z: zScore("power_proxy", powerProxy),
      Type_H: input.type === "H" ? 1 : 0,
      Type_L: input.type === "L" ? 1 : 0,
      Type_M: input.type === "M" ? 1 : 0
    },
    derived: {
      temperatureGap,
      powerProxy
    }
  };
}

function predictFailure(input) {
  const model = predictionModels[input.model];
  const vector = buildFeatureVector(input);
  const contributions = model.features.map((feature) => {
    const coefficient = Number(model.coefficients[feature] || 0);
    const value = Number(vector.values[feature] || 0);
    return {
      feature,
      label: featureLabels[feature] || feature,
      coefficient,
      value,
      contribution: coefficient * value
    };
  });

  const logit = Number(model.intercept) + contributions.reduce((sum, item) => sum + item.contribution, 0);
  const probability = sigmoid(logit);
  const threshold = Number(model.threshold);
  return {
    model,
    probability,
    threshold,
    decision: probability >= threshold ? "Failure Predicted" : "Normal Predicted",
    isHighRisk: probability >= threshold,
    derived: vector.derived,
    contributions
  };
}

function renderPrediction(result) {
  const probabilityPercent = Math.round(result.probability * 1000) / 10;
  document.querySelector("#output-model").textContent = result.model.label;
  document.querySelector("#risk-probability").textContent = `${probabilityPercent.toFixed(1)}%`;
  document.querySelector("#risk-ring").style.setProperty("--risk", `${probabilityPercent}%`);
  document.querySelector("#risk-ring").style.background =
    `radial-gradient(circle at center, #ffffff 0 56%, transparent 57%), conic-gradient(${result.isHighRisk ? colors.red : colors.green} ${probabilityPercent}%, #e5ecef 0)`;

  const label = document.querySelector("#risk-label");
  label.textContent = result.isHighRisk ? "High Risk" : "Low Risk";
  label.className = `result-badge ${result.isHighRisk ? "high" : "low"}`;

  document.querySelector("#risk-text").textContent = result.isHighRisk
    ? "The predicted probability exceeds the model threshold. Further inspection or maintenance scheduling is recommended."
    : "The predicted probability is below the model threshold. Under the current inputs, the machine is more likely to be normal.";
  document.querySelector("#output-threshold").textContent = result.threshold.toFixed(2);
  document.querySelector("#output-gap").textContent = `${result.derived.temperatureGap.toFixed(2)} K`;
  document.querySelector("#output-power").textContent = result.derived.powerProxy.toLocaleString("en-US", { maximumFractionDigits: 1 });
  document.querySelector("#output-decision").textContent = result.decision;

  const sorted = [...result.contributions]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, 6);
  const maxAbs = Math.max(...sorted.map((item) => Math.abs(item.contribution)), 0.001);

  document.querySelector("#contribution-list").innerHTML = sorted.map((item) => {
    const width = Math.max(5, Math.abs(item.contribution) / maxAbs * 100);
    return `
      <div class="contribution-item ${item.contribution < 0 ? "negative" : "positive"}">
        <div>
          <strong>${item.label}</strong>
          <div class="contribution-bar"><span style="width:${width}%"></span></div>
        </div>
        <span class="contribution-score">${item.contribution >= 0 ? "+" : ""}${item.contribution.toFixed(3)}</span>
      </div>
    `;
  }).join("");
}

function runPrediction() {
  const input = readPredictionInput();
  const numericValues = [input.airTemp, input.processTemp, input.rotSpeed, input.torque, input.toolWear];
  if (numericValues.some((value) => !Number.isFinite(value))) {
    document.querySelector("#contribution-list").innerHTML = `<div class="load-error">Please complete all numeric fields.</div>`;
    return;
  }
  renderPrediction(predictFailure(input));
}

function bindPredictionControls() {
  writePredictionInput(defaultInput);
  document.querySelector("#prediction-form").addEventListener("submit", (event) => {
    event.preventDefault();
    runPrediction();
  });
  document.querySelector("#example-button").addEventListener("click", () => {
    writePredictionInput(highRiskInput);
    runPrediction();
  });
  document.querySelector("#reset-button").addEventListener("click", () => {
    writePredictionInput(defaultInput);
    runPrediction();
  });
  document.querySelector("#model-select").addEventListener("change", runPrediction);
  runPrediction();
}

function updateOverview(eda, fedavgResults) {
  document.querySelector("#metric-rows").textContent = Number(eda.rows).toLocaleString("en-US");
  document.querySelector("#metric-failure-rate").textContent = pct(eda.target_positive_rate, 2);

  const bestFedAvg = maxBy(fedavgResults, "f1");
  document.querySelector("#snapshot-status").textContent =
    `Best FedAvg F1: ${num(bestFedAvg.f1)} (${bestFedAvg.strategy})`;
}

function buildResultRows(centralizedResults, fedavgResults) {
  const centralizedTest = centralizedResults.find((row) =>
    row.model === "Weighted Logistic Regression" && row.split === "test"
  );

  const rows = [
    {
      name: "Centralized Weighted Logistic",
      threshold: centralizedTest.threshold,
      accuracy: centralizedTest.accuracy,
      precision: centralizedTest.precision,
      recall: centralizedTest.recall,
      f1: centralizedTest.f1,
      tp: centralizedTest.tp,
      fp: centralizedTest.fp,
      fn: centralizedTest.fn
    },
    ...fedavgResults.map((row) => ({
      name: `FedAvg ${row.strategy}`,
      threshold: row.threshold,
      accuracy: row.accuracy,
      precision: row.precision,
      recall: row.recall,
      f1: row.f1,
      tp: row.tp,
      fp: row.fp,
      fn: row.fn
    }))
  ];

  rows.sort((a, b) => Number(b.f1) - Number(a.f1));
  return rows;
}

function renderResultsTable(rows) {
  const body = document.querySelector("#results-table");
  body.innerHTML = rows.map((row, index) => `
    <tr>
      <td><span class="badge">${index === 0 ? "Best" : "Model"}</span> ${row.name}</td>
      <td>${row.threshold}</td>
      <td>${num(row.accuracy)}</td>
      <td>${num(row.precision)}</td>
      <td>${num(row.recall)}</td>
      <td><strong>${num(row.f1)}</strong></td>
      <td>${row.tp} / ${row.fp} / ${row.fn}</td>
    </tr>
  `).join("");
}

function renderBarChart(rows) {
  const width = 760;
  const height = 280;
  const padding = { left: 52, right: 24, top: 24, bottom: 66 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;
  const groupW = chartW / rows.length;
  const metrics = [
    ["precision", colors.blue, "P"],
    ["recall", colors.green, "R"],
    ["f1", colors.amber, "F1"]
  ];

  const bars = rows.map((row, rowIndex) => {
    const x0 = padding.left + rowIndex * groupW;
    return metrics.map(([key, color], metricIndex) => {
      const barW = Math.max(12, groupW / 5);
      const gap = 5;
      const x = x0 + groupW / 2 - (barW * 1.5 + gap) + metricIndex * (barW + gap);
      const barH = Number(row[key]) * chartH;
      const y = padding.top + chartH - barH;
      return `<rect x="${x}" y="${y}" width="${barW}" height="${barH}" rx="4" fill="${color}"></rect>`;
    }).join("");
  }).join("");

  const labels = rows.map((row, index) => {
    const x = padding.left + index * groupW + groupW / 2;
    const label = row.name.replace("Centralized Weighted Logistic", "Centralized").replace("FedAvg ", "");
    return `<text class="chart-label" x="${x}" y="${height - 34}" text-anchor="middle">${label}</text>`;
  }).join("");

  const grid = [0, 0.25, 0.5, 0.75, 1].map((value) => {
    const y = padding.top + chartH - value * chartH;
    return `
      <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="${colors.line}"></line>
      <text class="axis-label" x="12" y="${y + 4}">${(value * 100).toFixed(0)}%</text>
    `;
  }).join("");

  document.querySelector("#result-bars").innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Precision, recall, and F1 comparison">
      ${grid}
      ${bars}
      ${labels}
      <g transform="translate(${padding.left}, ${height - 14})">
        <rect width="10" height="10" rx="2" fill="${colors.blue}"></rect><text class="axis-label" x="16" y="10">Precision</text>
        <rect x="96" width="10" height="10" rx="2" fill="${colors.green}"></rect><text class="axis-label" x="112" y="10">Recall</text>
        <rect x="170" width="10" height="10" rx="2" fill="${colors.amber}"></rect><text class="axis-label" x="186" y="10">F1</text>
      </g>
    </svg>
  `;
}

function renderLineChart(history) {
  const width = 760;
  const height = 280;
  const padding = { left: 52, right: 28, top: 22, bottom: 42 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;
  const strategies = ["iid", "moderate_non_iid", "highly_non_iid"];
  const strategyColors = [colors.blue, colors.green, colors.violet];
  const maxRound = Math.max(...history.map((row) => Number(row.round)));
  const maxF1 = Math.max(...history.map((row) => Number(row.val_f1))) * 1.08;

  const pathFor = (strategy) => {
    const points = history
      .filter((row) => row.strategy === strategy)
      .map((row) => {
        const x = padding.left + ((Number(row.round) - 1) / (maxRound - 1)) * chartW;
        const y = padding.top + chartH - (Number(row.val_f1) / maxF1) * chartH;
        return `${x.toFixed(2)},${y.toFixed(2)}`;
      });
    return points.join(" ");
  };

  const grid = [0, 0.1, 0.2, 0.3].map((value) => {
    const y = padding.top + chartH - (value / maxF1) * chartH;
    return `
      <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="${colors.line}"></line>
      <text class="axis-label" x="16" y="${y + 4}">${value.toFixed(1)}</text>
    `;
  }).join("");

  const lines = strategies.map((strategy, index) => `
    <polyline points="${pathFor(strategy)}" fill="none" stroke="${strategyColors[index]}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
  `).join("");

  const legend = strategies.map((strategy, index) => `
    <g transform="translate(${padding.left + index * 180}, ${height - 16})">
      <line x1="0" y1="0" x2="22" y2="0" stroke="${strategyColors[index]}" stroke-width="3"></line>
      <text class="axis-label" x="30" y="4">${strategy}</text>
    </g>
  `).join("");

  document.querySelector("#convergence-chart").innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="FedAvg validation F1 convergence trend">
      ${grid}
      <line x1="${padding.left}" y1="${padding.top + chartH}" x2="${width - padding.right}" y2="${padding.top + chartH}" stroke="${colors.line}"></line>
      ${lines}
      <text class="axis-label" x="${width - padding.right}" y="${padding.top + chartH + 26}" text-anchor="end">Round 50</text>
      ${legend}
    </svg>
  `;
}

function renderFactoryView(strategy) {
  currentStrategy = strategy;
  document.querySelector("#factory-subtitle").textContent = strategy;

  document.querySelectorAll(".segmented").forEach((button) => {
    button.classList.toggle("active", button.dataset.strategy === strategy);
  });

  const rows = factoryRows.filter((row) => row.strategy === strategy);
  const width = 640;
  const height = 280;
  const padding = { left: 52, right: 28, top: 24, bottom: 48 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;
  const maxRate = Math.max(...rows.map((row) => Number(row.failure_rate))) * 1.25;
  const barW = chartW / rows.length - 18;

  const bars = rows.map((row, index) => {
    const x = padding.left + index * (chartW / rows.length) + 9;
    const barH = Number(row.failure_rate) / maxRate * chartH;
    const y = padding.top + chartH - barH;
    return `
      <rect x="${x}" y="${y}" width="${barW}" height="${barH}" rx="5" fill="${index % 2 ? colors.blue : colors.teal}"></rect>
      <text class="chart-label" x="${x + barW / 2}" y="${height - 22}" text-anchor="middle">${row.factory.replace("factory_", "F")}</text>
      <text class="chart-label" x="${x + barW / 2}" y="${y - 7}" text-anchor="middle">${pct(row.failure_rate, 2)}</text>
    `;
  }).join("");

  document.querySelector("#factory-chart").innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${strategy} factory failure rate">
      <line x1="${padding.left}" y1="${padding.top + chartH}" x2="${width - padding.right}" y2="${padding.top + chartH}" stroke="${colors.line}"></line>
      ${bars}
    </svg>
  `;

  document.querySelector("#factory-table").innerHTML = rows.map((row) => `
    <div class="factory-row">
      <strong>${row.factory}</strong>
      <span><span class="muted">Rows</span> ${Number(row.rows).toLocaleString("en-US")}</span>
      <span><span class="muted">Failure</span> ${pct(row.failure_rate, 2)}</span>
      <span><span class="muted">Mode</span> ${dominantMode(row)}</span>
    </div>
  `).join("");
}

function bindControls() {
  document.querySelectorAll(".segmented").forEach((button) => {
    button.addEventListener("click", () => renderFactoryView(button.dataset.strategy));
  });

  const sections = [...document.querySelectorAll("main section[id]")];
  const navLinks = [...document.querySelectorAll(".nav a")];
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navLinks.forEach((link) => {
          link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`);
        });
      }
    });
  }, { rootMargin: "-35% 0px -55% 0px" });

  sections.forEach((section) => observer.observe(section));
}

async function init() {
  try {
    const [eda, centralized, fedavg, history, factories, centralizedModel, fedavgModels, standardizationRows] = await Promise.all([
      fetch(paths.eda).then((response) => response.json()),
      loadCSV(paths.centralized),
      loadCSV(paths.fedavg),
      loadCSV(paths.fedavgHistory),
      loadCSV(paths.factories),
      fetch(paths.centralizedModel).then((response) => response.json()),
      fetch(paths.fedavgModels).then((response) => response.json()),
      loadCSV(paths.standardization)
    ]);

    factoryRows = factories;
    standardization = buildStandardization(standardizationRows);
    predictionModels = buildPredictionModels(centralizedModel, fedavgModels);
    const resultRows = buildResultRows(centralized, fedavg);
    updateOverview(eda, fedavg);
    renderResultsTable(resultRows);
    renderBarChart(resultRows);
    renderLineChart(history);
    renderFactoryView(currentStrategy);
    bindControls();
    bindPredictionControls();
  } catch (error) {
    showError(`Unable to read local report files: ${error.message}. Please access the page through the local HTTP server instead of opening the HTML file directly.`);
  }
}

init();
