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

const apiBase = window.localStorage.getItem("iotFlApiBase") || "http://127.0.0.1:8000";
const tokenStorageKey = "iotFlAccessToken";

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
let accessToken = window.localStorage.getItem(tokenStorageKey) || "";
let currentUser = null;
let algorithmOptions = [];
let experiments = [];
let uploadedDatasets = [];
let dashboardData = null;
let comparisonResult = null;
let selectedExperimentId = null;
let experimentBusy = false;

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

function showAuthMessage(message, type = "") {
  const target = document.querySelector("#auth-message");
  if (!target) {
    return;
  }
  target.textContent = message;
  target.className = `auth-message ${type}`.trim();
}

function apiErrorMessage(payload, fallback) {
  if (Array.isArray(payload.detail)) {
    return payload.detail
      .map((item) => {
        const location = Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "";
        const prefix = location ? `${location}: ` : "";
        return `${prefix}${item.msg}`;
      })
      .join(" ");
  }
  if (typeof payload.detail === "string") {
    return payload.detail;
  }
  if (typeof payload.message === "string") {
    return payload.message;
  }
  return fallback;
}

async function apiRequest(path, options = {}) {
  const isFormData = options.body instanceof FormData;
  const headers = {
    ...(isFormData ? {} : { "Content-Type": "application/json" }),
    ...(options.headers || {})
  };
  if (accessToken) {
    headers.Authorization = `Bearer ${accessToken}`;
  }

  const response = await fetch(`${apiBase}${path}`, {
    ...options,
    headers
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      clearSession();
    }
    throw new Error(apiErrorMessage(payload, `Request failed with status ${response.status}`));
  }
  return payload;
}

function saveSession(token, user) {
  accessToken = token;
  currentUser = user;
  window.localStorage.setItem(tokenStorageKey, token);
  renderSession(user);
  loadExperimentWorkspace();
}

function clearSession() {
  accessToken = "";
  currentUser = null;
  window.localStorage.removeItem(tokenStorageKey);
  document.querySelector("#auth-status").textContent = "Not connected";
  document.querySelector("#session-box").innerHTML = `<div class="empty-state">Login or register to view the current user.</div>`;
  renderExperimentsLoggedOut();
  renderDashboardLoggedOut();
  renderCompareLoggedOut();
}

function renderSession(user) {
  document.querySelector("#auth-status").textContent = `${user.role} session`;
  document.querySelector("#session-box").innerHTML = `
    <div class="session-line"><span>User</span><strong>${user.username}</strong></div>
    <div class="session-line"><span>Email</span><strong>${user.email}</strong></div>
    <div class="session-line"><span>Role</span><strong>${user.role}</strong></div>
    <div class="session-line"><span>Factory</span><strong>${user.factory_id ?? "None"}</strong></div>
    <div class="session-line"><span>Status</span><strong>${user.is_active ? "Active" : "Inactive"}</strong></div>
  `;
}

function readRegisterPayload() {
  const role = document.querySelector("#register-role").value;
  const rawFactoryId = document.querySelector("#register-factory-id").value;
  return {
    username: document.querySelector("#register-username").value.trim(),
    email: document.querySelector("#register-email").value.trim(),
    password: document.querySelector("#register-password").value,
    role,
    factory_id: role === "client" ? Number(rawFactoryId) : null
  };
}

function bindAuthControls() {
  const loginForm = document.querySelector("#login-form");
  const registerForm = document.querySelector("#register-form");
  const meButton = document.querySelector("#me-button");
  const logoutButton = document.querySelector("#logout-button");
  const adminButton = document.querySelector("#admin-check-button");
  const roleSelect = document.querySelector("#register-role");
  const factoryInput = document.querySelector("#register-factory-id");

  if (!loginForm || !registerForm) {
    return;
  }

  roleSelect.addEventListener("change", () => {
    const isClient = roleSelect.value === "client";
    factoryInput.disabled = !isClient;
    factoryInput.required = isClient;
    if (!isClient) {
      factoryInput.value = "";
    } else if (!factoryInput.value) {
      factoryInput.value = "1";
    }
  });

  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const payload = {
        username: document.querySelector("#login-username").value.trim(),
        password: document.querySelector("#login-password").value
      };
      const result = await apiRequest("/api/auth/login", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      saveSession(result.access_token, result.user);
      showAuthMessage("Login successful.", "success");
    } catch (error) {
      showAuthMessage(error.message, "error");
    }
  });

  registerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const result = await apiRequest("/api/auth/register", {
        method: "POST",
        body: JSON.stringify(readRegisterPayload())
      });
      saveSession(result.access_token, result.user);
      showAuthMessage("Account created and logged in.", "success");
    } catch (error) {
      showAuthMessage(error.message, "error");
    }
  });

  meButton.addEventListener("click", async () => {
    try {
      const user = await apiRequest("/api/auth/me");
      renderSession(user);
      showAuthMessage("Current session loaded.", "success");
    } catch (error) {
      showAuthMessage(error.message, "error");
    }
  });

  adminButton.addEventListener("click", async () => {
    try {
      const users = await apiRequest("/api/admin/users");
      showAuthMessage(`Admin check passed. Visible users: ${users.length}.`, "success");
    } catch (error) {
      showAuthMessage(error.message, "error");
    }
  });

  logoutButton.addEventListener("click", async () => {
    try {
      if (accessToken) {
        await apiRequest("/api/auth/logout", { method: "POST" });
      }
      clearSession();
      showAuthMessage("Logged out.", "success");
    } catch (error) {
      clearSession();
      showAuthMessage(error.message, "error");
    }





  });

  if (accessToken) {
    apiRequest("/api/auth/me")
      .then((user) => {
        currentUser = user;
        renderSession(user);
        loadExperimentWorkspace();
      })
      .catch(() => clearSession());
  } else {
    renderExperimentsLoggedOut();
    renderDashboardLoggedOut();
    renderCompareLoggedOut();
  }
}

function distributionLabel(value) {
  const labels = {
    iid: "IID",
    moderate_non_iid: "Moderate Non-IID",
    highly_non_iid: "Highly Non-IID"
  };
  return labels[value] || value;
}

function statusClass(status) {
  return `status-pill ${String(status || "PENDING").toLowerCase()}`;
}

function formatMetric(value, digits = 3) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return Number(value).toFixed(digits);
}

function formatTime(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  return `${Number(value).toFixed(2)}s`;
}

function formatDateTime(value) {
  if (!value) {
    return "-";
  }
  return new Date(value).toLocaleString();
}

function setExperimentMessage(message, type = "") {
  const target = document.querySelector("#experiment-message");
  if (!target) {
    return;
  }
  target.textContent = message;
  target.className = `experiment-message ${type}`.trim();
}

function setDatasetMessage(message, type = "") {
  const target = document.querySelector("#dataset-message");
  if (!target) {
    return;
  }
  target.textContent = message;
  target.className = `experiment-message ${type}`.trim();
}

function setComparisonMessage(message, type = "") {
  const target = document.querySelector("#comparison-message");
  if (!target) {
    return;
  }
  target.textContent = message;
  target.className = `experiment-message ${type}`.trim();
}

function renderExperimentRows(targetId, rows, emptyText) {
  const body = document.querySelector(targetId);
  if (!body) {
    return;
  }
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="8">${emptyText}</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((experiment) => `
    <tr>
      <td>#${experiment.id}</td>
      <td>${labelForAlgorithm(experiment.algorithm)}</td>
      <td>${distributionLabel(experiment.distribution)}</td>
      <td><span class="${statusClass(experiment.status)}">${experiment.status}</span></td>
      <td>${formatMetric(experiment.recall)}</td>
      <td><strong>${formatMetric(experiment.f1_score)}</strong></td>
      <td>${formatTime(experiment.training_time)}</td>
      <td>${formatDateTime(experiment.created_at)}</td>
    </tr>
  `).join("");
}

function setExperimentBusy(isBusy) {
  experimentBusy = isBusy;
  const startButton = document.querySelector("#start-experiment-button");
  const refreshButton = document.querySelector("#refresh-experiments-button");
  const uploadButton = document.querySelector("#upload-dataset-button");
  if (startButton) {
    startButton.disabled = isBusy || !accessToken;
  }
  if (refreshButton) {
    refreshButton.disabled = isBusy || !accessToken;
  }
  if (uploadButton) {
    uploadButton.disabled = isBusy || !accessToken;
  }
}

function renderExperimentsLoggedOut() {
  algorithmOptions = [];
  experiments = [];
  uploadedDatasets = [];
  dashboardData = null;
  comparisonResult = null;
  selectedExperimentId = null;
  const notice = document.querySelector("#experiment-auth-notice");
  const algorithmSelect = document.querySelector("#experiment-algorithm");
  const formStatus = document.querySelector("#experiment-form-status");
  const history = document.querySelector("#experiment-history-table");
  const datasetSelect = document.querySelector("#experiment-dataset");
  const datasetList = document.querySelector("#dataset-list");
  const datasetSubtitle = document.querySelector("#dataset-list-subtitle");
  const statusGrid = document.querySelector("#experiment-status-grid");
  const metricsGrid = document.querySelector("#experiment-metrics-grid");
  const chart = document.querySelector("#experiment-convergence-chart");

  if (notice) {
    notice.textContent = "Login to create experiments and view your experiment history.";
    notice.className = "experiment-notice";
  }
  if (algorithmSelect) {
    algorithmSelect.innerHTML = `<option value="">Login to load algorithms</option>`;
    algorithmSelect.disabled = true;
  }
  if (formStatus) {
    formStatus.textContent = "Requires login";
  }
  if (history) {
    history.innerHTML = `<tr><td colspan="8">Login to load experiment history.</td></tr>`;
  }
  if (datasetSelect) {
    datasetSelect.innerHTML = `<option value="">Default project dataset</option>`;
    datasetSelect.disabled = true;
  }
  if (datasetList) {
    datasetList.innerHTML = `<div class="empty-state">Login to view uploaded datasets.</div>`;
  }
  if (datasetSubtitle) {
    datasetSubtitle.textContent = "Default dataset is always available";
  }
  if (statusGrid) {
    statusGrid.innerHTML = `<div class="empty-state">Login to create experiments and view status.</div>`;
  }
  if (metricsGrid) {
    metricsGrid.innerHTML = `<div class="empty-state">Completed experiment metrics will appear here.</div>`;
  }
  if (chart) {
    chart.innerHTML = `<div class="empty-state">No convergence history loaded.</div>`;
  }
  setExperimentMessage("");
  setDatasetMessage("");
  setExperimentBusy(false);
}

function renderDashboardLoggedOut() {
  dashboardData = null;
  const notice = document.querySelector("#dashboard-auth-notice");
  const grid = document.querySelector("#dashboard-grid");
  if (notice) {
    notice.textContent = "Login to load the role-based dashboard.";
    notice.className = "experiment-notice";
  }
  if (grid) {
    grid.innerHTML = `<article class="card"><div class="empty-state">Dashboard data will appear after login.</div></article>`;
  }
  renderExperimentRows("#dashboard-recent-table", [], "Login to load recent experiments.");
}

function metricCard(label, value, detail = "") {
  return `
    <article class="dashboard-card">
      <span>${label}</span>
      <strong>${value}</strong>
      ${detail ? `<small>${detail}</small>` : ""}
    </article>
  `;
}

function renderClientDashboard(data) {
  const grid = document.querySelector("#dashboard-grid");
  if (!grid) {
    return;
  }
  const distributions = data.distribution_types.length
    ? data.distribution_types.map(distributionLabel).join(", ")
    : "No factory clients";
  const performance = data.recent_model_performance.length
    ? data.recent_model_performance.map((item) => `
        <div class="dashboard-list-row">
          <strong>#${item.experiment_id} ${labelForAlgorithm(item.algorithm)}</strong>
          <span>${distributionLabel(item.distribution)} / Recall ${formatMetric(item.recall)} / F1 ${formatMetric(item.f1_score)} / ${formatTime(item.training_time)}</span>
        </div>
      `).join("")
    : `<div class="empty-state">No completed experiment metrics yet.</div>`;

  grid.innerHTML = `
    ${metricCard("Factory ID", `#${data.factory_id}`, data.factory_name)}
    ${metricCard("Samples", Number(data.total_samples).toLocaleString("en-US"), `${data.failure_samples} failure samples`)}
    ${metricCard("Failure Rate", `${(Number(data.failure_rate) * 100).toFixed(2)}%`, data.dominant_failure_mode ? `dominant ${data.dominant_failure_mode}` : "no dominant failure mode")}
    ${metricCard("Distribution", distributions, `${data.clients.length} factory client records`)}
    <article class="card dashboard-wide">
      <div class="card-head">
        <h3>Factory Clients</h3>
        <span>samples / failure / mode</span>
      </div>
      <div class="dashboard-list">
        ${data.clients.length ? data.clients.map((client) => `
          <div class="dashboard-list-row">
            <strong>${client.name}</strong>
            <span>${distributionLabel(client.distribution_type)} / ${client.total_rows.toLocaleString("en-US")} samples / ${(client.failure_ratio * 100).toFixed(2)}% failure</span>
          </div>
        `).join("") : `<div class="empty-state">No factory client records found.</div>`}
      </div>
    </article>
    <article class="card dashboard-wide">
      <div class="card-head">
        <h3>Recent Model Performance</h3>
        <span>completed runs only</span>
      </div>
      <div class="dashboard-list">${performance}</div>
    </article>
  `;
  renderExperimentRows("#dashboard-recent-table", data.recent_experiments, "No recent experiments yet.");
}

function renderAdminDashboard(data) {
  const grid = document.querySelector("#dashboard-grid");
  if (!grid) {
    return;
  }
  const usageRows = Object.entries(data.algorithm_usage).length
    ? Object.entries(data.algorithm_usage).map(([algorithm, count]) => `
      <div class="dashboard-list-row">
        <strong>${labelForAlgorithm(algorithm)}</strong>
        <span>${count} experiment${count === 1 ? "" : "s"}</span>
      </div>
    `).join("")
    : `<div class="empty-state">No algorithm usage yet.</div>`;
  const statusRows = Object.entries(data.status_counts).length
    ? Object.entries(data.status_counts).map(([status, count]) => `
      <div class="dashboard-list-row">
        <strong><span class="${statusClass(status)}">${status}</span></strong>
        <span>${count} experiment${count === 1 ? "" : "s"}</span>
      </div>
    `).join("")
    : `<div class="empty-state">No experiments yet.</div>`;

  grid.innerHTML = `
    ${metricCard("Registered Users", data.registered_users)}
    ${metricCard("Factory Clients", data.factory_clients)}
    ${metricCard("Factory IDs", data.factory_ids.length ? data.factory_ids.map((id) => `#${id}`).join(", ") : "-")}
    ${metricCard("Experiments", data.experiment_count)}
    <article class="card dashboard-wide">
      <div class="card-head">
        <h3>Algorithm Usage</h3>
        <span>stored experiments</span>
      </div>
      <div class="dashboard-list">${usageRows}</div>
    </article>
    <article class="card dashboard-wide">
      <div class="card-head">
        <h3>Experiment Status</h3>
        <span>all users</span>
      </div>
      <div class="dashboard-list">${statusRows}</div>
    </article>
  `;
  renderExperimentRows("#dashboard-recent-table", data.recent_experiments, "No recent experiments yet.");
}

async function loadDashboard() {
  if (!accessToken) {
    renderDashboardLoggedOut();
    return;
  }
  const notice = document.querySelector("#dashboard-auth-notice");
  if (notice) {
    notice.textContent = "Loading dashboard...";
    notice.className = "experiment-notice loading";
  }
  const endpoint = currentUser?.role === "admin"
    ? "/api/dashboard/admin"
    : "/api/dashboard/client";
  dashboardData = await apiRequest(endpoint);
  if (dashboardData.role === "admin") {
    renderAdminDashboard(dashboardData);
  } else {
    renderClientDashboard(dashboardData);
  }
  if (notice) {
    notice.textContent = currentUser
      ? `${currentUser.role} dashboard loaded for ${currentUser.username}.`
      : "Dashboard loaded.";
    notice.className = "experiment-notice success";
  }
}

function populateAlgorithmSelect() {
  const select = document.querySelector("#experiment-algorithm");
  if (!select) {
    return;
  }
  if (!algorithmOptions.length) {
    select.innerHTML = `<option value="">No algorithms available</option>`;
    select.disabled = true;
    return;
  }
  select.innerHTML = algorithmOptions.map((algorithm) => `
    <option value="${algorithm.name}">${algorithm.display_name}</option>
  `).join("");
  select.disabled = false;
  updateAlgorithmParameters();
}

function populateDatasetSelect() {
  const select = document.querySelector("#experiment-dataset");
  if (!select) {
    return;
  }
  const options = [
    `<option value="">Default project dataset</option>`,
    ...uploadedDatasets
      .filter((dataset) => dataset.status === "READY")
      .map((dataset) => `
        <option value="${dataset.id}">#${dataset.id} ${dataset.original_filename} (${dataset.rows} rows)</option>
      `)
  ];
  select.innerHTML = options.join("");
  select.disabled = !accessToken;
}

function renderDatasetList() {
  const target = document.querySelector("#dataset-list");
  const subtitle = document.querySelector("#dataset-list-subtitle");
  if (!target || !subtitle) {
    return;
  }
  subtitle.textContent = `${uploadedDatasets.length} uploaded dataset${uploadedDatasets.length === 1 ? "" : "s"}`;
  if (!uploadedDatasets.length) {
    target.innerHTML = `<div class="empty-state">No uploaded datasets yet. Experiments will use the default project dataset.</div>`;
    return;
  }
  target.innerHTML = uploadedDatasets.map((dataset) => `
    <button class="dataset-item" type="button" data-dataset-id="${dataset.id}">
      <div>
        <strong>#${dataset.id} ${dataset.original_filename}</strong>
        <span>${dataset.rows.toLocaleString("en-US")} rows / ${dataset.columns} columns / ${formatDateTime(dataset.created_at)}</span>
        ${dataset.error_message ? `<small>${dataset.error_message}</small>` : ""}
      </div>
      <span class="${statusClass(dataset.status)}">${dataset.status}</span>
    </button>
  `).join("");
  target.querySelectorAll(".dataset-item").forEach((item) => {
    item.addEventListener("click", () => {
      const select = document.querySelector("#experiment-dataset");
      if (select) {
        select.value = item.dataset.datasetId;
      }
      setDatasetMessage(`Selected dataset #${item.dataset.datasetId} for the next experiment.`, "success");
    });
  });
}

function renderExperimentHistory() {
  const body = document.querySelector("#experiment-history-table");
  if (!body) {
    return;
  }
  if (!experiments.length) {
    body.innerHTML = `<tr><td colspan="8">No experiments yet. Create one above.</td></tr>`;
    return;
  }
  body.innerHTML = experiments.map((experiment) => `
    <tr class="experiment-row ${experiment.id === selectedExperimentId ? "selected" : ""}" data-experiment-id="${experiment.id}" tabindex="0">
      <td>#${experiment.id}</td>
      <td>${labelForAlgorithm(experiment.algorithm)}</td>
      <td>${distributionLabel(experiment.distribution)}</td>
      <td><span class="${statusClass(experiment.status)}">${experiment.status}</span></td>
      <td>${formatMetric(experiment.recall)}</td>
      <td><strong>${formatMetric(experiment.f1_score)}</strong></td>
      <td>${formatTime(experiment.training_time)}</td>
      <td>${formatDateTime(experiment.created_at)}</td>
    </tr>
  `).join("");

  body.querySelectorAll(".experiment-row").forEach((row) => {
    row.addEventListener("click", () => selectExperiment(Number(row.dataset.experimentId)));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectExperiment(Number(row.dataset.experimentId));
      }
    });
  });
}

function labelForAlgorithm(name) {
  const match = algorithmOptions.find((algorithm) => algorithm.name === name);
  return match ? match.display_name : name;
}

function renderExperimentStatus(experiment) {
  selectedExperimentId = experiment?.id || null;
  const subtitle = document.querySelector("#experiment-detail-subtitle");
  const grid = document.querySelector("#experiment-status-grid");
  if (!subtitle || !grid) {
    return;
  }
  if (!experiment) {
    subtitle.textContent = "No experiment selected";
    grid.innerHTML = `<div class="empty-state">Create an experiment or select one from history.</div>`;
    return;
  }
  subtitle.textContent = `#${experiment.id} ${experiment.status}`;
  grid.innerHTML = `
    <div><span>Experiment ID</span><strong>#${experiment.id}</strong></div>
    <div><span>Algorithm</span><strong>${labelForAlgorithm(experiment.algorithm)}</strong></div>
    <div><span>Distribution</span><strong>${distributionLabel(experiment.distribution)}</strong></div>
    <div><span>Status</span><strong><span class="${statusClass(experiment.status)}">${experiment.status}</span></strong></div>
    <div><span>Created</span><strong>${formatDateTime(experiment.created_at)}</strong></div>
    <div><span>Rounds</span><strong>${experiment.rounds}</strong></div>
    <div><span>Local epochs</span><strong>${experiment.local_epochs}</strong></div>
    <div><span>Learning rate</span><strong>${experiment.learning_rate}</strong></div>
  `;
  renderExperimentHistory();
}

function renderExperimentMetrics(result) {
  const grid = document.querySelector("#experiment-metrics-grid");
  const subtitle = document.querySelector("#experiment-metrics-subtitle");
  if (!grid || !subtitle) {
    return;
  }
  if (!result || result.status !== "COMPLETED") {
    subtitle.textContent = result?.status === "FAILED" ? "Experiment failed" : "From stored backend result";
    const message = result?.status === "FAILED"
      ? result.error_message || "Experiment failed without a stored error message."
      : "Completed experiment metrics will appear here.";
    grid.innerHTML = `<div class="${result?.status === "FAILED" ? "load-error" : "empty-state"}">${message}</div>`;
    return;
  }
  subtitle.textContent = `Experiment #${result.id}`;
  grid.innerHTML = `
    <div><span>Accuracy</span><strong>${formatMetric(result.accuracy)}</strong></div>
    <div><span>Precision</span><strong>${formatMetric(result.precision)}</strong></div>
    <div><span>Recall</span><strong>${formatMetric(result.recall)}</strong></div>
    <div><span>F1</span><strong>${formatMetric(result.f1_score)}</strong></div>
    <div><span>Communication Cost</span><strong>${formatMetric(result.communication_cost, 0)}</strong></div>
    <div><span>Training Time</span><strong>${formatTime(result.training_time)}</strong></div>
  `;
}

function metricFromHistory(row) {
  if (row.val_f1 !== undefined) {
    return { key: "val_f1", label: "Validation F1", value: Number(row.val_f1) };
  }
  if (row.val_f1_at_0_5 !== undefined) {
    return { key: "val_f1_at_0_5", label: "Validation F1", value: Number(row.val_f1_at_0_5) };
  }
  if (row.validation_recall_at_0_5 !== undefined) {
    return { key: "validation_recall_at_0_5", label: "Validation Recall", value: Number(row.validation_recall_at_0_5) };
  }
  if (row.val_recall_at_0_5 !== undefined) {
    return { key: "val_recall_at_0_5", label: "Validation Recall", value: Number(row.val_recall_at_0_5) };
  }
  if (row.mean_client_loss !== undefined) {
    return { key: "mean_client_loss", label: "Mean Client Loss", value: Number(row.mean_client_loss) };
  }
  return null;
}

function renderExperimentConvergenceChart(history) {
  const target = document.querySelector("#experiment-convergence-chart");
  if (!target) {
    return;
  }
  if (!Array.isArray(history) || !history.length) {
    target.innerHTML = `<div class="empty-state">No convergence history loaded.</div>`;
    return;
  }

  const points = history
    .map((row) => ({ row, metric: metricFromHistory(row) }))
    .filter((item) => item.metric && Number.isFinite(item.metric.value))
    .map((item) => ({
      round: Number(item.row.round),
      value: item.metric.value,
      label: item.metric.label
    }));

  if (!points.length) {
    target.innerHTML = `<div class="empty-state">Convergence history does not include a plottable metric.</div>`;
    return;
  }

  const width = 760;
  const height = 280;
  const padding = { left: 54, right: 28, top: 24, bottom: 48 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;
  const minRound = Math.min(...points.map((point) => point.round));
  const maxRound = Math.max(...points.map((point) => point.round));
  const maxValue = Math.max(...points.map((point) => point.value), 0.01) * 1.08;
  const minValue = Math.min(...points.map((point) => point.value), 0);
  const spanRound = Math.max(maxRound - minRound, 1);
  const spanValue = Math.max(maxValue - minValue, 0.001);
  const label = points[0].label;
  const polyline = points.map((point) => {
    const x = padding.left + ((point.round - minRound) / spanRound) * chartW;
    const y = padding.top + chartH - ((point.value - minValue) / spanValue) * chartH;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const value = minValue + ratio * spanValue;
    const y = padding.top + chartH - ratio * chartH;
    return `
      <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="${colors.line}"></line>
      <text class="axis-label" x="12" y="${y + 4}">${value.toFixed(2)}</text>
    `;
  }).join("");

  target.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label} convergence chart">
      ${grid}
      <line x1="${padding.left}" y1="${padding.top + chartH}" x2="${width - padding.right}" y2="${padding.top + chartH}" stroke="${colors.line}"></line>
      <polyline points="${polyline}" fill="none" stroke="${colors.teal}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
      ${points.map((point) => {
        const x = padding.left + ((point.round - minRound) / spanRound) * chartW;
        const y = padding.top + chartH - ((point.value - minValue) / spanValue) * chartH;
        return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="3.5" fill="${colors.teal}"></circle>`;
      }).join("")}
      <text class="axis-label" x="${padding.left}" y="${height - 16}">Round ${minRound}</text>
      <text class="axis-label" x="${width - padding.right}" y="${height - 16}" text-anchor="end">Round ${maxRound}</text>
      <text class="axis-label" x="${padding.left}" y="14">${label}</text>
    </svg>
  `;
}

function renderCompareLoggedOut() {
  comparisonResult = null;
  const notice = document.querySelector("#comparison-notice");
  const options = document.querySelector("#comparison-options");
  const count = document.querySelector("#comparison-count");
  const button = document.querySelector("#comparison-button");
  const metricsChart = document.querySelector("#comparison-metrics-chart");
  const convergenceChart = document.querySelector("#comparison-convergence-chart");
  const table = document.querySelector("#comparison-table");
  if (notice) {
    notice.textContent = "Login and complete experiments before comparing results.";
    notice.className = "experiment-notice";
  }
  if (options) {
    options.innerHTML = `<div class="empty-state">No completed experiments available.</div>`;
  }
  if (count) {
    count.textContent = "0 available";
  }
  if (button) {
    button.disabled = true;
  }
  if (metricsChart) {
    metricsChart.innerHTML = `<div class="empty-state">Select completed experiments to compare.</div>`;
  }
  if (convergenceChart) {
    convergenceChart.innerHTML = `<div class="empty-state">No convergence history selected.</div>`;
  }
  if (table) {
    table.innerHTML = `<tr><td colspan="13">Select completed experiments to compare.</td></tr>`;
  }
  setComparisonMessage("");
}

function renderCompareOptions() {
  const options = document.querySelector("#comparison-options");
  const count = document.querySelector("#comparison-count");
  const button = document.querySelector("#comparison-button");
  const notice = document.querySelector("#comparison-notice");
  if (!options || !count || !button) {
    return;
  }
  const completed = experiments.filter((experiment) => experiment.status === "COMPLETED");
  count.textContent = `${completed.length} available`;
  button.disabled = !accessToken || !completed.length;
  if (notice && accessToken) {
    notice.textContent = completed.length
      ? "Choose completed experiments, then compare stored backend results."
      : "No completed experiments yet. Run an experiment to enable comparison.";
    notice.className = completed.length ? "experiment-notice success" : "experiment-notice";
  }
  if (!completed.length) {
    options.innerHTML = `<div class="empty-state">No completed experiments available.</div>`;
    return;
  }
  options.innerHTML = completed.map((experiment, index) => `
    <label class="comparison-option">
      <input type="checkbox" value="${experiment.id}" ${index < 3 ? "checked" : ""}>
      <span>
        <strong>#${experiment.id} ${labelForAlgorithm(experiment.algorithm)}</strong>
        <small>${distributionLabel(experiment.distribution)} / Recall ${formatMetric(experiment.recall)} / F1 ${formatMetric(experiment.f1_score)} / ${formatTime(experiment.training_time)}</small>
      </span>
    </label>
  `).join("");
}

function selectedComparisonIds() {
  return [...document.querySelectorAll("#comparison-options input:checked")]
    .map((input) => Number(input.value))
    .filter((id) => Number.isFinite(id));
}

async function compareSelectedExperiments() {
  if (!accessToken) {
    setComparisonMessage("Please login before comparing experiments.", "error");
    return;
  }
  const experimentIds = selectedComparisonIds();
  if (!experimentIds.length) {
    setComparisonMessage("Select at least one completed experiment.", "error");
    return;
  }
  setComparisonMessage("Loading comparison...", "loading");
  try {
    comparisonResult = await apiRequest("/api/experiments/compare", {
      method: "POST",
      body: JSON.stringify({ experiment_ids: experimentIds })
    });
    renderComparisonResult(comparisonResult);
    setComparisonMessage("Comparison loaded.", "success");
  } catch (error) {
    setComparisonMessage(error.message, "error");
  }
}

function renderComparisonResult(result) {
  renderComparisonTable(result.experiments);
  renderComparisonMetricChart(result.experiments);
  renderComparisonConvergenceChart(result.convergence);
}

function renderComparisonTable(rows) {
  const body = document.querySelector("#comparison-table");
  if (!body) {
    return;
  }
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="13">Select completed experiments to compare.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((row) => `
    <tr>
      <td>#${row.id}</td>
      <td>${labelForAlgorithm(row.algorithm)}</td>
      <td>${distributionLabel(row.distribution)}</td>
      <td>${row.rounds}</td>
      <td>${row.local_epochs}</td>
      <td>${row.learning_rate}</td>
      <td>${formatMetric(row.accuracy)}</td>
      <td>${formatMetric(row.precision)}</td>
      <td>${formatMetric(row.recall)}</td>
      <td><strong>${formatMetric(row.f1_score)}</strong></td>
      <td>${formatMetric(row.communication_cost, 0)}</td>
      <td>${formatTime(row.training_time)}</td>
      <td>${formatDateTime(row.created_at)}</td>
    </tr>
  `).join("");
}

function renderComparisonMetricChart(rows) {
  const target = document.querySelector("#comparison-metrics-chart");
  if (!target) {
    return;
  }
  if (!rows.length) {
    target.innerHTML = `<div class="empty-state">Select completed experiments to compare.</div>`;
    return;
  }
  const width = 760;
  const height = 300;
  const padding = { left: 54, right: 24, top: 24, bottom: 76 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;
  const groupW = chartW / rows.length;
  const maxCost = Math.max(...rows.map((row) => Number(row.communication_cost) || 0), 1);
  const metrics = [
    ["recall", colors.green, "Recall", 1],
    ["f1_score", colors.amber, "F1", 1],
    ["communication_cost", colors.violet, "Comm", maxCost]
  ];
  const bars = rows.map((row, rowIndex) => {
    const x0 = padding.left + rowIndex * groupW;
    return metrics.map(([key, color, label, max], metricIndex) => {
      const barW = Math.max(12, groupW / 5.2);
      const gap = 5;
      const value = Number(row[key]) || 0;
      const ratio = Math.min(value / Number(max), 1);
      const barH = ratio * chartH;
      const x = x0 + groupW / 2 - (barW * 1.5 + gap) + metricIndex * (barW + gap);
      const y = padding.top + chartH - barH;
      const textValue = key === "communication_cost" ? value.toFixed(0) : value.toFixed(2);
      return `
        <rect x="${x}" y="${y}" width="${barW}" height="${barH}" rx="4" fill="${color}"></rect>
        <text class="chart-label" x="${x + barW / 2}" y="${Math.max(14, y - 6)}" text-anchor="middle">${textValue}</text>
      `;
    }).join("");
  }).join("");
  const labels = rows.map((row, index) => {
    const x = padding.left + index * groupW + groupW / 2;
    return `<text class="chart-label" x="${x}" y="${height - 42}" text-anchor="middle">#${row.id}</text>`;
  }).join("");
  const grid = [0, 0.25, 0.5, 0.75, 1].map((value) => {
    const y = padding.top + chartH - value * chartH;
    return `
      <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="${colors.line}"></line>
      <text class="axis-label" x="12" y="${y + 4}">${(value * 100).toFixed(0)}%</text>
    `;
  }).join("");
  const legend = metrics.map(([, color, label], index) => `
    <g transform="translate(${padding.left + index * 112}, ${height - 16})">
      <rect width="10" height="10" rx="2" fill="${color}"></rect>
      <text class="axis-label" x="16" y="10">${label}</text>
    </g>
  `).join("");

  target.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Comparison chart">
      ${grid}
      ${bars}
      ${labels}
      ${legend}
    </svg>
  `;
}

function renderComparisonConvergenceChart(series) {
  const target = document.querySelector("#comparison-convergence-chart");
  if (!target) {
    return;
  }
  const palette = [colors.teal, colors.blue, colors.green, colors.amber, colors.violet, colors.red];
  const prepared = series.map((item, index) => {
    const points = item.history
      .map((row) => ({ row, metric: metricFromHistory(row) }))
      .filter((point) => point.metric && Number.isFinite(point.metric.value))
      .map((point) => ({
        round: Number(point.row.round),
        value: point.metric.value,
        label: point.metric.label
      }));
    return {
      ...item,
      color: palette[index % palette.length],
      points
    };
  }).filter((item) => item.points.length);

  if (!prepared.length) {
    target.innerHTML = `<div class="empty-state">Selected experiments do not include plottable convergence history.</div>`;
    return;
  }

  const width = 900;
  const height = 320;
  const padding = { left: 58, right: 28, top: 24, bottom: 66 };
  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;
  const rounds = prepared.flatMap((item) => item.points.map((point) => point.round));
  const values = prepared.flatMap((item) => item.points.map((point) => point.value));
  const minRound = Math.min(...rounds);
  const maxRound = Math.max(...rounds);
  const minValue = Math.min(...values, 0);
  const maxValue = Math.max(...values, 0.01) * 1.08;
  const spanRound = Math.max(maxRound - minRound, 1);
  const spanValue = Math.max(maxValue - minValue, 0.001);
  const grid = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const value = minValue + ratio * spanValue;
    const y = padding.top + chartH - ratio * chartH;
    return `
      <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="${colors.line}"></line>
      <text class="axis-label" x="12" y="${y + 4}">${value.toFixed(2)}</text>
    `;
  }).join("");
  const lines = prepared.map((item) => {
    const polyline = item.points.map((point) => {
      const x = padding.left + ((point.round - minRound) / spanRound) * chartW;
      const y = padding.top + chartH - ((point.value - minValue) / spanValue) * chartH;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    return `<polyline points="${polyline}" fill="none" stroke="${item.color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>`;
  }).join("");
  const legend = prepared.map((item, index) => `
    <g transform="translate(${padding.left + (index % 4) * 190}, ${height - 42 + Math.floor(index / 4) * 18})">
      <line x1="0" y1="0" x2="22" y2="0" stroke="${item.color}" stroke-width="3"></line>
      <text class="axis-label" x="30" y="4">#${item.experiment_id} ${labelForAlgorithm(item.algorithm)}</text>
    </g>
  `).join("");

  target.innerHTML = `
    <svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Comparison convergence chart">
      ${grid}
      <line x1="${padding.left}" y1="${padding.top + chartH}" x2="${width - padding.right}" y2="${padding.top + chartH}" stroke="${colors.line}"></line>
      ${lines}
      <text class="axis-label" x="${padding.left}" y="${height - 16}">Round ${minRound}</text>
      <text class="axis-label" x="${width - padding.right}" y="${height - 16}" text-anchor="end">Round ${maxRound}</text>
      ${legend}
    </svg>
  `;
}

async function loadAlgorithms() {
  algorithmOptions = await apiRequest("/api/algorithms");
  populateAlgorithmSelect();
  const formStatus = document.querySelector("#experiment-form-status");
  if (formStatus) {
    formStatus.textContent = `${algorithmOptions.length} algorithms available`;
  }
}

async function loadExperimentHistory() {
  experiments = await apiRequest("/api/experiments");
  renderExperimentHistory();
  renderCompareOptions();
  if (experiments.length && !selectedExperimentId) {
    await selectExperiment(experiments[0].id);
  } else if (!experiments.length) {
    renderExperimentStatus(null);
    renderExperimentMetrics(null);
    renderExperimentConvergenceChart([]);
  }
}

async function loadDatasets() {
  uploadedDatasets = await apiRequest("/api/datasets");
  populateDatasetSelect();
  renderDatasetList();
}

async function loadExperimentWorkspace() {
  if (!accessToken) {
    renderExperimentsLoggedOut();
    return;
  }
  const notice = document.querySelector("#experiment-auth-notice");
  if (notice) {
    notice.textContent = "Loading algorithms and experiment history...";
    notice.className = "experiment-notice loading";
  }
  try {
    await loadAlgorithms();
    await loadDatasets();
    await loadExperimentHistory();
    await loadDashboard();
    if (notice) {
      notice.textContent = currentUser
        ? `Connected as ${currentUser.username}. Experiments are stored under this account.`
        : "Connected. Experiments are stored under the current account.";
      notice.className = "experiment-notice success";
    }
    setExperimentMessage("");
    setDatasetMessage("");
    setExperimentBusy(false);
  } catch (error) {
    if (notice) {
      notice.textContent = error.message;
      notice.className = "experiment-notice error";
    }
    setExperimentMessage(error.message, "error");
    setExperimentBusy(false);
  }
}

async function selectExperiment(experimentId) {
  if (!accessToken || experimentBusy) {
    return;
  }
  try {
    selectedExperimentId = experimentId;
    renderExperimentHistory();
    const result = await apiRequest(`/api/experiments/${experimentId}/results`);
    renderExperimentStatus(result);
    renderExperimentMetrics(result);
    renderExperimentConvergenceChart(result.convergence_history);
  } catch (error) {
    setExperimentMessage(error.message, "error");
  }
}

function readExperimentPayload() {
  const datasetValue = document.querySelector("#experiment-dataset").value;

  const payload = {
    algorithm: document.querySelector("#experiment-algorithm").value,
    distribution: document.querySelector("#experiment-distribution").value,
    rounds: Number(document.querySelector("#experiment-rounds").value),
    local_epochs: Number(document.querySelector("#experiment-local-epochs").value),
    learning_rate: Number(document.querySelector("#experiment-learning-rate").value),
    parameters: {}
  };

  if (
    payload.algorithm === "failure_aware_v1" ||
    payload.algorithm === "failure_aware_v2"
  ) {
    payload.parameters.alpha = Number(
      document.querySelector("#experiment-alpha").value
    );
  }

  if (payload.algorithm === "dynamic_failure_aware") {
    payload.parameters.schedule =
      document.querySelector("#experiment-schedule").value;

    payload.parameters.lambda_max = Number(
      document.querySelector("#experiment-lambda-max").value
    );

    payload.parameters.target_recall = Number(
      document.querySelector("#experiment-target-recall").value
    );

    payload.parameters.eta = Number(
      document.querySelector("#experiment-eta").value
    );
  }

  if (datasetValue) {
    payload.dataset_id = Number(datasetValue);
  }

  return payload;
}

function validateExperimentPayload(payload) {
  if (!payload.algorithm) {
    return "Choose an algorithm.";
  }
  if (!payload.distribution) {
    return "Choose a distribution.";
  }
  if (!Number.isFinite(payload.rounds) || payload.rounds <= 0) {
    return "Global rounds must be greater than 0.";
  }
  if (!Number.isFinite(payload.local_epochs) || payload.local_epochs <= 0) {
    return "Local epochs must be greater than 0.";
  }
  if (!Number.isFinite(payload.learning_rate) || payload.learning_rate <= 0) {
    return "Learning rate must be greater than 0.";
  }
  return "";
}

async function startExperiment() {
  if (!accessToken) {
    setExperimentMessage("Please login before starting an experiment.", "error");
    return;
  }
  const payload = readExperimentPayload();
  const validationError = validateExperimentPayload(payload);
  if (validationError) {
    setExperimentMessage(validationError, "error");
    return;
  }

  setExperimentBusy(true);
  setExperimentMessage("Starting experiment...", "loading");
  try {
    const created = await apiRequest("/api/experiments", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    selectedExperimentId = created.id;
    experiments = [created, ...experiments.filter((experiment) => experiment.id !== created.id)];
    renderExperimentStatus(created);
    renderExperimentMetrics(created);
    renderExperimentConvergenceChart([]);
    setExperimentMessage("Experiment running...", "loading");

    const executed = await apiRequest(`/api/experiments/${created.id}/run`, {
      method: "POST"
    });
    experiments = [executed, ...experiments.filter((experiment) => experiment.id !== executed.id)];
    renderExperimentStatus(executed);
    renderExperimentHistory();
    renderCompareOptions();

    const result = await apiRequest(`/api/experiments/${created.id}/results`);
    renderExperimentStatus(result);
    renderExperimentMetrics(result);
    renderExperimentConvergenceChart(result.convergence_history);
    await loadExperimentHistory();
    await loadDashboard();

    if (result.status === "COMPLETED") {
      setExperimentMessage("Experiment completed.", "success");
    } else if (result.status === "FAILED") {
      setExperimentMessage(`Experiment failed. ${result.error_message || ""}`.trim(), "error");
    } else {
      setExperimentMessage(`Experiment status: ${result.status}.`, "loading");
    }
  } catch (error) {
    setExperimentMessage(error.message, "error");
  } finally {
    setExperimentBusy(false);
  }
}

async function uploadDataset() {
  if (!accessToken) {
    setDatasetMessage("Please login before uploading a dataset.", "error");
    return;
  }
  const fileInput = document.querySelector("#dataset-file");
  const file = fileInput?.files?.[0];
  if (!file) {
    setDatasetMessage("Choose a CSV file first.", "error");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  setExperimentBusy(true);
  setDatasetMessage("Uploading and preparing dataset...", "loading");
  try {
    const dataset = await apiRequest("/api/datasets", {
      method: "POST",
      body: formData
    });
    uploadedDatasets = [dataset, ...uploadedDatasets.filter((item) => item.id !== dataset.id)];
    populateDatasetSelect();
    renderDatasetList();
    const select = document.querySelector("#experiment-dataset");
    if (select) {
      select.value = String(dataset.id);
    }
    if (fileInput) {
      fileInput.value = "";
    }
    setDatasetMessage(`Dataset #${dataset.id} is ready and selected.`, "success");
  } catch (error) {
    setDatasetMessage(error.message, "error");
  } finally {
    setExperimentBusy(false);
  }
}

function bindExperimentControls() {
  const form = document.querySelector("#experiment-form");
  const refreshButton = document.querySelector("#refresh-experiments-button");
  const uploadForm = document.querySelector("#dataset-upload-form");
  const algorithmSelect = document.querySelector("#experiment-algorithm");

  if (!form || !refreshButton || !uploadForm || !algorithmSelect) {
    return;
  }

  algorithmSelect.addEventListener("change", updateAlgorithmParameters);

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    startExperiment();
  });

  refreshButton.addEventListener("click", () => {
    if (!accessToken) {
      setExperimentMessage("Please login to refresh experiments.", "error");
      return;
    }

    setExperimentMessage(
      "Refreshing experiment history...",
      "loading"
    );
    loadExperimentWorkspace();
  });

  uploadForm.addEventListener("submit", (event) => {
    event.preventDefault();
    uploadDataset();
  });
}

function bindComparisonControls() {
  const compareButton = document.querySelector("#comparison-button");
  if (!compareButton) {
    return;
  }
  compareButton.addEventListener("click", compareSelectedExperiments);
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
  bindExperimentControls();
  bindComparisonControls();
  bindAuthControls();
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

function updateAlgorithmParameters() {
  const algorithm = document.querySelector("#experiment-algorithm").value;

  const container = document.querySelector("#algorithm-parameters");
  const alphaParameter = document.querySelector("#alpha-parameter");
  const scheduleParameter = document.querySelector("#schedule-parameter");
  const lambdaMaxParameter = document.querySelector("#lambda-max-parameter");
  const targetRecallParameter = document.querySelector("#target-recall-parameter");
  const etaParameter = document.querySelector("#eta-parameter");

  container.hidden = true;
  alphaParameter.hidden = true;
  scheduleParameter.hidden = true;
  lambdaMaxParameter.hidden = true;
  targetRecallParameter.hidden = true;
  etaParameter.hidden = true;

  if (
    algorithm === "failure_aware_v1" ||
    algorithm === "failure_aware_v2"
  ) {
    container.hidden = false;
    alphaParameter.hidden = false;
  }

  if (algorithm === "dynamic_failure_aware") {
    container.hidden = false;
    scheduleParameter.hidden = false;
    lambdaMaxParameter.hidden = false;
    targetRecallParameter.hidden = false;
    etaParameter.hidden = false;
  }
}

init();
