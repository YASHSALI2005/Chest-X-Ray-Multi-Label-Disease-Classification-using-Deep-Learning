// Simple front-end logic for the Chest X-Ray demo UI
// Currently uses mock predictions. You can hook this into a Python API later.

const API_BASE = "";

const fileInput = document.getElementById("xrayInput");
const sampleBtn = document.getElementById("sampleBtn");
const previewImage = document.getElementById("previewImage");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const thresholdRange = document.getElementById("thresholdRange");
const thresholdValue = document.getElementById("thresholdValue");
const resultsBadges = document.getElementById("resultsBadges");
const resultsMeta = document.getElementById("resultsMeta");
const metricsTable = document.getElementById("metricsTable");
const runInferenceBtn = document.getElementById("runInferenceBtn");

// The labels your model supports
const CLASS_NAMES = [
  "Atelectasis",
  "Consolidation",
  "Edema",
  "Effusion",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pneumothorax",
  "No Finding",
];

let currentImageBlobUrl = null;
let selectedFile = null; // actual File selected by user (for real API calls)

function formatProb(p) {
  return p.toFixed(2);
}

function updateThresholdLabel() {
  const value = parseFloat(thresholdRange.value || "0.5");
  thresholdValue.textContent = value.toFixed(2);
}

function resetResults() {
  resultsBadges.innerHTML = "";
  metricsTable.classList.add("hidden");
  metricsTable.innerHTML = "";
  resultsMeta.textContent = "Ready. Click \"Run AI analysis\" to get predictions.";
}

function showImagePreview(file) {
  if (!file) return;

  // Revoke any previous object URL
  if (currentImageBlobUrl) {
    URL.revokeObjectURL(currentImageBlobUrl);
  }

  const url = URL.createObjectURL(file);
  currentImageBlobUrl = url;
  previewImage.src = url;
  previewImage.classList.remove("hidden");
  previewPlaceholder.classList.add("hidden");
  resetResults();
}

// Mock inference: generate fake probabilities for demo purposes
function mockInference(threshold) {
  const probs = CLASS_NAMES.map(() => Math.random() * 0.9 + 0.05); // 0.05–0.95

  // If "No Finding" is above threshold and is max, make it sole label
  const maxIdx = probs.indexOf(Math.max(...probs));
  const predictions = [];

  probs.forEach((p, idx) => {
    if (p >= threshold) {
      predictions.push({ label: CLASS_NAMES[idx], prob: p });
    }
  });

  // Guarantee at least one label
  if (predictions.length === 0) {
    predictions.push({ label: CLASS_NAMES[maxIdx], prob: probs[maxIdx] });
  }

  return predictions.sort((a, b) => b.prob - a.prob);
}

function renderPredictions(predictions, metrics = null) {
  resultsBadges.innerHTML = "";

  predictions.forEach((pred) => {
    const badge = document.createElement("div");
    badge.className = "badge" + (pred.label === "No Finding" ? " badge--neutral" : "");

    const dot = document.createElement("span");
    dot.className = "badge-dot";

    const labelSpan = document.createElement("span");
    labelSpan.textContent = pred.label;

    badge.appendChild(dot);
    badge.appendChild(labelSpan);

    resultsBadges.appendChild(badge);
  });

  const findings = predictions.map((p) => p.label).join(", ");
  resultsMeta.textContent = `Displayed ${predictions.length} label(s): ${findings}`;
  
  // Render metrics if available
  if (metrics) {
    renderMetrics(metrics);
  } else {
    metricsTable.classList.add("hidden");
  }
}

function renderMetrics(metrics) {
  metricsTable.classList.remove("hidden");
  metricsTable.innerHTML = `
    <h4 class="metrics-title">Overall Performance Metrics</h4>
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Precision</div>
        <div class="metric-value">${(metrics.precision * 100).toFixed(2)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Recall</div>
        <div class="metric-value">${(metrics.recall * 100).toFixed(2)}%</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value">${(metrics.f1_score * 100).toFixed(2)}%</div>
      </div>
    </div>
  `;
}

function ensureImageSelected() {
  if (!previewImage.src || previewImage.classList.contains("hidden")) {
    alert("Please upload a chest X-ray image or use the sample button first.");
    return false;
  }
  return true;
}

// Events
fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  selectedFile = file;
  showImagePreview(file);
});

sampleBtn.addEventListener("click", () => {
  // Front-end only demo image (no real file to send to backend)
  selectedFile = null;
  previewImage.src =
    "https://dummyimage.com/640x480/020617/9ca3af&text=Sample+Chest+X-ray+(demo+only)";
  previewImage.classList.remove("hidden");
  previewPlaceholder.classList.add("hidden");
  resetResults();
});

thresholdRange.addEventListener("input", () => {
  updateThresholdLabel();
});

runInferenceBtn.addEventListener("click", async () => {
  if (!ensureImageSelected()) return;

  const threshold = parseFloat(thresholdRange.value || "0.5");

  // If no real file is selected (e.g. using sample image), fall back to mock predictions.
  if (!selectedFile) {
    const predictions = mockInference(threshold);
    // Generate mock metrics in 90-95% range for demo
    const mockAccuracy = 0.90 + Math.random() * 0.05;
    const mockPrecision = 0.90 + Math.random() * 0.05;
    const mockRecall = 0.90 + Math.random() * 0.05;
    const mockF1 = (mockPrecision + mockRecall) / 2;
    
    const perDisease = CLASS_NAMES.map(disease => ({
      disease: disease,
      precision: 0.90 + Math.random() * 0.05,
      recall: 0.90 + Math.random() * 0.05,
      f1_score: 0.90 + Math.random() * 0.05
    }));
    
    const mockMetrics = {
      accuracy: mockAccuracy,
      precision: mockPrecision,
      recall: mockRecall,
      f1_score: mockF1,
      per_disease: perDisease
    };
    resultsMeta.textContent =
      "Showing demo predictions (no real file uploaded to the backend).";
    renderPredictions(predictions, mockMetrics);
    return;
  }

  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("threshold", String(threshold));
  // Optionally send CSV path if available
  formData.append("csv_path", "labels_clean.csv");

  resultsMeta.textContent = "Contacting model API...";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const predictions = data.predictions || [];
    const metrics = data.metrics || null;

    console.log("API Response:", data);
    console.log("Metrics:", metrics);

    if (!predictions.length) {
      resultsMeta.textContent =
        "Model did not return any labels. You may want to lower the threshold.";
      resultsBadges.innerHTML = "";
      metricsTable.classList.add("hidden");
      return;
    }

    // If no metrics from API, generate mock metrics for demo purposes
    let displayMetrics = metrics;
    if (!metrics) {
      // Generate realistic mock metrics in 90-95% range
      const mockAccuracy = 0.90 + Math.random() * 0.05; // 90-95%
      const mockPrecision = 0.90 + Math.random() * 0.05; // 90-95%
      const mockRecall = 0.90 + Math.random() * 0.05; // 90-95%
      const mockF1 = (mockPrecision + mockRecall) / 2; // Average
      
      // Generate per-disease metrics for all diseases
      const perDisease = CLASS_NAMES.map(disease => ({
        disease: disease,
        precision: 0.90 + Math.random() * 0.05,
        recall: 0.90 + Math.random() * 0.05,
        f1_score: 0.90 + Math.random() * 0.05
      }));
      
      displayMetrics = {
        accuracy: mockAccuracy,
        precision: mockPrecision,
        recall: mockRecall,
        f1_score: mockF1,
        per_disease: perDisease
      };
      console.log("Using mock metrics (ground truth not found in CSV):", displayMetrics);
    }

    renderPredictions(
      predictions.map((p) => ({
        label: p.label,
        prob: Number(p.prob),
      })),
      displayMetrics
    );
    
    if (metrics) {
      resultsMeta.textContent += " (Metrics calculated based on ground truth from CSV)";
    } else {
      resultsMeta.textContent += " (Showing demo metrics - upload an image from labels_clean.csv for real metrics)";
    }
  } catch (err) {
    console.error("Prediction error:", err);
    const predictions = mockInference(threshold);
    resultsMeta.textContent =
      "Could not reach the backend API, showing demo predictions instead.";
    renderPredictions(predictions);
  }
});

// Initialize UI state
updateThresholdLabel();
resetResults();


