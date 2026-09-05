const form = document.getElementById("predictForm");
const submitBtn = document.getElementById("submitBtn");
const resultBox = document.getElementById("result");
const gaugeArc = document.getElementById("gaugeArc");
const gaugeNeedle = document.getElementById("gaugeNeedle");
const gaugeLabel = document.getElementById("gaugeLabel");

const ARC_LENGTH = 219; // matches stroke-dasharray in the SVG
// Rough reference ceiling for the gauge fill (not a hard model limit).
const GAUGE_CEILING = 1_500_000;

function setGauge(fraction, label) {
  const clamped = Math.max(0, Math.min(1, fraction));
  gaugeArc.style.strokeDashoffset = String(ARC_LENGTH * (1 - clamped));
  gaugeNeedle.style.transform = `rotate(${-90 + clamped * 180}deg)`;
  gaugeLabel.textContent = label;
}

function formatINR(value) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  submitBtn.classList.toggle("is-loading", isLoading);
}

function showSuccess(price) {
  resultBox.className = "result is-success";
  resultBox.innerHTML = `
    <span class="result__figure">${formatINR(price)}</span>
    estimated fair market price
  `;
  setGauge(Math.min(price / GAUGE_CEILING, 1), formatINR(price));
}

function showError(message) {
  resultBox.className = "result is-error";
  resultBox.textContent = message;
  setGauge(0, "estimate failed");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultBox.className = "result";
  resultBox.textContent = "";

  const payload = {
    name: document.getElementById("name").value.trim(),
    company: document.getElementById("company").value,
    year: Number(document.getElementById("year").value),
    kms_driven: Number(document.getElementById("kms_driven").value),
    fuel_type: document.getElementById("fuel_type").value,
  };

  if (!payload.name || !payload.company || !payload.year || !payload.fuel_type || Number.isNaN(payload.kms_driven)) {
    showError("Please fill in every field before estimating.");
    return;
  }

  setLoading(true);
  setGauge(0.05, "calculating…");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.detail || "Something went wrong while estimating the price.");
      return;
    }

    showSuccess(data.predicted_price);
  } catch (err) {
    showError("Couldn't reach the prediction service. Check your connection and try again.");
  } finally {
    setLoading(false);
  }
});
