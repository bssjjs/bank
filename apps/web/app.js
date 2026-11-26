const GROUPS = [
  {
    label: "客户资料",
    icon: '<path d="M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8Zm-7 8a7 7 0 0 1 14 0"/>',
    fields: ["年龄", "职业", "婚姻状态", "教育程度"],
  },
  {
    label: "财务情况",
    icon: '<path d="M4 7h16v12H4zM4 11h16M8 15h3"/>',
    fields: ["是否信贷违约", "账户余额", "住房贷款", "个人贷款"],
  },
  {
    label: "联系安排",
    icon: '<path d="M5 4h14v16H5zM8 8h8M8 12h3M8 16h6"/>',
    fields: ["联系方式", "联系日期", "联系月份"],
  },
  {
    label: "联系记录",
    icon: '<path d="M4 12h4l2-6 4 12 2-6h4"/>',
    fields: ["本次联系次数", "距上次联系天数", "历史联系次数", "上次营销结果"],
  },
];

const state = {
  inputs: [],
  values: {},
  errors: {},
  ready: false,
  loading: false,
  predicting: false,
  threshold: 0.5,
  version: "",
};

const form = document.querySelector("#predict-form");
const fieldsHost = document.querySelector("#dynamic-fields");
const formLoading = document.querySelector("#form-loading");
const formError = document.querySelector("#form-error");
const submitButton = document.querySelector("#predict-button");
const statusButton = document.querySelector("#service-status");
const statusText = document.querySelector("#service-status-text");
const serviceMessage = document.querySelector("#service-message");
const modelVersion = document.querySelector("#model-version");
const resultContent = document.querySelector("#result-content");
const resultState = document.querySelector("#result-state");

function setStatus(mode, text) {
  statusButton.classList.remove("is-loading", "is-ready", "is-offline");
  statusButton.classList.add(`is-${mode}`);
  statusText.textContent = text;
}

function showServiceMessage(text = "") {
  serviceMessage.hidden = !text;
  serviceMessage.textContent = text;
}

function isNumeric(field) {
  return !field.categories?.length;
}

function rangeText(field) {
  const min = field.min ?? "—";
  const max = field.max ?? "—";
  return `${min}~${max}`;
}

function validateField(field, rawValue) {
  const value = String(rawValue ?? "").trim();
  if (!value) return "必填";
  if (!isNumeric(field)) {
    return field.categories.includes(value) ? "" : "输入无效";
  }
  const number = Number(value);
  if (!Number.isFinite(number)) return "输入无效";
  if (field.min != null && number < field.min) return `已超出范围：${rangeText(field)}`;
  if (field.max != null && number > field.max) return `已超出范围：${rangeText(field)}`;
  return "";
}

function controlId(name) {
  return `field-${encodeURIComponent(name)}`;
}

function renderControl(field) {
  const id = controlId(field.name);
  const errorId = `${id}-error`;
  const placeholder = isNumeric(field) ? rangeText(field) : "请选择";
  const name = escapeHtml(field.name);
  const unit = field.unit ? `<span class="unit">${escapeHtml(field.unit)}</span>` : "";
  const common = `
    id="${id}"
    name="${name}"
    class="control${field.unit ? " has-unit" : ""}"
    aria-describedby="${errorId}"
    aria-required="true"
    data-field="${name}"
  `;
  let control;
  if (isNumeric(field)) {
    control = `<input ${common}
      type="number"
      inputmode="decimal"
      step="any"
      placeholder="${placeholder}"
      ${field.min != null ? `min="${field.min}"` : ""}
      ${field.max != null ? `max="${field.max}"` : ""}
    />`;
  } else {
    const options = field.categories
      .map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`)
      .join("");
    control = `<select ${common} required><option value="" disabled selected>${placeholder}</option>${options}</select>`;
  }
  return `
    <div class="field" data-field-wrap="${name}">
      <label class="field-label" for="${id}">${name}</label>
      <div class="control-wrap">${control}${unit}</div>
      <div class="field-error" id="${errorId}" aria-live="polite"></div>
    </div>
  `;
}

function renderForm(inputs) {
  const assigned = new Set();
  const groups = GROUPS.map((group) => {
    const fields = group.fields.filter((name) => inputs.some((field) => field.name === name));
    fields.forEach((name) => assigned.add(name));
    return { ...group, fields };
  }).filter((group) => group.fields.length);

  const extras = inputs.filter((field) => !assigned.has(field.name)).map((field) => field.name);
  if (extras.length) {
    groups.push({
      label: "其他信息",
      icon: '<path d="M5 4h14v16H5zM8 8h8M8 12h8M8 16h5"/>',
      fields: extras,
    });
  }

  fieldsHost.innerHTML = groups
    .map((group) => {
      const fields = group.fields
        .map((name) => inputs.find((field) => field.name === name))
        .filter(Boolean);
      return `
        <fieldset class="field-group">
          <legend>
            <svg viewBox="0 0 24 24" role="img" aria-label="${group.label}">${group.icon}</svg>
            <span>${group.label}</span>
          </legend>
          <div class="field-grid">${fields.map(renderControl).join("")}</div>
        </fieldset>
      `;
    })
    .join("");

  fieldsHost.querySelectorAll(".control").forEach((control) => {
    control.addEventListener("input", () => {
      const name = control.dataset.field;
      state.values[name] = control.value;
      state.errors[name] = validateField(state.inputs.find((field) => field.name === name), control.value);
      paintField(name);
      updateSubmitState();
      formError.hidden = true;
    });
    control.addEventListener("change", () => {
      if (control.tagName === "SELECT") control.dispatchEvent(new Event("input", { bubbles: true }));
    });
  });
}

function paintField(name) {
  const wrapper = fieldsHost.querySelector(`[data-field-wrap="${name}"]`);
  const control = fieldsHost.querySelector(`[data-field="${name}"]`);
  if (!wrapper || !control) return;
  const message = state.errors[name] ?? "";
  wrapper.classList.toggle("is-invalid", Boolean(message));
  control.setAttribute("aria-invalid", message ? "true" : "false");
  wrapper.querySelector(".field-error").textContent = message;
}

function allValid() {
  return state.inputs.length > 0 && state.inputs.every((field) => !validateField(field, state.values[field.name]));
}

function updateSubmitState() {
  const hasErrors = Object.values(state.errors).some(Boolean);
  submitButton.disabled = !state.ready || !allValid() || hasErrors || state.loading || state.predicting;
}

function collectPayload() {
  return Object.fromEntries(
    state.inputs.map((field) => [
      field.name,
      isNumeric(field) ? Number(state.values[field.name]) : state.values[field.name],
    ]),
  );
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers ?? {}) },
  });
  const body = await response.json().catch(() => ({}));
  return { response, body };
}

async function loadModel() {
  state.ready = false;
  state.loading = true;
  setStatus("loading", "连接中");
  showServiceMessage();
  formLoading.hidden = false;
  fieldsHost.hidden = true;
  fieldsHost.setAttribute("aria-busy", "true");
  updateSubmitState();

  try {
    const [{ response: health }, { response: ready }, { response: model, body }] = await Promise.all([
      request("/health"),
      request("/ready"),
      request("/v1/model"),
    ]);
    if (!health.ok || !ready.ok || !model.ok) throw new Error("not-ready");
    if (!Array.isArray(body.inputs) || body.inputs.length === 0) throw new Error("empty-model");

    state.inputs = body.inputs;
    state.values = Object.fromEntries(body.inputs.map((field) => [field.name, ""]));
    state.errors = {};
    state.threshold = Number(body.decision?.threshold ?? 0.5);
    state.version = body.version;
    state.ready = true;
    modelVersion.textContent = `模型 v${body.version}`;
    modelVersion.hidden = false;
    renderForm(body.inputs);
    formLoading.hidden = true;
    fieldsHost.hidden = false;
    fieldsHost.setAttribute("aria-busy", "false");
    setStatus("ready", "服务就绪");
  } catch {
    state.ready = false;
    modelVersion.hidden = true;
    setStatus("offline", "服务未就绪");
    showServiceMessage("服务未就绪");
    formLoading.hidden = true;
    fieldsHost.hidden = true;
  } finally {
    state.loading = false;
    updateSubmitState();
  }
}

const VERDICTS = {
  优先: {
    text: "优先联系",
    tone: "priority",
    icon: '<path d="m5 12 4 4L19 6"/>',
  },
  备选: {
    text: "备选联系",
    tone: "backup",
    icon: '<circle cx="12" cy="12" r="8"/><path d="M12 8v4l3 2"/>',
  },
  不建议: {
    text: "不建议联系",
    tone: "negative",
    icon: '<path d="m7 7 10 10M17 7 7 17"/>',
  },
};

function verdictFor(level, score) {
  return (
    VERDICTS[level] ?? {
      text: score >= state.threshold ? "优先联系" : "不建议联系",
      tone: score >= state.threshold ? "priority" : "negative",
      icon:
        score >= state.threshold
          ? '<path d="m5 12 4 4L19 6"/>'
          : '<path d="m7 7 10 10M17 7 7 17"/>',
    }
  );
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function factorMarkup(factor, index, max) {
  const positive = factor.contribution >= 0;
  const width = Math.max(8, Math.round((Math.abs(factor.contribution) / max) * 100));
  return `
    <div class="factor${positive ? "" : " is-negative"}">
      <div class="factor-topline">
        <span class="factor-name">${escapeHtml(factor.feature)}</span>
        ${factor.value != null ? `<span class="factor-value">${escapeHtml(factor.value)}</span>` : ""}
        <span class="factor-contribution">${positive ? "+" : "−"}${Math.abs(factor.contribution).toFixed(3)}</span>
      </div>
      <div class="factor-track" role="img" aria-label="${escapeHtml(factor.feature)}，${positive ? "正向" : "负向"}影响 ${Math.abs(factor.contribution).toFixed(3)}">
        <div class="factor-bar" data-width="${width}" style="transition-delay:${180 + index * 80}ms"></div>
      </div>
    </div>
  `;
}

function showEmptyResult() {
  resultState.textContent = "待预测";
  resultContent.innerHTML = `
    <div class="empty-state">
      <div class="empty-visual" aria-hidden="true">
        <span class="orbit orbit-outer"></span>
        <span class="orbit orbit-inner"></span>
        <svg viewBox="0 0 64 64" fill="none">
          <path d="M14 45V31m9 14V23m9 22V17m9 28V27m9 18V13M11 51h42" />
        </svg>
      </div>
      <span class="empty-label">待预测</span>
    </div>
  `;
}

function showLoadingResult() {
  resultState.textContent = "预测中";
  resultContent.innerHTML = `
    <div class="loading-result" role="status">
      <span class="spinner" aria-hidden="true"></span>
      <span>预测中</span>
    </div>
  `;
}

function renderResult(result) {
  const score = Math.min(1, Math.max(0, Number(result.score)));
  const verdict = verdictFor(result.level, score);
  const negative = verdict.tone === "negative";
  const factors = (result.factors ?? []).slice(0, 5);
  const max = Math.max(...factors.map((factor) => Math.abs(Number(factor.contribution))), 0.001);

  resultState.textContent = result.level;
  resultContent.innerHTML = `
    <article class="prediction" aria-label="成交概率 ${(score * 100).toFixed(1)}%，${verdict.text}">
      <div class="score-area">
        <div class="score-ring">
          <svg viewBox="0 0 240 240" role="img" aria-label="成交概率 ${(score * 100).toFixed(1)}%">
            <defs>
              <linearGradient id="gold-gradient" x1="20" y1="20" x2="220" y2="220" gradientUnits="userSpaceOnUse">
                <stop stop-color="#c8a45d" />
                <stop offset="1" stop-color="#e0c584" />
              </linearGradient>
            </defs>
            <circle class="ring-track" cx="120" cy="120" r="98" />
            <circle class="ring-inner" cx="120" cy="120" r="86" />
            <circle class="ring-progress" cx="120" cy="120" r="98" pathLength="100" data-score="${score * 100}" />
          </svg>
          <div class="score-value">
            <span class="score-number" data-score="${score * 100}">0.0</span>
            <span class="score-percent">%</span>
          </div>
        </div>
        <div class="verdict${negative ? " is-negative" : verdict.tone === "backup" ? " is-backup" : ""}">
          <svg viewBox="0 0 24 24" role="img" aria-label="${verdict.text}">${verdict.icon}</svg>
          <span>${verdict.text}</span>
          <span class="level-tag">${escapeHtml(result.level)}</span>
        </div>
      </div>
      <h3 class="factors-title">主要因素</h3>
      <div class="factors">${factors.map((factor, index) => factorMarkup(factor, index, max)).join("")}</div>
      <p class="advice">${escapeHtml(result.advice)}</p>
      <div class="result-meta">
        <span>模型 v${escapeHtml(result.model_version)}</span>
        <span>${escapeHtml(result.request_id)}</span>
      </div>
    </article>
  `;

  const paintResult = () => {
    const progress = resultContent.querySelector(".ring-progress");
    progress.style.strokeDashoffset = String(100 - score * 100);
    resultContent.querySelectorAll(".factor-bar").forEach((bar) => {
      bar.style.width = `${bar.dataset.width}%`;
    });
    animateNumber(resultContent.querySelector(".score-number"), score * 100);
  };

  if (document.hidden || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    paintResult();
  } else {
    requestAnimationFrame(paintResult);
  }
}

function animateNumber(element, target) {
  if (document.hidden || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    element.textContent = target.toFixed(1);
    return;
  }
  const start = performance.now();
  const duration = 900;
  const tick = (now) => {
    const progress = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - progress, 3);
    element.textContent = (target * eased).toFixed(1);
    if (progress < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function serverMessage(detail, field) {
  const raw = typeof detail.msg === "string" ? detail.msg.trim() : "";
  if (!raw) {
    return isNumeric(field) ? validateField(field, state.values[field.name]) || "输入无效" : "输入无效";
  }
  const common = {
    "Field required": "必填",
    "Input should be a valid number": "请输入数字",
    "Input should be a valid integer": "请输入整数",
  };
  if (common[raw]) return common[raw];
  const range = raw.match(/^Input should be (greater than or equal to|less than or equal to) (.+)$/);
  if (range) {
    return range[1] === "greater than or equal to" ? `不能小于 ${range[2]}` : `不能大于 ${range[2]}`;
  }
  if (raw.startsWith("Input should be")) return "输入格式不正确";
  return raw.replace(/^Value error,\s*/i, "");
}

function applyServerErrors(details) {
  fieldsHost.querySelectorAll(".field").forEach((field) => field.classList.remove("is-invalid"));
  if (!Array.isArray(details)) return;
  for (const detail of details) {
    const name = detail.loc?.[1];
    if (typeof name !== "string") continue;
    const field = state.inputs.find((item) => item.name === name);
    if (!field) continue;
    state.errors[name] = serverMessage(detail, field);
    paintField(name);
  }
  updateSubmitState();
}

async function predict(event) {
  event.preventDefault();
  if (!state.ready || state.predicting) return;
  if (!allValid()) {
    formError.textContent = "请检查标红字段";
    formError.hidden = false;
    showEmptyResult();
    return;
  }
  state.predicting = true;
  formError.hidden = true;
  updateSubmitState();
  submitButton.classList.add("is-loading");
  submitButton.querySelector("span").textContent = "预测中";
  showLoadingResult();

  try {
    const { response, body } = await request("/v1/predict", {
      method: "POST",
      body: JSON.stringify(collectPayload()),
    });
    if (response.status === 422) {
      applyServerErrors(body.detail ?? []);
      formError.textContent = "请检查标红字段";
      formError.hidden = false;
      showEmptyResult();
      return;
    }
    if (!response.ok) throw new Error("predict-failed");
    renderResult(body);
  } catch {
    formError.textContent = "预测失败，请确认服务已启动";
    formError.hidden = false;
    showEmptyResult();
    setStatus("offline", "服务异常");
    state.ready = false;
  } finally {
    state.predicting = false;
    submitButton.classList.remove("is-loading");
    submitButton.querySelector("span").textContent = "预测成交";
    updateSubmitState();
  }
}

form.addEventListener("submit", predict);
statusButton.addEventListener("click", loadModel);
window.addEventListener("focus", () => {
  if (!state.ready && !state.loading) loadModel();
});

loadModel();
