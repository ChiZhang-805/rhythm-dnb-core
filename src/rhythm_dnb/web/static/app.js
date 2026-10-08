'use strict';
const $ = (id) => document.getElementById(id);
let apiBase = '', ready = false, busy = false, contract = null, epoch = 0, connectionEpoch = 0;
let pollTimer = null;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function updateButton() { $('submit').disabled = !ready || busy || !$('text').value.trim(); }
function resetResult() {
  $('results').hidden = true;
  $('empty').hidden = false;
}
function category() { return document.querySelector('input[name="category"]:checked').value; }
async function request(path, options = {}, timeout = 90000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(apiBase + path, {...options, signal: controller.signal, credentials: 'omit', cache: 'no-store'});
    const data = await response.json().catch(() => null);
    if (!response.ok || !data) throw new Error(data?.error || '模型服务暂时不可用。');
    return data;
  } finally { clearTimeout(timer); }
}
async function connect() {
  const current = ++connectionEpoch;
  clearTimeout(pollTimer);
  $('retry').hidden = true;
  ready = false; resetResult(); updateButton();
  try {
    const health = await request('/api/health', {}, 12000);
    if (current !== connectionEpoch) return;
    if (health.failed) throw new Error('模型加载失败。');
    if (!health.ready) {
      status('模型加载中……');
      pollTimer = setTimeout(connect, 5000); return;
    }
    const nextContract = await request('/api/schema', {}, 12000);
    if (current !== connectionEpoch) return;
    if (!Array.isArray(nextContract.categories) || nextContract.contract_id !== 'chinese_category_regression_17') throw new Error('这个服务不是当前文本评分模型。');
    contract = nextContract; ready = true;
    status('');
  } catch (error) {
    if (current !== connectionEpoch) return;
    status(error.name === 'AbortError' ? '连接超时。' : '模型未连接。', true);
    $('retry').hidden = false;
  }
  updateButton();
}
function render(payload, input) {
  const definition = contract.categories.find((item) => item.id === input.category);
  const output = payload.result;
  if (!definition || output?.category !== input.category || !output.estimates) throw new Error('模型返回格式不匹配，请重试。');
  const fragment = document.createDocumentFragment();
  for (const metric of definition.outputs) {
    const value = output.estimates[metric.key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 100) throw new Error('模型返回了无效分数。');
    const row = document.createElement('div'); row.className = 'metric';
    const top = document.createElement('div'); top.className = 'metric-top';
    const label = document.createElement('span'); label.textContent = metric.label;
    const number = document.createElement('span'); number.className = 'metric-number'; number.textContent = value.toFixed(1);
    const unit = document.createElement('small'); unit.textContent = ' / 100'; number.append(unit); top.append(label, number);
    const track = document.createElement('div'); track.className = 'track'; track.setAttribute('role', 'meter');
    track.setAttribute('aria-label', metric.label); track.setAttribute('aria-valuenow', value.toFixed(1)); track.setAttribute('aria-valuemin', '0'); track.setAttribute('aria-valuemax', '100');
    const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${value}%`; track.append(fill);
    row.append(top, track); fragment.append(row);
  }
  $('bars').replaceChildren(fragment);
  $('empty').hidden = true; $('results').hidden = false;
}
$('text').addEventListener('input', () => { ++epoch; $('count').textContent = `${Array.from($('text').value).length} / 500`; resetResult(); updateButton(); });
$('categories').addEventListener('change', () => { ++epoch; resetResult(); status(''); });
$('score-form').addEventListener('submit', async (event) => {
  event.preventDefault(); if (!ready || busy) return;
  const input = {category: category(), text: $('text').value.trim()};
  const current = ++epoch; busy = true; resetResult(); updateButton(); status('分析中……');
  try {
    const payload = await request('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(input)});
    if (current === epoch) { render(payload, input); status(''); }
    else status('输入已改变，请重新分析。');
  } catch (error) {
    status(error.name === 'AbortError' ? '等待超时，请稍后重试。' : error.message, true);
  } finally { busy = false; updateButton(); }
});
$('retry').addEventListener('click', () => { if (!busy) connect(); });
(async () => {
  try { const config = await fetch('./config.json', {cache: 'no-store'}).then((r) => r.json()); apiBase = config.apiBase || ''; } catch { apiBase = ''; }
  connect();
})();
