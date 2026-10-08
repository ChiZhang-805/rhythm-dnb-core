'use strict';
const $ = (id) => document.getElementById(id);
let apiBase = '', ready = false, busy = false, lastResult = null, contract = null, epoch = 0, connectionEpoch = 0;
let pollTimer = null;
const placeholders = {emotion: '说说你今天的感受……', sleep: '说说昨晚入睡、夜间醒来或早上的状态……', diet: '说说今天的胃口、饭量或吃饭时间……', social: '说说最近与人相处时的感受……', stress: '说说最近让你有压力或轻松下来的事情……'};

function status(message, error = false) {
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function updateButton() { $('submit').disabled = !ready || busy || !$('text').value.trim(); }
function resetResult() {
  lastResult = null;
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
  ready = false; resetResult(); updateButton();
  try {
    const health = await request('/api/health', {}, 12000);
    if (current !== connectionEpoch) return;
    if (health.failed) throw new Error('模型加载失败，请检查运行环境、显存和权重。');
    if (!health.ready) {
      status('模型正在加载，页面会自动检查，请稍候……');
      pollTimer = setTimeout(connect, 5000); return;
    }
    const nextContract = await request('/api/schema', {}, 12000);
    if (current !== connectionEpoch) return;
    if (!Array.isArray(nextContract.categories) || nextContract.contract_id !== 'chinese_category_regression_17') throw new Error('这个服务不是当前文本评分模型。');
    contract = nextContract; ready = true;
    status(apiBase ? '已连接模型服务。输入文字即可分析。' : '模型已就绪，输入文字即可分析。');
  } catch (error) {
    if (current !== connectionEpoch) return;
    status(error.name === 'AbortError' ? '连接超时。可稍后重新连接，或下载本地版。' : '尚未连接可用模型。请启动本地版，或在下方填写模型服务地址。', true);
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
    const direction = document.createElement('div'); direction.className = 'metric-direction';
    direction.textContent = metric.direction === 'very_bad_to_very_good' ? '分数越高，心情、质量或满意度越好' : '分数越高，该项程度越强';
    row.append(top, track, direction); fragment.append(row);
  }
  $('bars').replaceChildren(fragment);
  $('result-context').textContent = `${definition.label} · 本次输入：${input.text}`;
  $('timing').textContent = `实验估计 · ${payload.elapsed_seconds} 秒 · 不代表诊断或概率`;
  $('empty').hidden = true; $('results').hidden = false;
  lastResult = {input, ...payload};
}
$('text').addEventListener('input', () => { ++epoch; $('count').textContent = `${Array.from($('text').value).length} / 500`; resetResult(); updateButton(); });
$('categories').addEventListener('change', () => { ++epoch; $('text').placeholder = placeholders[category()]; resetResult(); });
$('score-form').addEventListener('submit', async (event) => {
  event.preventDefault(); if (!ready || busy) return;
  const input = {category: category(), text: $('text').value.trim()};
  const current = ++epoch; busy = true; resetResult(); updateButton(); status('正在阅读这段文字，请稍候……');
  try {
    const payload = await request('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(input)});
    if (current === epoch) { render(payload, input); status('分析完成。分数是模型的实验估计，请结合原文判断。'); }
    else status('输入已改变，请重新分析。');
  } catch (error) {
    status(error.name === 'AbortError' ? '等待超时，请稍后重试。' : error.message, true);
  } finally { busy = false; updateButton(); }
});
$('download').addEventListener('click', () => {
  if (!lastResult) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastResult, null, 2)], {type: 'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = 'rhythm-text-result.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('connect').addEventListener('click', () => {
  if (busy) return;
  try {
    const value = $('endpoint').value.trim();
    const url = value ? new URL(value) : null;
    const local = url && ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    if (url && (url.username || url.password || url.search || url.hash || (url.protocol !== 'https:' && !(local && url.protocol === 'http:')))) throw new Error();
    apiBase = url ? url.href.replace(/\/$/, '') : ''; connect();
  } catch { status('请填写不包含密钥的 HTTPS 地址；本机可以使用 http://127.0.0.1:7860。', true); }
});
(async () => {
  try { const config = await fetch('./config.json', {cache: 'no-store'}).then((r) => r.json()); apiBase = config.apiBase || ''; } catch { apiBase = ''; }
  $('endpoint').value = apiBase; connect();
})();
