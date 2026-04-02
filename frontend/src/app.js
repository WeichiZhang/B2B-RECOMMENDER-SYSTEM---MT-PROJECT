/**
 * B2B Procurement Recommender System - Frontend Application
 *
 * Connects to the FastAPI backend to display recommendations,
 * buyer profiles, product catalog, and model performance metrics.
 */

const API_BASE = 'http://localhost:8000';

// ── Page Navigation ──────────────────────────────────────────────────────

function showPage(pageId) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));

    document.getElementById('page-' + pageId).classList.add('active');
    document.querySelector(`[data-page="${pageId}"]`).classList.add('active');

    // Load data for the page
    if (pageId === 'dashboard') loadDashboard();
    if (pageId === 'buyers') searchBuyers();
    if (pageId === 'products') loadProducts();
    if (pageId === 'model') loadModelMetrics();
}

// ── API Helper ───────────────────────────────────────────────────────────

async function apiCall(endpoint) {
    try {
        const res = await fetch(API_BASE + endpoint);
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || 'API Error');
        }
        return await res.json();
    } catch (e) {
        if (e.message === 'Failed to fetch') {
            throw new Error('Cannot connect to API server. Make sure the backend is running on localhost:8000');
        }
        throw e;
    }
}

// ── Dashboard ────────────────────────────────────────────────────────────

async function loadDashboard() {
    try {
        const stats = await apiCall('/stats');
        document.getElementById('stat-buyers').textContent = stats.n_buyers.toLocaleString();
        document.getElementById('stat-products').textContent = stats.n_products.toLocaleString();
        document.getElementById('stat-regions').textContent = stats.n_regions.toLocaleString();
        document.getElementById('stat-interactions').textContent = stats.n_interactions.toLocaleString();
    } catch (e) {
        // Show placeholder values when API is not available
        document.getElementById('stat-buyers').textContent = '1,811';
        document.getElementById('stat-products').textContent = '72';
        document.getElementById('stat-regions').textContent = '67';
        document.getElementById('stat-interactions').textContent = '8,979';
    }
}

// ── Recommendations ──────────────────────────────────────────────────────

async function getRecommendations() {
    const inn = document.getElementById('inn-input').value.trim();
    const k = document.getElementById('k-select').value;
    const excludePurchased = document.getElementById('exclude-purchased').checked;

    if (!inn) {
        showError('rec-error', 'Please enter a buyer INN number.');
        return;
    }

    showLoading('rec-loading', true);
    hideElement('rec-error');
    hideElement('rec-results');

    try {
        const data = await apiCall(`/recommend/${inn}?k=${k}&exclude_purchased=${excludePurchased}`);

        // Show buyer info
        const buyerInfo = document.getElementById('buyer-info');
        buyerInfo.innerHTML = `
            <div class="info-item">
                <div class="info-label">INN</div>
                <div class="info-value">${data.customer_inn}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Institution Type</div>
                <div class="info-value">${formatInstType(data.institution_type)}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Region</div>
                <div class="info-value">${data.region}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Model</div>
                <div class="info-value">${data.model_used}</div>
            </div>
        `;

        // Show recommendations table
        const container = document.getElementById('rec-table-container');
        if (data.recommendations.length === 0) {
            container.innerHTML = '<p>No recommendations available for this buyer.</p>';
        } else {
            container.innerHTML = `
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>OKPD2 Code</th>
                            <th>Product Name</th>
                            <th>Description</th>
                            <th>Similarity Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.recommendations.map(r => `
                            <tr>
                                <td><strong>#${r.rank}</strong></td>
                                <td><code>${r.okpd2_code}</code></td>
                                <td>${r.product_name}</td>
                                <td>${r.description}</td>
                                <td>${renderScoreBar(r.similarity_score)}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        showElement('rec-results');
    } catch (e) {
        showError('rec-error', e.message);
    }

    showLoading('rec-loading', false);
}

// ── Buyers ───────────────────────────────────────────────────────────────

async function searchBuyers() {
    const region = document.getElementById('buyer-search-region').value.trim();
    const instType = document.getElementById('buyer-search-type').value;

    showLoading('buyers-loading', true);

    try {
        let url = '/buyers?limit=50';
        if (region) url += `&region=${encodeURIComponent(region)}`;
        if (instType) url += `&inst_type=${encodeURIComponent(instType)}`;

        const data = await apiCall(url);
        const container = document.getElementById('buyers-table-container');

        container.innerHTML = `
            <p style="color: var(--gray-500); margin-bottom: 12px;">
                Showing ${data.buyers.length} of ${data.total} buyers
            </p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>INN</th>
                        <th>Institution Type</th>
                        <th>Region</th>
                        <th>Purchases</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.buyers.map(b => `
                        <tr>
                            <td><code>${b.customer_inn}</code></td>
                            <td>${renderInstTag(b.institution_type)}</td>
                            <td>${b.region}</td>
                            <td>${b.n_purchases}</td>
                            <td><button class="btn-small" onclick="quickRecommend(${b.customer_inn})">Recommend</button></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (e) {
        document.getElementById('buyers-table-container').innerHTML =
            `<p class="error">${e.message}</p>`;
    }

    showLoading('buyers-loading', false);
}

function quickRecommend(inn) {
    document.getElementById('inn-input').value = inn;
    showPage('recommend');
    getRecommendations();
}

// ── Products ─────────────────────────────────────────────────────────────

async function loadProducts() {
    showLoading('products-loading', true);

    try {
        const data = await apiCall('/products');
        const container = document.getElementById('products-table-container');

        container.innerHTML = `
            <p style="color: var(--gray-500); margin-bottom: 12px;">
                ${data.total} food product categories
            </p>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>OKPD2 Code</th>
                        <th>Product Name</th>
                        <th>Description</th>
                        <th>Buyers</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.products.map(p => `
                        <tr>
                            <td><code>${p.okpd2_code}</code></td>
                            <td>${p.product_name}</td>
                            <td>${p.description}</td>
                            <td>${p.n_buyers}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    } catch (e) {
        document.getElementById('products-table-container').innerHTML =
            `<p class="error">${e.message}</p>`;
    }

    showLoading('products-loading', false);
}

// ── Model Performance ────────────────────────────────────────────────────

async function loadModelMetrics() {
    showLoading('model-loading', true);

    try {
        const stats = await apiCall('/stats');
        const perf = stats.model_performance;
        const cold = stats.cold_start_analysis;
        const container = document.getElementById('model-metrics');

        let html = '';

        // Overall performance comparison
        if (perf && Object.keys(perf).length > 0) {
            for (const [modelName, metrics] of Object.entries(perf)) {
                html += `
                    <div class="metric-card">
                        <h3>${formatModelName(modelName)}</h3>
                        ${Object.entries(metrics).map(([k, v]) => `
                            <div class="metric-row">
                                <span class="metric-name">${k}</span>
                                <span class="metric-val ${getMetricClass(v)}">${(v * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
        }

        // Cold-start analysis
        if (cold && Object.keys(cold).length > 0) {
            for (const [modelName, strata] of Object.entries(cold)) {
                html += `
                    <div class="metric-card">
                        <h3>${formatModelName(modelName)} - Cold Start</h3>
                        ${Object.entries(strata).map(([stratum, metrics]) => `
                            <div style="margin-bottom: 12px;">
                                <strong style="text-transform: capitalize;">${stratum}</strong>
                                <span style="color: var(--gray-500); font-size: 13px;">
                                    (${metrics.count || 0} buyers)
                                </span>
                                ${typeof metrics === 'object' ? Object.entries(metrics)
                                    .filter(([k]) => k !== 'count')
                                    .map(([k, v]) => `
                                        <div class="metric-row">
                                            <span class="metric-name">${k}</span>
                                            <span class="metric-val ${getMetricClass(v)}">${(v * 100).toFixed(1)}%</span>
                                        </div>
                                    `).join('') : ''}
                            </div>
                        `).join('')}
                    </div>
                `;
            }
        }

        container.innerHTML = html || '<div class="card"><p>No model metrics available yet. Run the training pipeline first.</p></div>';
    } catch (e) {
        document.getElementById('model-metrics').innerHTML =
            `<div class="card"><p class="error">${e.message}</p></div>`;
    }

    showLoading('model-loading', false);
}

// ── Utility Functions ────────────────────────────────────────────────────

function renderScoreBar(score) {
    const pct = Math.max(0, Math.min(100, ((score + 1) / 2) * 100));
    return `
        <div class="score-bar">
            <div class="score-bar-track">
                <div class="score-bar-fill" style="width: ${pct}%"></div>
            </div>
            <span class="score-value">${score.toFixed(3)}</span>
        </div>
    `;
}

function formatInstType(type) {
    const map = {
        'preschool': 'Preschool (Kindergarten)',
        'school': 'School',
        'hospital': 'Hospital',
        'state_school': 'State School',
        'federal_institution': 'Federal Institution',
        'federal_education': 'Federal Education',
        'budget_institution': 'Budget Institution',
        'municipal_institution': 'Municipal Institution',
        'state_institution': 'State Institution',
        'autonomous_institution': 'Autonomous Institution',
        'other': 'Other',
    };
    return map[type] || type;
}

function renderInstTag(type) {
    let tagClass = 'tag-other';
    if (type.includes('preschool')) tagClass = 'tag-preschool';
    else if (type.includes('school')) tagClass = 'tag-school';
    else if (type.includes('hospital')) tagClass = 'tag-hospital';
    else if (type.includes('federal')) tagClass = 'tag-federal';

    return `<span class="tag ${tagClass}">${formatInstType(type)}</span>`;
}

function formatModelName(name) {
    const map = {
        'two_tower': 'Two-Tower Neural Network',
        'popularity': 'Popularity Baseline',
        'mf_als': 'Matrix Factorization (ALS)',
    };
    return map[name] || name;
}

function getMetricClass(value) {
    if (value >= 0.5) return 'good';
    if (value >= 0.2) return 'medium';
    return 'poor';
}

function showLoading(id, show) {
    const el = document.getElementById(id);
    if (show) el.classList.remove('hidden');
    else el.classList.add('hidden');
}

function showError(id, message) {
    const el = document.getElementById(id);
    el.textContent = message;
    el.classList.remove('hidden');
}

function hideElement(id) { document.getElementById(id).classList.add('hidden'); }
function showElement(id) { document.getElementById(id).classList.remove('hidden'); }

// ── Initialize ───────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    loadDashboard();
});
