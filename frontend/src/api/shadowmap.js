const API_BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export async function fetchBlocks() {
    const response = await fetch(`${API_BASE}/blocks`);
    if (!response.ok) {
        throw new Error(`Failed to fetch blocks: ${response.statusText}`);
    }
    return response.json();
}

export async function fetchBlock(blockId) {
    const response = await fetch(`${API_BASE}/block/${blockId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch block ${blockId}: ${response.statusText}`);
    }
    return response.json();
}

export async function fetchWhatIf({ block_id, delta_buildings = 0, delta_trees = 0, delta_albedo = 0 }) {
    const response = await fetch(`${API_BASE}/whatif`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ block_id, delta_buildings, delta_trees, delta_albedo }),
    });
    if (!response.ok) {
        throw new Error(`What-If simulation failed: ${response.statusText}`);
    }
    return response.json();
}

export async function fetchWhatIfCityWide({ delta_buildings = 0, delta_trees = 0, delta_albedo = 0, scope = 'all', top_n = 20 }) {
    const response = await fetch(`${API_BASE}/whatif-citywide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ delta_buildings, delta_trees, delta_albedo, scope, top_n }),
    });
    if (!response.ok) {
        throw new Error(`City-wide What-If simulation failed: ${response.statusText}`);
    }
    return response.json();
}

export async function fetchCityStats() {
    const response = await fetch(`${API_BASE}/city-stats`);
    if (!response.ok) {
        throw new Error(`Failed to fetch city stats: ${response.statusText}`);
    }
    return response.json();
}

export async function fetchModelInfo() {
    const response = await fetch(`${API_BASE}/model-info`);
    if (!response.ok) {
        throw new Error(`Failed to fetch model info: ${response.statusText}`);
    }
    return response.json();
}
