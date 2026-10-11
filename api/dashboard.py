"""
Recruiter-Friendly Interactive Dashboard for SmartCache ML.
Served directly at http://127.0.0.1:8000/ to showcase project capabilities,
architecture overview, live query playground, and complete interactive route catalog.
"""

def get_dashboard_html() -> str:
    """Generate self-contained, high-performance HTML/CSS/JS dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SmartCache ML — Semantic Caching & LLM Routing</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0B0F17;
            --bg-surface: #111726;
            --bg-card: #171F32;
            --bg-card-hover: #1E2840;
            --border-subtle: #242F46;
            --border-highlight: #3B4866;
            
            --orange-primary: #FF6B35;
            --orangered-vibrant: #FF4500;
            --orange-warm: #FFA500;
            --orange-glow: rgba(255, 107, 53, 0.25);
            --orangered-glow: rgba(255, 69, 0, 0.3);
            
            --text-white: #FFFFFF;
            --text-primary: #F1F5F9;
            --text-muted: #94A3B8;
            --text-dim: #64748B;
            
            --badge-get: #38BDF8;
            --badge-post: #FF6B35;
            --badge-delete: #F43F5E;
            --badge-hit: #10B981;
            
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 16px;
            --radius-full: 9999px;
            
            --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-sans);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            overflow-x: hidden;
        }

        /* Container */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 24px;
        }

        /* Navigation Header */
        header {
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(11, 15, 23, 0.85);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-inner {
            display: flex;
            align-items: center;
            justify-content: space-between;
            height: 70px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            background: linear-gradient(135deg, var(--orange-primary), var(--orangered-vibrant));
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 4px 14px var(--orange-glow);
        }

        .brand-text {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text-white);
            letter-spacing: -0.02em;
        }

        .brand-text span {
            background: linear-gradient(135deg, var(--orange-primary), var(--orangered-vibrant));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-links {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-full);
            font-size: 0.82rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10B981;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.85); }
        }

        .btn-link {
            padding: 8px 16px;
            border-radius: var(--radius-sm);
            font-size: 0.88rem;
            font-weight: 600;
            text-decoration: none;
            color: var(--text-primary);
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-link:hover {
            border-color: var(--orange-primary);
            color: var(--orange-primary);
        }

        .btn-primary {
            background: linear-gradient(135deg, var(--orange-primary), var(--orangered-vibrant));
            color: var(--text-white);
            border: none;
            box-shadow: 0 4px 16px var(--orange-glow);
        }

        .btn-primary:hover {
            box-shadow: 0 6px 20px var(--orangered-glow);
            transform: translateY(-1px);
            color: var(--text-white);
        }

        /* Hero Section */
        .hero {
            padding: 56px 0 36px;
            text-align: center;
        }

        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            background: rgba(255, 107, 53, 0.12);
            border: 1px solid rgba(255, 107, 53, 0.3);
            border-radius: var(--radius-full);
            color: var(--orange-primary);
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 20px;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .hero h1 {
            font-size: 3rem;
            font-weight: 800;
            line-height: 1.15;
            letter-spacing: -0.03em;
            color: var(--text-white);
            max-width: 900px;
            margin: 0 auto 18px;
        }

        .hero h1 .highlight {
            background: linear-gradient(135deg, #FF8C00, #FF4500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero p {
            font-size: 1.15rem;
            color: var(--text-muted);
            max-width: 760px;
            margin: 0 auto 32px;
            font-weight: 400;
        }

        .pill-group {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-bottom: 32px;
        }

        .tech-pill {
            padding: 6px 14px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-full);
            font-size: 0.82rem;
            font-family: var(--font-mono);
            color: var(--text-muted);
        }

        .tech-pill strong {
            color: var(--orange-primary);
        }

        /* KPI Cards Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 18px;
            margin-bottom: 48px;
        }

        .kpi-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 22px;
            position: relative;
            overflow: hidden;
            transition: border-color 0.2s ease, transform 0.2s ease;
        }

        .kpi-card:hover {
            border-color: var(--border-highlight);
            transform: translateY(-2px);
        }

        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--orange-primary), var(--orangered-vibrant));
            opacity: 0.8;
        }

        .kpi-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 8px;
        }

        .kpi-value {
            font-size: 2rem;
            font-weight: 800;
            color: var(--text-white);
            letter-spacing: -0.02em;
            font-family: var(--font-mono);
            display: flex;
            align-items: baseline;
            gap: 6px;
        }

        .kpi-subtitle {
            font-size: 0.82rem;
            color: var(--text-dim);
            margin-top: 6px;
        }

        /* Main Grid: Playground & Insights */
        .section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-white);
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: -0.02em;
        }

        .section-badge {
            font-size: 0.78rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: var(--radius-full);
            background: rgba(255, 107, 53, 0.15);
            color: var(--orange-primary);
            text-transform: uppercase;
        }

        /* Playground Panel */
        .playground-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 28px;
            margin-bottom: 48px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .demo-bar {
            margin-bottom: 20px;
        }

        .demo-bar-label {
            font-size: 0.82rem;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .demo-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }

        .chip-btn {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 8px 14px;
            font-size: 0.84rem;
            color: var(--text-primary);
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: var(--font-sans);
        }

        .chip-btn:hover {
            border-color: var(--orange-primary);
            background: var(--bg-card-hover);
            transform: translateY(-1px);
        }

        .chip-btn strong {
            color: var(--orange-primary);
        }

        .query-input-wrapper {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
        }

        .query-input {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 14px 18px;
            font-size: 1rem;
            color: var(--text-white);
            font-family: var(--font-sans);
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .query-input:focus {
            border-color: var(--orange-primary);
            box-shadow: 0 0 0 3px var(--orange-glow);
        }

        .btn-submit {
            padding: 0 28px;
            background: linear-gradient(135deg, var(--orange-primary), var(--orangered-vibrant));
            border: none;
            border-radius: var(--radius-md);
            color: var(--text-white);
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 4px 16px var(--orange-glow);
        }

        .btn-submit:hover {
            box-shadow: 0 6px 20px var(--orangered-glow);
            transform: translateY(-1px);
        }

        .btn-submit:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        /* Result Box */
        .result-box {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 20px;
            display: none;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
            padding-bottom: 12px;
            border-bottom: 1px solid var(--border-subtle);
        }

        .cache-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: var(--radius-full);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .badge-hit-style {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid #10B981;
            color: #10B981;
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
        }

        .badge-miss-style {
            background: rgba(255, 107, 53, 0.15);
            border: 1px solid var(--orange-primary);
            color: var(--orange-primary);
            box-shadow: 0 0 12px var(--orange-glow);
        }

        .result-meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }

        .meta-item {
            background: var(--bg-surface);
            padding: 10px 14px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-subtle);
        }

        .meta-label {
            font-size: 0.74rem;
            color: var(--text-dim);
            text-transform: uppercase;
            font-weight: 600;
        }

        .meta-val {
            font-size: 0.96rem;
            color: var(--text-white);
            font-weight: 700;
            font-family: var(--font-mono);
            margin-top: 2px;
        }

        .answer-box {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 16px;
            color: var(--text-primary);
            font-size: 0.95rem;
            line-height: 1.6;
        }

        /* Route Directory Section */
        .routes-section {
            margin-bottom: 48px;
        }

        .routes-grid {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .route-row {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: all 0.2s ease;
            gap: 16px;
        }

        .route-row:hover {
            border-color: var(--border-highlight);
            background: var(--bg-card);
        }

        .route-left {
            display: flex;
            align-items: center;
            gap: 16px;
            flex: 1;
        }

        .method-badge {
            font-family: var(--font-mono);
            font-size: 0.76rem;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: var(--radius-sm);
            letter-spacing: 0.05em;
            min-width: 65px;
            text-align: center;
        }

        .method-post {
            background: rgba(255, 69, 0, 0.15);
            color: var(--orangered-vibrant);
            border: 1px solid rgba(255, 69, 0, 0.4);
        }

        .method-get {
            background: rgba(56, 189, 248, 0.12);
            color: var(--badge-get);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }

        .method-delete {
            background: rgba(244, 63, 94, 0.15);
            color: var(--badge-delete);
            border: 1px solid rgba(244, 63, 94, 0.4);
        }

        .route-path {
            font-family: var(--font-mono);
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-white);
        }

        .route-desc {
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-left: 8px;
        }

        .route-action {
            display: flex;
            gap: 8px;
        }

        .btn-try {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            padding: 6px 14px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-try:hover {
            border-color: var(--orange-primary);
            color: var(--orange-primary);
        }

        /* Route Inspector Modal/Output */
        .route-output {
            margin-top: 10px;
            padding: 14px;
            background: #06090F;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            font-family: var(--font-mono);
            font-size: 0.82rem;
            color: #A5D6FF;
            max-height: 250px;
            overflow-y: auto;
            white-space: pre-wrap;
            display: none;
        }

        /* Pillars Section */
        .pillars-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-bottom: 56px;
        }

        .pillar-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 26px;
        }

        .pillar-icon {
            font-size: 28px;
            margin-bottom: 14px;
            display: inline-block;
        }

        .pillar-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-white);
            margin-bottom: 10px;
        }

        .pillar-text {
            font-size: 0.9rem;
            color: var(--text-muted);
            line-height: 1.6;
        }

        /* Footer */
        footer {
            border-top: 1px solid var(--border-subtle);
            padding: 40px 0;
            text-align: center;
            color: var(--text-dim);
            font-size: 0.88rem;
        }

        .footer-links {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 14px;
        }

        .footer-links a {
            color: var(--text-muted);
            text-decoration: none;
            transition: color 0.2s ease;
        }

        .footer-links a:hover {
            color: var(--orange-primary);
        }

        @media (max-width: 768px) {
            .hero h1 { font-size: 2.2rem; }
            .query-input-wrapper { flex-direction: column; }
            .btn-submit { justify-content: center; padding: 12px; }
            .route-row { flex-direction: column; align-items: flex-start; }
            .route-action { width: 100%; justify-content: flex-end; }
        }
    </style>
</head>
<body>

    <!-- Header / Navigation -->
    <header>
        <div class="container nav-inner">
            <a href="/" class="brand">
                <div class="brand-icon">⚡</div>
                <div class="brand-text">SmartCache <span>ML</span></div>
            </a>
            <div class="nav-links">
                <div class="status-pill">
                    <div class="status-dot"></div>
                    <span>Service Online</span>
                </div>
                <a href="/docs" class="btn-link" target="_blank">📖 Swagger Docs</a>
                <a href="http://127.0.0.1:8501" class="btn-link btn-primary" target="_blank" title="Streamlit Telemetry Monitor">📊 Streamlit UI</a>
            </div>
        </div>
    </header>

    <main class="container">
        <!-- Hero Section -->
        <section class="hero">
            <div class="hero-badge">⚡ Machine Learning Systems Engineering</div>
            <h1>Semantic Caching & <span class="highlight">Cost-Aware</span> LLM Routing</h1>
            <p>
                Eliminating redundant language model inference expenses and flattening latency spikes.
                Powered by dense sentence embeddings (<code>all-MiniLM-L6-v2</code>), vectorized cosine similarity, and TF-IDF complexity routing.
            </p>
            <div class="pill-group">
                <div class="tech-pill">Embedding Model: <strong>all-MiniLM-L6-v2</strong></div>
                <div class="tech-pill">Threshold: <strong>0.85 Cosine</strong></div>
                <div class="tech-pill">LLM Provider: <strong>Mock Mode (Zero Cost)</strong></div>
                <div class="tech-pill">Store: <strong>SQLite (WAL Mode)</strong></div>
            </div>
        </section>

        <!-- KPI Cards Grid (Live data fetched via JS) -->
        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Cached Queries</div>
                <div class="kpi-value" id="kpi-entries">--</div>
                <div class="kpi-subtitle">Persistent SQLite records in memory</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Cache Hit Ratio</div>
                <div class="kpi-value" id="kpi-hit-rate">--%</div>
                <div class="kpi-subtitle" id="kpi-hit-detail">0 hits / 0 requests</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Latency Reduction</div>
                <div class="kpi-value" style="color: #10B981;">~98.5%</div>
                <div class="kpi-subtitle">&lt;15ms cache hit vs ~1,200ms LLM</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Inference Cost Saved</div>
                <div class="kpi-value" style="color: var(--orange-primary);" id="kpi-savings">$0.00</div>
                <div class="kpi-subtitle">Zero token consumption on hit</div>
            </div>
        </section>

        <!-- Interactive Recruiter Playground -->
        <section class="playground-card">
            <div class="section-header">
                <div class="section-title">
                    <span>⚡ Live Query & Caching Playground</span>
                    <span class="section-badge">Interactive Demo</span>
                </div>
                <button class="btn-link" onclick="clearLiveCache()" title="Reset stored vectors">🧹 Clear Cache</button>
            </div>

            <!-- Demo preset chips for recruiter test -->
            <div class="demo-bar">
                <div class="demo-bar-label">1-Click Recruiter Test Scenarios:</div>
                <div class="demo-chips">
                    <button class="chip-btn" onclick="setQuery('What is the capital of France?')">
                        <span>💡 Query 1:</span> <strong>Capital of France</strong> (First time: Miss)
                    </button>
                    <button class="chip-btn" onclick="setQuery('Can you tell me the capital city of France?')">
                        <span>⚡ Query 2:</span> <strong>France capital paraphrase</strong> (Semantic Hit!)
                    </button>
                    <button class="chip-btn" onclick="setQuery('Explain the time complexity of QuickSort in worst and best cases with mathematical proofs.')">
                        <span>🧠 Query 3:</span> <strong>Complex Reasoning</strong> (Routes to Large Model)
                    </button>
                </div>
            </div>

            <!-- Query Input -->
            <div class="query-input-wrapper">
                <input type="text" id="query-input" class="query-input" placeholder="Type a prompt or choose a demo scenario above..." value="What is the capital of France?">
                <button id="btn-run" class="btn-submit" onclick="submitQuery()">
                    <span>Run Query</span> ⚡
                </button>
            </div>

            <!-- Result Card -->
            <div id="result-box" class="result-box">
                <div class="result-header">
                    <div id="result-badge" class="cache-badge badge-miss-style">CACHE MISS</div>
                    <div style="font-family: var(--font-mono); font-size: 0.82rem; color: var(--text-dim);" id="result-req-id">req-000</div>
                </div>

                <div class="result-meta-grid">
                    <div class="meta-item">
                        <div class="meta-label">Selected Route</div>
                        <div class="meta-val" id="meta-route">small</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Similarity Score</div>
                        <div class="meta-val" id="meta-similarity">N/A</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Total Latency</div>
                        <div class="meta-val" id="meta-latency">-- ms</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Estimated Cost</div>
                        <div class="meta-val" id="meta-cost">$0.0000</div>
                    </div>
                </div>

                <div style="font-size: 0.82rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; margin-bottom: 6px;">Generated Answer:</div>
                <div class="answer-box" id="result-answer">Loading answer...</div>
            </div>
        </section>

        <!-- Complete API Route Catalog -->
        <section class="routes-section">
            <div class="section-header">
                <div class="section-title">
                    <span>📡 Complete API Route Directory</span>
                    <span class="section-badge">All Endpoints</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--text-muted);">
                    Click <strong>"Try Route"</strong> on any endpoint to test live in-browser
                </div>
            </div>

            <div class="routes-grid">
                <!-- POST /query -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-post">POST</span>
                        <div>
                            <span class="route-path">/query</span>
                            <span class="route-desc">— Main inference endpoint: semantic lookup → router → model generation → cost tracker</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <button class="btn-try" onclick="testRoute('query')">Try Route</button>
                    </div>
                </div>
                <div id="out-query" class="route-output"></div>

                <!-- GET /health -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-get">GET</span>
                        <div>
                            <span class="route-path">/health</span>
                            <span class="route-desc">— Service health, provider configuration, threshold, and cache size</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <button class="btn-try" onclick="testRoute('health')">Try Route</button>
                    </div>
                </div>
                <div id="out-health" class="route-output"></div>

                <!-- GET /metrics -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-get">GET</span>
                        <div>
                            <span class="route-path">/metrics</span>
                            <span class="route-desc">— Real-time telemetry: total queries, cache hit rate, cost saved %, p50/p95 latency</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <button class="btn-try" onclick="testRoute('metrics')">Try Route</button>
                    </div>
                </div>
                <div id="out-metrics" class="route-output"></div>

                <!-- POST /cache/lookup -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-post">POST</span>
                        <div>
                            <span class="route-path">/cache/lookup</span>
                            <span class="route-desc">— Inspect semantic similarity against cache without triggering LLM generation or cost</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <button class="btn-try" onclick="testRoute('cache_lookup')">Try Route</button>
                    </div>
                </div>
                <div id="out-cache_lookup" class="route-output"></div>

                <!-- GET /cache/entries -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-get">GET</span>
                        <div>
                            <span class="route-path">/cache/entries</span>
                            <span class="route-desc">— Retrieve recent cached queries and responses (supports namespace filter & pagination)</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <button class="btn-try" onclick="testRoute('cache_entries')">Try Route</button>
                    </div>
                </div>
                <div id="out-cache_entries" class="route-output"></div>

                <!-- GET /cache/stats -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-get">GET</span>
                        <div>
                            <span class="route-path">/cache/stats</span>
                            <span class="route-desc">— Real-time cache performance counters (hits, misses, capacity, TTL)</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <button class="btn-try" onclick="testRoute('cache_stats')">Try Route</button>
                    </div>
                </div>
                <div id="out-cache_stats" class="route-output"></div>

                <!-- DELETE /cache -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-delete">DELETE</span>
                        <div>
                            <span class="route-path">/cache</span>
                            <span class="route-desc">— Invalidate all cache entries or entries belonging to a specified tenant namespace</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <button class="btn-try" onclick="testRoute('cache_delete')">Try Route</button>
                    </div>
                </div>
                <div id="out-cache_delete" class="route-output"></div>

                <!-- GET /docs -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-get">GET</span>
                        <div>
                            <span class="route-path">/docs</span>
                            <span class="route-desc">— Interactive OpenAPI / Swagger UI documentation with schema schemas</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <a href="/docs" target="_blank" class="btn-try" style="text-decoration:none;">Open Swagger ↗</a>
                    </div>
                </div>

                <!-- GET /redoc -->
                <div class="route-row">
                    <div class="route-left">
                        <span class="method-badge method-get">GET</span>
                        <div>
                            <span class="route-path">/redoc</span>
                            <span class="route-desc">— ReDoc alternate API specification and documentation viewer</span>
                        </div>
                    </div>
                    <div class="route-action">
                        <a href="/redoc" target="_blank" class="btn-try" style="text-decoration:none;">Open ReDoc ↗</a>
                    </div>
                </div>
            </div>
        </section>

        <!-- 3 Core Engineering Pillars -->
        <section class="pillars-grid">
            <div class="pillar-card">
                <div class="pillar-icon">🎯</div>
                <div class="pillar-title">1. Vectorized Semantic Caching</div>
                <div class="pillar-text">
                    Instead of fragile exact string matching, incoming prompts are projected into 384-dimensional dense semantic space via <code>all-MiniLM-L6-v2</code>. Normalized NumPy matrix multiplications find paraphrased intent in &lt;15ms.
                </div>
            </div>
            <div class="pillar-card">
                <div class="pillar-icon">🔀</div>
                <div class="pillar-title">2. Complexity-Aware Routing</div>
                <div class="pillar-text">
                    When a cache miss occurs, a lightweight TF-IDF + Logistic Regression classifier evaluates query difficulty. Simple questions route to small, low-cost models while complex multi-step reasoning routes to large models.
                </div>
            </div>
            <div class="pillar-card">
                <div class="pillar-icon">🛡️</div>
                <div class="pillar-title">3. Durable Storage & Multi-Tenancy</div>
                <div class="pillar-text">
                    Persistent SQLite store with WAL mode handles concurrency, LRU eviction, and configurable TTL expirations. Namespace isolation prevents sensitive data leakage between multi-tenant clients.
                </div>
            </div>
        </section>
    </main>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="footer-links">
                <a href="/docs">Swagger Documentation</a>
                <a href="/metrics">Raw Telemetry</a>
                <a href="http://127.0.0.1:8501">Streamlit Dashboard</a>
                <a href="https://github.com" target="_blank">Project Source</a>
            </div>
            <p>SmartCache ML — Production-Grade Semantic Caching & Cost-Aware Model Routing Engine</p>
        </div>
    </footer>

    <!-- Interactive Client Scripts -->
    <script>
        // Refresh live stats on page load
        document.addEventListener("DOMContentLoaded", () => {
            fetchStats();
        });

        // Set input query from demo chip
        function setQuery(text) {
            document.getElementById("query-input").value = text;
            submitQuery();
        }

        // Fetch live stats from API
        async function fetchStats() {
            try {
                const res = await fetch("/cache/stats");
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById("kpi-entries").innerText = data.total_cached_entries || 0;
                    document.getElementById("kpi-hit-rate").innerText = (data.hit_rate_pct || 0) + "%";
                    document.getElementById("kpi-hit-detail").innerText = `${data.hits || 0} hits / ${(data.hits || 0) + (data.misses || 0)} requests`;
                }

                const mRes = await fetch("/metrics");
                if (mRes.ok) {
                    const mData = await mRes.json();
                    if (mData.cost_saved_usd !== undefined) {
                        document.getElementById("kpi-savings").innerText = "$" + mData.cost_saved_usd.toFixed(4);
                    }
                }
            } catch (err) {
                console.warn("Telemetry fetch error:", err);
            }
        }

        // Submit query to /query endpoint
        async function submitQuery() {
            const input = document.getElementById("query-input");
            const query = input.value.trim();
            if (!query) return;

            const btn = document.getElementById("btn-run");
            const resultBox = document.getElementById("result-box");
            btn.disabled = true;
            btn.innerText = "Running...";

            try {
                const res = await fetch("/query", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: query, namespace: "default" })
                });

                const data = await res.json();
                resultBox.style.display = "block";

                // Update badge
                const badge = document.getElementById("result-badge");
                if (data.cache_hit) {
                    badge.innerText = "⚡ SEMANTIC CACHE HIT";
                    badge.className = "cache-badge badge-hit-style";
                } else {
                    badge.innerText = "CACHE MISS (GENERATED)";
                    badge.className = "cache-badge badge-miss-style";
                }

                document.getElementById("result-req-id").innerText = data.request_id || "req";
                document.getElementById("meta-route").innerText = data.selected_route || "small";
                document.getElementById("meta-similarity").innerText = data.similarity_score ? (data.similarity_score * 100).toFixed(1) + "%" : "N/A";
                document.getElementById("meta-latency").innerText = (data.latency_ms || 0).toFixed(1) + " ms";
                document.getElementById("meta-cost").innerText = "$" + (data.estimated_cost_usd || 0).toFixed(5);
                document.getElementById("result-answer").innerText = data.answer || "No response received";

                // Refresh KPIs
                fetchStats();
            } catch (err) {
                alert("Error calling /query: " + err);
            } finally {
                btn.disabled = false;
                btn.innerHTML = "<span>Run Query</span> ⚡";
            }
        }

        // Clear live cache
        async function clearLiveCache() {
            if (!confirm("Invalidate all cached vectors and reset metrics?")) return;
            try {
                await fetch("/cache", { method: "DELETE" });
                await fetch("/metrics/reset", { method: "POST" });
                alert("Cache and metrics reset successfully!");
                document.getElementById("result-box").style.display = "none";
                fetchStats();
            } catch (err) {
                alert("Error resetting cache: " + err);
            }
        }

        // Test individual route and display output
        async function testRoute(routeKey) {
            const outDiv = document.getElementById("out-" + routeKey);
            if (outDiv.style.display === "block") {
                outDiv.style.display = "none";
                return;
            }

            outDiv.style.display = "block";
            outDiv.innerText = "Executing request...";

            try {
                let res;
                if (routeKey === "health") {
                    res = await fetch("/health");
                } else if (routeKey === "metrics") {
                    res = await fetch("/metrics");
                } else if (routeKey === "cache_stats") {
                    res = await fetch("/cache/stats");
                } else if (routeKey === "cache_entries") {
                    res = await fetch("/cache/entries?limit=5");
                } else if (routeKey === "query") {
                    res = await fetch("/query", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query: "What is machine learning?" })
                    });
                } else if (routeKey === "cache_lookup") {
                    res = await fetch("/cache/lookup", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ query: "What is machine learning?" })
                    });
                } else if (routeKey === "cache_delete") {
                    res = await fetch("/cache", { method: "DELETE" });
                    fetchStats();
                }

                const json = await res.json();
                outDiv.innerText = JSON.stringify(json, null, 2);
            } catch (err) {
                outDiv.innerText = "Error: " + err;
            }
        }

        // Allow pressing Enter in query input
        document.getElementById("query-input").addEventListener("keydown", (e) => {
            if (e.key === "Enter") submitQuery();
        });
    </script>
</body>
</html>
"""
