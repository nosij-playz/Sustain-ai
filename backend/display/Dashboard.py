import os
import json
import webbrowser
import re

class DashboardModule:
    def __init__(self, model_name=None):
        # model_name is kept only for backward compatibility; dashboard rendering is deterministic.
        self.model_name = model_name
        self.base_template_path = "backend/interface/base.html"
        self.output_file = "backend/interface/dashboard.html"
        self.system_name = os.getenv("SUSTAINAI_SYSTEM_NAME", "SustainAi")
        self._opened = False

    def _normalize_image_path(self, path: str) -> str:
        """
        Fixes duplicate directory mutations and normalizes the image paths
        to render correctly from inside backend/interface/dashboard.html
        """
        if not path:
            return ""
        
        # If it's already a full URL, return it as is.
        if path.startswith("http://") or path.startswith("https://") or path.startswith("data:image"):
            return path
            
        # 1. Normalize all variations of separation slashes to forward slashes
        clean_path = path.replace("\\", "/")
        
        # 2. Extract out just the final filename + extensions to bypass 
        # any broken nested folder strings (e.g. backend/backend/display/)
        filename = os.path.basename(clean_path)
        
        # 3. Since the dashboard file resides in 'backend/interface/', 
        # going up one folder ('../') drops it directly into 'backend/',
        # allowing it to access 'display/filename' perfectly.
        return f"../display/{filename}"

    def _build_dashboard_prompt(self, payload: dict) -> str:
        header = payload.get("header") or {}
        section_counts = {
            "hero": len(payload.get("hero") or []),
            "atmosphere": len(payload.get("atmosphere") or []),
            "soil": len(payload.get("soil") or []),
            "weather": len(payload.get("weather") or []),
            "risks": len(payload.get("risks") or []),
            "insights": len(payload.get("insights") or []),
            "research": len(payload.get("research") or []),
            "visuals": len(payload.get("visuals") or []),
            "diagnostics": len(payload.get("diagnostics") or []),
        }

        return (
            "You are designing a premium, executive-grade environmental intelligence dashboard. "
            "Generate a clean HTML dashboard that is elegant, calm, and highly structured.\n\n"
            "GOAL:\n"
            "Turn the incoming payload into a polished decision-support dashboard for leaders. "
            "The output must feel premium, readable, and data-first.\n\n"
            "VISUAL DIRECTION:\n"
            "- Use a dark, luxury glassmorphism theme with strong spacing, soft gradients, and restrained highlights.\n"
            "- Prefer rounded cards, subtle borders, and clear hierarchy over dense panels.\n"
            "- Avoid repetitive blocks, duplicated titles, and oversized prose.\n"
            "- Keep the top hero compact and authoritative.\n\n"
            "LAYOUT RULES:\n"
            "- Start with one compact header showing the system name, location, refresh time, and cache state.\n"
            "- If hero metrics exist, show them as 3-4 high-value KPI cards.\n"
            "- If visuals exist, show only the most relevant ones first and keep each card concise.\n"
            "- If environmental metrics exist, group them into atmosphere, soil, and weather with short labeled cards.\n"
            "- Only show the risk section when there is real scored data. Do not invent default risk values.\n"
            "- Only show the sidebar when it contains actual content. If it is empty, let the main area expand fully.\n"
            "- Use research and diagnostics sparingly and keep them compact; do not repeat the same data twice.\n"
            "- Show a clean empty-state message when no live data has arrived.\n\n"
            "CONTENT RULES:\n"
            "- Keep titles short.\n"
            "- Summaries should be executive-friendly: one or two lines, not raw model output dumps.\n"
            "- Prefer bullet-like microcopy and concise captions over paragraphs.\n"
            "- Hide or collapse low-value sections when they have no meaningful content.\n"
            "- The dashboard must adapt to the incoming payload instead of rendering a fixed template.\n\n"
            "DATA SNAPSHOT:\n"
            f"System: {header.get('system_name') or self.system_name}\n"
            f"Location: {header.get('location') or 'Unknown'}\n"
            f"Updated: {header.get('updated_at') or '--'}\n"
            f"Cache: {header.get('cache_status') or 'MISS'}\n"
            f"Counts: {section_counts}\n\n"
            "RETURN RULES:\n"
            "- Return only the final dashboard HTML body or a JSON-ready dashboard spec if requested.\n"
            "- Keep the response structured, polished, and free from duplicated blocks.\n"
            "- The final experience should feel like a professional command center, not a raw report." 
        )

    def _read_base_template(self):
        try:
            with open(self.base_template_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print("❌ Error: base.html not found. Please ensure it exists in the interface folder.")
            return None

    def _escape(self, value):
        if value is None:
            return ""
        return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _shorten(self, value, limit=240):
        text = re.sub(r"\s+", " ", self._escape(value)).strip()
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)].rstrip() + "..."

    def _render_read_more(self, title: str, full_text, teaser_limit: int = 220, panel_class: str = "", title_class: str = "") -> str:
        teaser = self._shorten(full_text, teaser_limit)
        full = self._escape(full_text) if full_text is not None else ""
        panel = panel_class or "glass-premium p-4 rounded-3xl border border-slate-800 shadow-xl"
        label_class = title_class or "text-[11px] text-slate-400 uppercase tracking-[0.28em] line-clamp-1"
        return (
            f"<details class='{panel} group overflow-hidden'>"
            f"<summary class='flex items-start justify-between gap-4 cursor-pointer list-none'>"
            f"<span class='{label_class}'>{self._escape(title)}</span>"
            "<span class='readmore-chip shrink-0'>Read more</span>"
            "</summary>"
            f"<div class='dashboard-snippet mt-3 text-sm text-slate-300 leading-relaxed'>{teaser}</div>"
            f"<div class='readmore-full mt-4 text-sm text-slate-200 leading-7'>{full}</div>"
            "</details>"
        )

    def _render_kpi_card(self, label, value):
        return (
            "<div class='glass-premium p-6 rounded-2xl border border-slate-800 shadow-xl'>"
            f"<p class='text-xs text-slate-500 uppercase tracking-wider mb-2'>{self._escape(label)}</p>"
            f"<p class='text-2xl font-medium text-white'>{self._escape(value) if value not in (None, '') else 'N/A'}</p>"
            "</div>"
        )

    def _render_dashboard_body(self, payload: dict) -> str:
        header = payload.get("header") or {}
        hero = payload.get("hero") or []
        atmosphere = payload.get("atmosphere") or []
        soil = payload.get("soil") or []
        weather = payload.get("weather") or []
        risks = payload.get("risks") or []
        insights = payload.get("insights") or []
        research = payload.get("research") or []
        visuals = payload.get("visuals") or []
        diagnostics = payload.get("diagnostics") or []
        tables = payload.get("tables") or []

        subtitle = header.get("subtitle") or "Live Environmental Waste Intelligence"
        location = header.get("location") or "Unknown"
        updated_at = header.get("updated_at") or "--"
        cache_status = header.get("cache_status") or "MISS"

        def has_items(items):
            return isinstance(items, list) and any(item not in (None, {}, [], "") for item in items)

        def section_card(title, content_html):
            return (
                "<div class='glass-premium p-6 rounded-[28px] border border-slate-800 shadow-xl space-y-5'>"
                f"<h2 class='text-[11px] uppercase tracking-[0.32em] text-slate-400 font-semibold'>{self._escape(title)}</h2>"
                f"{content_html}"
                "</div>"
            )

        def render_metric_cards(items):
            if not items:
                return "<div class='text-slate-500 text-sm italic'>No metrics available.</div>"
            cards = []
            for item in items:
                label = item.get("label") or "Metric"
                value = item.get("value") or "N/A"
                cards.append(
                    "<div class='glass-premium p-4 rounded-2xl border border-slate-800 shadow-xl'>"
                    f"<p class='text-xs text-slate-500 uppercase tracking-wider mb-2'>{self._escape(label)}</p>"
                    f"<p class='text-lg font-semibold text-white'>{self._escape(value)}</p>"
                    "</div>"
                )
            return "<div class='grid grid-cols-1 md:grid-cols-3 gap-4'>" + "".join(cards) + "</div>"

        def render_hero(items):
            if not items:
                return ""
            cards = []
            for item in items:
                label = item.get("label") or "KPI"
                value = item.get("value") or "N/A"
                cards.append(
                    "<div class='glass-premium p-5 rounded-2xl border border-slate-800 shadow-xl gradient-card'>"
                    f"<p class='text-xs text-slate-400 uppercase tracking-widest mb-2'>{self._escape(label)}</p>"
                    f"<p class='text-3xl font-semibold text-white number-glow'>{self._escape(value)}</p>"
                    "</div>"
                )
            return "<div class='grid grid-cols-1 md:grid-cols-4 gap-4'>" + "".join(cards) + "</div>"

        def render_risks(items):
            if not items:
                return "<div class='text-slate-500 text-sm italic'>No risk scoring available.</div>"
            cards = []
            for item in items:
                label = item.get("label") or "Risk"
                score = item.get("score")
                level = item.get("level") or "LOW"
                note = item.get("note") or ""
                level_class = {
                    "LOW": "text-emerald-300",
                    "MODERATE": "text-amber-300",
                    "HIGH": "text-orange-300",
                    "SEVERE": "text-red-400",
                }.get(level, "text-slate-300")

                cards.append(
                    "<div class='glass-premium p-4 rounded-2xl border border-slate-800 shadow-xl space-y-2'>"
                    f"<div class='flex items-center justify-between'><p class='text-sm font-semibold text-white'>{self._escape(label)}</p>"
                    f"<span class='text-xs font-semibold uppercase {level_class}'>{self._escape(level)}</span></div>"
                    f"<p class='text-xl font-semibold text-white'>{self._escape(score)}</p>"
                    f"<p class='text-xs text-slate-400'>{self._escape(note)}</p>"
                    "</div>"
                )
            return "<div class='grid grid-cols-1 md:grid-cols-2 gap-4'>" + "".join(cards) + "</div>"

        def render_visuals(items):
            if not items:
                return "<div class='text-slate-500 text-sm italic'>No visuals available.</div>"
            cards = []
            for item in items:
                title = item.get("title") or "Visual"
                image = item.get("image")
                description = item.get("description") or ""
                if not image:
                    continue
                
                # Dynamic normalization to enforce standard relative rendering path mapping
                normalized_img = self._normalize_image_path(image)

                # Check if it's a local file and if it exists
                if not normalized_img.startswith("http") and not normalized_img.startswith("data:image"):
                    # Reconstruct absolute path to check existence
                    check_path = os.path.abspath(os.path.join(os.path.dirname(self.output_file), normalized_img))
                    if not os.path.exists(check_path):
                        continue # Skip this card if the image is missing

                cards.append(
                    "<div class='glass-premium p-4 rounded-3xl border border-slate-800 shadow-xl space-y-3'>"
                    f"<p class='text-[11px] text-slate-400 uppercase tracking-[0.28em] line-clamp-1'>{self._escape(title)}</p>"
                    f"<img src='{self._escape(normalized_img)}' class='rounded-xl border border-slate-700 w-full h-auto shadow-lg' alt='{self._escape(title)}' />"
                    f"{self._render_read_more('Read full visual analysis', description, teaser_limit=320, panel_class='rounded-2xl border border-slate-800/70 bg-slate-950/30 p-4 shadow-inner') }"
                    "</div>"
                )
            return "<div class='grid grid-cols-1 md:grid-cols-2 gap-4'>" + "".join(cards) + "</div>"

        def render_insights(items):
            if not items:
                return "<div class='text-slate-500 text-sm italic'>No AI insights yet.</div>"
            cards = []
            for insight in items:
                cards.append(self._render_read_more("AI Strategic Insight", insight, teaser_limit=280, panel_class="glass-premium p-4 rounded-3xl border border-slate-800 shadow-xl"))
            return "<div class='grid grid-cols-1 md:grid-cols-2 gap-4'>" + "".join(cards) + "</div>"

        def render_research(items):
            if not items:
                return "<div class='text-slate-500 text-sm italic'>No research intelligence loaded.</div>"
            cards = []
            for item in items:
                title = item.get("title") or "Research"
                summary = item.get("summary") or ""
                link = item.get("link")
                image = item.get("image")
                link_html = ""
                if link:
                    link_html = f"<a class='text-xs text-cyan-300 hover:text-cyan-200' href='{self._escape(link)}' target='_blank' rel='noreferrer'>Read More</a>"
                image_html = ""
                if image:
                    normalized_res_img = self._normalize_image_path(image)
                    image_html = (
                        f"<img src='{self._escape(normalized_res_img)}' alt='{self._escape(title)}' "
                        "class='rounded-xl border border-slate-800 w-full h-32 object-cover' />"
                    )
                cards.append(
                    "<div class='glass-premium p-5 rounded-3xl border border-slate-800 shadow-xl space-y-3'>"
                    f"<div class='flex items-center justify-between gap-3'><p class='text-sm font-semibold text-white'>{self._shorten(title, 72)}</p>{link_html}</div>"
                    f"{image_html}"
                    f"{self._render_read_more('Read full research summary', summary, teaser_limit=240, panel_class='rounded-2xl border border-slate-800/70 bg-slate-950/30 p-4 shadow-inner') }"
                    "</div>"
                )
            return "<div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6'>" + "".join(cards) + "</div>"

        def render_tables(items):
            if not items:
                return "<div class='text-slate-500 text-sm italic'>No summary tables available.</div>"

            cards = []
            for index, item in enumerate(items, start=1):
                title = item.get("title") or f"Summary Table {index}"
                headers = item.get("headers") or []
                rows = item.get("rows") or []
                column_count = max([len(headers)] + [len(row) for row in rows] if rows else [len(headers), 0])
                if column_count <= 0:
                    continue

                normalized_headers = list(headers) + [f"Column {i + 1}" for i in range(len(headers), column_count)]
                normalized_rows = [list(row) + [""] * max(0, column_count - len(row)) for row in rows]

                thead = "".join(
                    f"<th class='px-4 py-3 text-left text-[11px] uppercase tracking-[0.28em] text-slate-400 border-b border-slate-800 bg-slate-950/50'>{self._escape(col)}</th>"
                    for col in normalized_headers[:column_count]
                )
                tbody = "".join(
                    "<tr class='border-b border-slate-800/60 hover:bg-slate-900/60 transition-colors'>"
                    + "".join(
                        f"<td class='px-4 py-3 text-sm text-slate-200 align-top'>{self._escape(cell)}</td>"
                        for cell in row[:column_count]
                    )
                    + "</tr>"
                    for row in normalized_rows
                )

                cards.append(
                    "<details class='glass-premium p-5 rounded-3xl border border-slate-800 shadow-xl group overflow-hidden'>"
                    "<summary class='flex items-start justify-between gap-4 cursor-pointer list-none'>"
                    "<div class='space-y-1'>"
                    "<p class='text-[11px] text-slate-400 uppercase tracking-[0.28em]'>Structured Table</p>"
                    f"<p class='text-sm font-semibold text-white'>{self._shorten(title, 72)}</p>"
                    "</div>"
                    "<span class='readmore-chip shrink-0'>View table</span>"
                    "</summary>"
                    "<div class='mt-4 overflow-x-auto rounded-2xl border border-slate-800/80'>"
                    "<table class='dashboard-table min-w-full border-collapse'>"
                    f"<thead><tr>{thead}</tr></thead>"
                    f"<tbody>{tbody}</tbody>"
                    "</table></div>"
                    "</details>"
                )

            if not cards:
                return "<div class='text-slate-500 text-sm italic'>No valid summary tables found.</div>"

            return "<div class='grid grid-cols-1 gap-4'>" + "".join(cards) + "</div>"

        def render_diagnostics(items):
            if not items:
                return "<div class='text-slate-500 text-sm italic'>No diagnostics available.</div>"
            rows = []
            for item in items:
                rows.append(
                    "<div class='flex items-center justify-between border-b border-slate-800/70 py-2'>"
                    f"<span class='text-xs text-slate-400'>{self._escape(item.get('label'))}</span>"
                    f"<span class='text-xs text-slate-200'>{self._escape(item.get('value'))}</span>"
                    "</div>"
                )
            return "<div class='space-y-1'>" + "".join(rows) + "</div>"

        main_sections = []
        if has_items(hero):
            main_sections.append(section_card("Executive Overview", render_hero(hero)))

        if has_items(visuals):
            main_sections.append(section_card("Visual Intelligence", render_visuals(visuals)))

        environmental_blocks = []
        if has_items(atmosphere):
            environmental_blocks.append(
                "<div><p class='text-xs uppercase tracking-wider text-slate-500 mb-2'>Atmosphere</p>"
                f"{render_metric_cards(atmosphere)}</div>"
            )
        if has_items(soil):
            environmental_blocks.append(
                "<div><p class='text-xs uppercase tracking-wider text-slate-500 mb-2'>Soil Intelligence</p>"
                f"{render_metric_cards(soil)}</div>"
            )
        if has_items(weather):
            environmental_blocks.append(
                "<div><p class='text-xs uppercase tracking-wider text-slate-500 mb-2'>Weather Events</p>"
                f"{render_metric_cards(weather)}</div>"
            )
        if environmental_blocks:
            main_sections.append(
                section_card(
                    "Environmental Breakdown",
                    "<div class='space-y-4'>" + "".join(environmental_blocks) + "</div>",
                )
            )

        sidebar_sections = []
        if has_items(risks):
            sidebar_sections.append(section_card("Waste Risk Assessment", render_risks(risks)))

        if has_items(insights):
            sidebar_sections.append(section_card("AI Strategic Recommendations", render_insights(insights)))

        if has_items(research):
            main_sections.append(section_card("Latest Environmental Intelligence", render_research(research)))

        if has_items(tables):
            main_sections.append(section_card("Summary Tables", render_tables(tables)))

        if has_items(diagnostics):
            sidebar_sections.append(
                "<details class='glass-premium p-6 rounded-3xl border border-slate-800 shadow-xl'>"
                "<summary class='text-sm uppercase tracking-widest text-slate-400 cursor-pointer'>Advanced Diagnostics</summary>"
                f"<div class='mt-4'>{render_diagnostics(diagnostics)}</div>"
                "</details>"
            )

        if not (main_sections or sidebar_sections):
            main_sections.append(
                section_card(
                    "Dashboard",
                    "<div class='text-slate-400 text-sm'>No live data has been loaded yet. Run a full data report to populate KPIs, visuals, and insights.</div>",
                )
            )

        sidebar_html = ""
        content_grid_class = "grid grid-cols-1 gap-6"
        if sidebar_sections:
            sidebar_html = (
                "<div class='space-y-6 xl:sticky xl:top-6'>"
                f"{''.join(sidebar_sections)}"
                "</div>"
            )
            content_grid_class = "grid grid-cols-1 xl:grid-cols-[minmax(0,1.6fr)_minmax(340px,0.9fr)] gap-6 items-start"

        return (
            "<div class='space-y-8 font-sans text-slate-200'>"
            "<section class='glass-premium p-6 md:p-7 rounded-[30px] border border-slate-800 shadow-xl space-y-5'>"
            "<div class='flex flex-wrap items-start justify-between gap-5'>"
            "<div class='space-y-3'>"
            f"<p class='text-[11px] uppercase tracking-[0.35em] text-slate-500'>{self._escape(subtitle)}</p>"
            f"<h1 class='text-2xl md:text-4xl font-semibold text-white leading-tight'>Operational Snapshot</h1>"
            f"<p class='max-w-2xl text-sm md:text-base text-slate-400 leading-7'>{self._escape(location)} · Updated {self._escape(updated_at)} · Cache {self._escape(cache_status)}</p>"
            "</div>"
            "<div class='grid grid-cols-2 gap-3 shrink-0'>"
            "<div class='px-4 py-3 rounded-2xl border border-emerald-500/20 bg-emerald-500/10 text-emerald-300 text-xs uppercase tracking-[0.24em] text-center'>System Online</div>"
            f"<div class='px-4 py-3 rounded-2xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-300 text-xs uppercase tracking-[0.24em] text-center'>Cache {self._escape(cache_status)}</div>"
            "</div>"
            "</div>"
            f"{render_hero(hero)}"
            "</section>"

            f"<section class='{content_grid_class}'>"
            "<div class='space-y-6'>"
            f"{''.join(main_sections)}"
            "</div>"
            f"{sidebar_html}"
            "</section>"

            "<div class='flex justify-end pt-2'>"
            "<button onclick='window.close()' class='px-4 py-2 text-xs uppercase tracking-[0.24em] bg-slate-800/80 hover:bg-red-900/40 border border-slate-700 transition-all duration-300 rounded-full text-slate-300'>Close Dashboard</button>"
            "</div>"
            "</div>"
        )

    def generate_dashboard(self, data_json):
        """Transforms agent data into a high-end HTML dashboard (deterministic, no LLM)."""
        print("🎨 Orchestrating real-time data into luxury dashboard...")

        payload = data_json
        if isinstance(data_json, list):
            payload = {"items": data_json}

        if not isinstance(payload, dict):
            payload = {"items": [str(payload)]}

        html_content = self._render_dashboard_body(payload)

        base_html = self._read_base_template()
        if not base_html:
            return False

        final_html = (
            base_html
            .replace("{{CONTENT}}", html_content)
            .replace("{{DASHBOARD_BODY}}", html_content)
            .replace("{{SYSTEM_NAME}}", self._escape(self.system_name))
        )

        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(final_html)
            return True
        except Exception as e:
            print(f"Error generating dashboard: {e}")
            return False

    def open_dashboard(self):
        """Opens the generated dashboard in a new window."""
        path = os.path.abspath(self.output_file)
        if not self._opened:
            self._opened = True
            print(f"🚀 Dashboard opened: {path}")

    def close_dashboard(self):
        print("ℹ️ Please use the 'Close Dashboard' button within the browser window.")