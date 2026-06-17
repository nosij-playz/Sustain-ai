import os
import ollama
import re
import sys
import subprocess
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import hashlib


class AIPlotter:
    def __init__(self, model_name="qwen3-coder:480b-cloud"):
        self.model_name = model_name
        self.display_folder = "backend/display"
        self._counter = 0
        self.max_tokens = 16000  # Increased token limit for quality
        
        if not os.path.exists(self.display_folder):
            os.makedirs(self.display_folder)

    def _slugify(self, text, max_len=24):
        text = (text or "").strip().lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            text = "plot"
        return text[:max_len]

    def build_output_path(self, user_query):
        slug = self._slugify(user_query)
        ts = datetime.now().strftime("%y%m%d%H%M%S%f")
        self._counter = (self._counter + 1) % 10000
        filename = f"{slug}_{ts}_{self._counter:04d}.png"
        return os.path.join(self.display_folder, filename)

    def generate_code_from_text(self, user_query, output_path):
        """
        plotter reads the text, extracts the data, and writes the plotting code.
        """
        print(f"🧠 plotter is parsing data and designing the plot...")

        plot_plan = self._build_plot_plan(user_query)
        if plot_plan:
            print("📌 Plot plan:")
            for line in plot_plan:
                print(f"- {line}")
        
        prompt = (
            f"You are a professional Data Scientist. The user will provide a request that includes both "
            f"the type of graph they want and the data for that graph as text.\n\n"
            f"USER REQUEST AND DATA: {user_query}\n\n"
            f"YOUR TASK:\n"
            f"1. Extract the data points from the text provided.\n"
            f"2. Write Python code to visualize this data using matplotlib only.\n"
            f"3. Use plain Python lists or dictionaries for any tabular structure; do not import pandas.\n"
            f"4. Create a *multi-panel* analytics pack when possible (use subplots in a single figure).\n"
            f"   - Aim for multiple complementary diagrams (e.g., bar/line/scatter + distribution + correlation/heatmap when applicable).\n"
            f"   - If the data is too small/simple for some panels, include alternative views (sorted bar, annotated trend, summary table).\n"
            f"   - If the user asks for a flowchart/digraph, include a simple flowchart-style subplot using matplotlib patches/arrows.\n"
            f"   - Prefer clear labeled matplotlib bars and lines; avoid dependencies that may be blocked in managed Windows environments.\n"
            f"   - When labeling bars, call ax.bar_label(ax.containers[0], padding=3) and do not loop over rectangles.\n"
            f"5. Save the final figure to the exact file path in the variable OUTPUT_PATH (do not choose your own path).\n"
            f"   - Use: plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')\n"
            f"   - Do not call plt.show()\n"
            f"6. Return ONLY the raw Python code. Do NOT provide explanations, do NOT use markdown backticks (like ```python), and do NOT include comments or hashtags.\n"
            f"7. Ensure matplotlib.pyplot is imported. Use numpy only if it helps; do not import pandas or seaborn.\n\n"
            f"PLOT PLAN (use these panels when possible):\n"
            f"{chr(10).join(f'- {line}' for line in plot_plan) if plot_plan else '- Primary plot + 1 alternative view'}\n\n"
            f"OUTPUT_PATH = {output_path!r}"
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            code = response['message']['content']
            return self._clean_code(code)
        except Exception as e:
            print(f"Error generating code: {e}")
            return None

    def _generate_code_for_plan(self, user_query, plan_item, output_path, chart_type, disallow_types):
        prompt = (
            "You are a professional Data Scientist. Generate a single plot for the plan item below.\n\n"
            f"USER REQUEST AND DATA: {user_query}\n\n"
            f"PLOT PLAN ITEM: {plan_item}\n\n"
            f"REQUIRED CHART TYPE: {chart_type}.\n"
            f"DO NOT USE these chart types: {', '.join(disallow_types) if disallow_types else 'None'}.\n\n"
            "REQUIREMENTS:\n"
            "1. Extract the data points from the text provided.\n"
            "2. Write Python code to visualize this data using matplotlib only.\n"
            "3. Use plain Python lists or dictionaries for any tabular structure; do not import pandas.\n"
            "4. Create a single, focused plot for this plan item (no subplots).\n"
            "5. If the extracted data is too small or identical across plots, derive at least one extra series "
            "   (e.g., normalized index, rolling average, ranking, delta, or risk score) so each plot is unique.\n"
            "6. Use a different encoding (orientation, aggregation, or axis) than other plots when possible.\n"
            "5. Save the final figure to OUTPUT_PATH (do not choose your own path).\n"
            "   - Use: plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches='tight')\n"
            "   - Do not call plt.show()\n"
            "6. Return ONLY the raw Python code. No markdown, no explanations, no hashtags or comments.\n"
            "7. Ensure matplotlib.pyplot is imported. Use numpy only if it helps; do not import pandas or seaborn.\n"
            "9. When labeling bars, call ax.bar_label(ax.containers[0], padding=3) and do not loop over rectangles.\n\n"
            f"OUTPUT_PATH = {output_path!r}"
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                stream=False
            )
            code = response['message']['content']
            return self._clean_code(code)
        except Exception as e:
            print(f"Error generating code: {e}")
            return None

    def _hash_file(self, file_path):
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception:
            return None

    def generate_plots_from_text(self, user_query):
        plot_plan = self._build_plot_plan(user_query)
        if plot_plan:
            print("📌 Plot plan:")
            for line in plot_plan:
                print(f"- {line}")

        # Fallback to a single multi-panel plot when no plan is available.
        if not plot_plan:
            output_path = self.build_output_path(user_query)
            generated_code = self.generate_code_from_text(user_query, output_path)
            if not generated_code:
                return []
            success, saved_path = self._execute_code_subprocess(generated_code, output_path, context_text=user_query)
            return [saved_path] if success and saved_path else []

        unique_plan = []
        seen = set()
        for item in plot_plan:
            key = (item or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_plan.append(item)

        chart_types = [
            "bar",
            "horizontal bar",
            "line",
            "scatter",
            "box",
            "violin",
            "heatmap",
            "area",
            "lollipop",
        ]

        plan_with_types = []
        for idx, item in enumerate(unique_plan):
            chart_type = chart_types[idx % len(chart_types)]
            disallow = [t for t in chart_types if t != chart_type]
            plan_with_types.append((item, chart_type, disallow))

        def run_item(plan_tuple):
            plan_item, chart_type, disallow = plan_tuple
            output_path = self.build_output_path(f"{plan_item}")
            code = self._generate_code_for_plan(user_query, plan_item, output_path, chart_type, disallow)
            if not code:
                return None
            success, saved_path = self._execute_code_subprocess(
                code,
                output_path,
                context_text=f"{user_query}\n{plan_item}",
                chart_type=chart_type,
            )
            return saved_path if success else None

        max_workers = min(4, len(plan_with_types))
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for saved_path in ex.map(run_item, plan_with_types):
                if saved_path:
                    results.append(saved_path)

        # De-duplicate identical images by content hash.
        deduped = []
        seen_hashes = set()
        for path in results:
            file_hash = self._hash_file(path)
            if not file_hash or file_hash in seen_hashes:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                continue
            seen_hashes.add(file_hash)
            deduped.append(path)
        return deduped

    def _clean_code(self, code):
        """
        Removes any accidental markdown, conversational text, and comment lines.
        """
        # Remove markdown code blocks if the AI added them
        code = re.sub(r"```python\n?", "", code)
        code = re.sub(r"```", "", code)
        # Remove any lines that start with # if the AI added "Here is the code..." or comment lines
        lines = code.split('\n')
        cleaned_lines = [line for line in lines if not line.strip().startswith('Here is') and not line.strip().startswith('#')]
        return "\n".join(cleaned_lines).strip()

    def _build_plot_plan(self, user_query):
        prompt = (
            "You are a senior data scientist. Provide a concise plot plan (3-6 items) based on the user's data. "
            "Return only short bullet points without code. Be specific and actionable. No hashtags or special characters.\n\n"
            f"USER REQUEST AND DATA: {user_query}"
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            text = (response.get("message", {}) or {}).get("content", "")
            # Remove hashtags and special characters from lines
            lines = [line.strip("-• \t#") for line in text.split("\n") if line.strip()]
            return [line for line in lines if line and not line.startswith("#")]
        except Exception:
            return []

    def _patch_plot_code(self, code: str) -> str:
        if not code:
            return code

        # Fix seaborn API calls
        code = code.replace("sns.set_theme_style(", "sns.set_style(")
        code = code.replace("sns.set_theme(", "plt.style.use(")

        # Fix common bar_label misuse on Rectangle objects.
        pattern = re.compile(
            r"^([ \t]*)for\s+\w+\s+in\s+([A-Za-z0-9_\.]+)\.containers\[0\]:\s*$\n"
            r"\1\s*\2\.bar_label\(\w+,\s*padding\s*=\s*([0-9\.]+)\)\s*$",
            re.MULTILINE,
        )
        code = pattern.sub(r"\1\2.bar_label(\2.containers[0], padding=\3)", code)

        # Patch seaborn barplot palette deprecation by adding hue + legend=False when possible.
        lines = code.split("\n")
        patched_lines = []
        for line in lines:
            if "sns.barplot(" in line and "palette=" in line and "hue=" not in line and "x=" in line:
                match = re.search(r"x\s*=\s*([^,\)]+)", line)
                if match:
                    x_value = match.group(1).strip()
                    if ")" in line:
                        line = line.replace(")", f", hue={x_value}, legend=False)")
            patched_lines.append(line)
        return "\n".join(patched_lines)

    def _extract_numbers(self, text: str):
        if not text:
            return []
        values = []
        for match in re.findall(r"-?\d+(?:\.\d+)?", text):
            try:
                values.append(float(match) if "." in match else int(match))
            except Exception:
                continue
        return values

    def _looks_like_blocked_dependency_error(self, stderr: str) -> bool:
        if not stderr:
            return False
        lowered = stderr.lower()
        return any(token in lowered for token in ["dll load failed", "blocked this file", "importerror", "pandas", "seaborn"])

    def _safe_title(self, context_text: str, chart_type: str = "") -> str:
        text = (context_text or "").strip()
        if not text:
            return (chart_type or "waste intelligence snapshot").replace("_", " ").title()
        compact = re.sub(r"\s+", " ", text)
        return compact[:72].strip().title()

    def _generate_safe_fallback_plot(self, output_path, context_text="", chart_type="", title=""):
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as e:
            print(f"Fallback plotting unavailable: {e}")
            return False, None

        numbers = self._extract_numbers(context_text)
        kind = (chart_type or "").lower()
        if not numbers:
            seed = sum(ord(ch) for ch in (context_text or "")) or 1
            numbers = [((seed // (idx + 1)) % 80) + 15 for idx in range(5)]

        labels = [f"Point {idx + 1}" for idx in range(len(numbers))]
        title_text = title or self._safe_title(context_text, chart_type)

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 6))

        if "line" in kind or "trend" in (context_text or "").lower():
            x_values = np.arange(1, len(numbers) + 1)
            ax.plot(x_values, numbers, marker="o", linewidth=2.5, color="#2E86AB")
            ax.fill_between(x_values, numbers, alpha=0.15, color="#2E86AB")
            ax.set_xticks(x_values)
            ax.set_xticklabels(labels, rotation=30, ha="right")
        elif "scatter" in kind and len(numbers) > 1:
            x_values = np.arange(1, len(numbers) + 1)
            bubble_sizes = np.clip(np.asarray(numbers, dtype=float) * 16, 40, 320)
            scatter = ax.scatter(x_values, numbers, s=bubble_sizes, c=numbers, cmap="viridis", edgecolors="black")
            fig.colorbar(scatter, ax=ax, label="Relative value")
            ax.plot(x_values, numbers, alpha=0.25, color="#2E86AB")
            ax.set_xticks(x_values)
            ax.set_xticklabels(labels, rotation=30, ha="right")
        elif "heatmap" in kind:
            matrix_values = numbers[:9]
            while len(matrix_values) < 9:
                matrix_values.append(matrix_values[-1] if matrix_values else 0)
            matrix = np.asarray(matrix_values, dtype=float).reshape(3, 3)
            image = ax.imshow(matrix, cmap="viridis")
            fig.colorbar(image, ax=ax, label="Intensity")
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(["A", "B", "C"])
            ax.set_yticklabels(["1", "2", "3"])
        else:
            positions = np.arange(len(numbers))
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(numbers)))
            bars = ax.bar(positions, numbers, color=colors)
            ax.bar_label(bars, padding=3, fmt="%.0f")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right")

        ax.set_title(title_text)
        ax.set_ylabel("Value")
        ax.set_xlabel("Category")
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return True, output_path

    def execute_code(self, code, output_path):
        """
        Executes the generated Python code.
        """
        if not code or not output_path:
            return False, None

        print("\n🚀 Executing generated code...")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Execute the code in a local scope and provide OUTPUT_PATH
            exec_globals = {
                "__name__": "__main__",
                "OUTPUT_PATH": output_path,
            }
            patched_code = self._patch_plot_code(code)
            exec(patched_code, exec_globals)

            if os.path.exists(output_path):
                print(f"✅ Plot successfully generated and saved: {output_path}")
                return True, output_path

            print("❌ Plot code ran, but no image was saved to OUTPUT_PATH.")
            return False, None
        except Exception as e:
            print(f"❌ Execution Error: {e}")
            print("\n--- Generated Code That Failed ---\n")
            print(code)
            return False, None

    def _execute_code_subprocess(self, code, output_path, context_text="", chart_type=""):
        if not code or not output_path:
            return False, None

        print("\n🚀 Executing generated code...")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            patched_code = self._patch_plot_code(code)
            payload = f"OUTPUT_PATH = {output_path!r}\n" + patched_code

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(payload)
                tmp_path = tmp.name

            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                check=False,
            )

            if proc.returncode != 0:
                err = proc.stderr.strip() or "Unknown error"
                print(f"❌ Execution Error: {err}")
                print("\n--- Generated Code That Failed ---\n")
                print(code)
                if self._looks_like_blocked_dependency_error(err):
                    print("↩️ Falling back to a pure-matplotlib plot because the generated code hit a blocked dependency.")
                    return self._generate_safe_fallback_plot(
                        output_path,
                        context_text=context_text,
                        chart_type=chart_type,
                    )
                return False, None

            if os.path.exists(output_path):
                print(f"✅ Plot successfully generated and saved: {output_path}")
                return True, output_path

            print("❌ Plot code ran, but no image was saved to OUTPUT_PATH.")
            return False, None
        finally:
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

