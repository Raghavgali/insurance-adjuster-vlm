from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EXPORT_DIR = ROOT / "wandb_exports"
PLOTS_DIR = ROOT / "plots"


RUN_SPECS = {
    "controlled_g1_ebs6_acc6_fpbf16": {"label": "Controlled", "gpus": 1},
    "controlled_g2_ebs6_acc3_fpbf16": {"label": "Controlled", "gpus": 2},
    "real_g1_ebs8_acc8_fpbf16": {"label": "Real", "gpus": 1},
    "real_g2_ebs16_acc8_fpbf16": {"label": "Real", "gpus": 2},
}


def load_runs() -> dict[str, dict[str, str]]:
    runs: dict[str, dict[str, str]] = {}
    for path in sorted(EXPORT_DIR.glob("wandb_export_*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = row.get("Name", "")
                if name in RUN_SPECS and row.get("train/samples_per_second_global"):
                    runs[name] = row
    return runs


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; fill: #14213d; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".subtitle { font-size: 12px; fill: #52607a; }",
        ".axis { font-size: 12px; }",
        ".label { font-size: 12px; font-weight: 700; }",
        ".value { font-size: 12px; font-weight: 700; }",
        ".small { font-size: 11px; fill: #52607a; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#fffdf6"/>',
    ]


def write_svg(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def build_throughput_plot(runs: dict[str, dict[str, str]]) -> None:
    width, height = 980, 540
    left, right, top, bottom = 90, 40, 90, 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_y = 0.8

    groups = [
        {
            "name": "Real",
            "bars": [
                ("1 GPU", float(runs["real_g1_ebs8_acc8_fpbf16"]["train/samples_per_second_global"]), "#355070"),
                ("2 GPU", float(runs["real_g2_ebs16_acc8_fpbf16"]["train/samples_per_second_global"]), "#eaac8b"),
            ],
        },
        {
            "name": "Controlled",
            "bars": [
                ("1 GPU", float(runs["controlled_g1_ebs6_acc6_fpbf16"]["train/samples_per_second_global"]), "#355070"),
                ("2 GPU", float(runs["controlled_g2_ebs6_acc3_fpbf16"]["train/samples_per_second_global"]), "#eaac8b"),
            ],
        },
    ]

    lines = svg_header(width, height)
    lines += [
        '<text x="90" y="38" class="title">DDP Throughput Comparison</text>',
        '<text x="90" y="60" class="subtitle">Samples/sec from the committed W&amp;B benchmark exports. Higher is better.</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#8d99ae" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#8d99ae" stroke-width="1.5"/>',
    ]

    for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
        y = top + chart_h - (tick / max_y) * chart_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>')

    group_centers = [left + chart_w * 0.27, left + chart_w * 0.73]
    bar_width = 90
    bar_gap = 28

    for center, group in zip(group_centers, groups):
        total_width = 2 * bar_width + bar_gap
        start = center - total_width / 2
        for idx, (bar_label, value, color) in enumerate(group["bars"]):
            x = start + idx * (bar_width + bar_gap)
            bar_h = (value / max_y) * chart_h
            y = top + chart_h - bar_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" rx="10" fill="{color}"/>')
            lines.append(f'<text x="{x + bar_width/2:.1f}" y="{y - 10:.1f}" text-anchor="middle" class="value">{value:.3f}</text>')
            lines.append(f'<text x="{x + bar_width/2:.1f}" y="{top + chart_h + 22:.1f}" text-anchor="middle" class="axis">{bar_label}</text>')
        lines.append(f'<text x="{center:.1f}" y="{top + chart_h + 52:.1f}" text-anchor="middle" class="label">{group["name"]}</text>')

    lines += [
        '<rect x="690" y="86" width="16" height="16" rx="3" fill="#355070"/>',
        '<text x="714" y="99" class="axis">1 GPU</text>',
        '<rect x="790" y="86" width="16" height="16" rx="3" fill="#eaac8b"/>',
        '<text x="814" y="99" class="axis">2 GPU</text>',
        "</svg>",
    ]
    write_svg(PLOTS_DIR / "throughput_comparison.svg", lines)


def build_efficiency_plot(runs: dict[str, dict[str, str]]) -> None:
    width, height = 980, 540
    left, right, top, bottom = 90, 40, 90, 90
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_y = 100.0

    controlled_speedup = float(runs["controlled_g2_ebs6_acc3_fpbf16"]["train/samples_per_second_global"]) / float(
        runs["controlled_g1_ebs6_acc6_fpbf16"]["train/samples_per_second_global"]
    )
    real_speedup = float(runs["real_g2_ebs16_acc8_fpbf16"]["train/samples_per_second_global"]) / float(
        runs["real_g1_ebs8_acc8_fpbf16"]["train/samples_per_second_global"]
    )

    bars = [
        ("Controlled", controlled_speedup / 2.0 * 100.0, controlled_speedup),
        ("Real", real_speedup / 2.0 * 100.0, real_speedup),
    ]

    lines = svg_header(width, height)
    lines += [
        '<text x="90" y="38" class="title">2-GPU Scaling Efficiency</text>',
        '<text x="90" y="60" class="subtitle">Efficiency = actual speedup / ideal 2x speedup. Controlled run is the stronger DDP claim.</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#8d99ae" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#8d99ae" stroke-width="1.5"/>',
    ]

    for tick in [0, 20, 40, 60, 80, 100]:
        y = top + chart_h - (tick / max_y) * chart_h
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick}%</text>')

    ideal_y = top + chart_h - chart_h
    lines.append(f'<line x1="{left}" y1="{ideal_y:.1f}" x2="{left + chart_w}" y2="{ideal_y:.1f}" stroke="#d62828" stroke-width="2" stroke-dasharray="8 6"/>')
    lines.append(f'<text x="{left + chart_w - 4}" y="{ideal_y - 8:.1f}" text-anchor="end" class="small">Ideal linear scaling (100%)</text>')

    centers = [left + chart_w * 0.33, left + chart_w * 0.68]
    bar_width = 140
    colors = ["#3a5a40", "#6d597a"]

    for center, (label, efficiency, speedup), color in zip(centers, bars, colors):
        x = center - bar_width / 2
        bar_h = (efficiency / max_y) * chart_h
        y = top + chart_h - bar_h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_h:.1f}" rx="12" fill="{color}"/>')
        lines.append(f'<text x="{center:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="value">{efficiency:.1f}%</text>')
        lines.append(f'<text x="{center:.1f}" y="{top + chart_h + 24:.1f}" text-anchor="middle" class="label">{label}</text>')
        lines.append(f'<text x="{center:.1f}" y="{top + chart_h + 44:.1f}" text-anchor="middle" class="small">speedup {speedup:.2f}x</text>')

    lines.append("</svg>")
    write_svg(PLOTS_DIR / "scaling_efficiency.svg", lines)


def main() -> None:
    runs = load_runs()
    missing = set(RUN_SPECS) - set(runs)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise SystemExit(f"Missing benchmark rows for: {missing_names}")
    build_throughput_plot(runs)
    build_efficiency_plot(runs)


if __name__ == "__main__":
    main()
