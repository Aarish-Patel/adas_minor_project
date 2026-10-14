"""Validation dashboard for the parking framework (TODO V4): CSV export, summary plots and one self-contained HTML page.

    python -m sim.parking_dashboard             # reads reports/parking_validation/results_{suite,mc}.csv (+ traces)
    -> reports/parking_validation/{results.csv, summary.csv, dashboard.html, plots/*.png}

Reads what `python -m sim.parking_validation suite|mc` wrote. The Monte Carlo runs carry the statistics; the nominal suite
(seed 0, no randomisation) is shown as a reference dot where a plot has one.
"""
import base64
import csv
import html
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "..", "reports", "parking_validation")

NUM = ("seed", "mc", "success", "planned", "refused", "arrived", "pos_err_cm", "lat_err_cm", "long_err_cm", "head_err_deg", "inside",
       "corrections", "gear_changes", "planned_gear_changes", "replans", "strokes", "collisions", "min_clear_cm", "time_s", "path_m",
       "sim_end_s", "start_x", "start_y", "start_deg", "selection_ok", "spaces", "free_spaces", "space_len_cm", "space_ratio", "plan_ms")
BLUE, ORANGE, GREEN, RED, GREY = "#2f6db5", "#d9822b", "#3a8f5c", "#c0392b", "#8a8f98"


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in NUM:
                if k in r:
                    r[k] = float(r[k]) if r[k] not in ("", None) else None
            rows.append(r)
    return rows


def target_len_cm(rows):
    """Length of the target space of each parallel scenario (from the deterministic scenario definitions), for the width classes."""
    from sim.parking_validation import all_cases
    spec = {n: p for n, g, p in all_cases() if g == "parallel"}
    out = {}
    for n, p in spec.items():
        if p["spaces"]:
            out[n] = p["spaces"]
    for r in rows:
        if r["group"] == "parallel" and r["name"] in out and r.get("target") not in ("", None):
            try:
                r["target_len_cm"] = 100 * out[r["name"]][int(float(r["target"]))][0]
            except (ValueError, IndexError):
                r["target_len_cm"] = None
    return rows


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


def pct(v, q):
    v = [x for x in v if x is not None]
    return float(np.percentile(v, q)) if v else None


def fig_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def plots(rows, traces, suite, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(out_dir, exist_ok=True)
    figs = {}
    mc = [r for r in rows if r["mc"] == 1]

    def by_name(rs):
        d = {}
        for r in rs:
            d.setdefault(r["name"], []).append(r)
        return d
    groups = by_name(mc)
    sref = {r["name"]: r for r in suite}

    # 1 success rate per scenario
    for world in ("perpendicular", "parallel"):
        names = sorted(n for n, rs in groups.items() if rs[0]["group"] == world)
        if world == "parallel":                                   # the 48 layout runs are summarised per layout, both lanes together
            names = [n for n in names if not n.startswith("par_layout")]
            lay = {}
            for n, rs in groups.items():
                if n.startswith("par_layout"):
                    lay.setdefault(n[:-1] + ("*"), []).extend(rs)
            groups_plot = dict({n: groups[n] for n in names}, **{k: v for k, v in lay.items()})
            names = sorted(groups_plot)
        else:
            groups_plot = groups
        fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.22 * len(names) + 1)))
        rate = [100 * np.mean([r["success"] for r in groups_plot[n]]) for n in names]
        cols = [GREEN if x >= 90 else (ORANGE if x >= 50 else RED) for x in rate]
        ax.barh(range(len(names)), rate, color=cols)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("success rate over the Monte Carlo runs (%)")
        for i, n in enumerate(names):
            ax.text(rate[i] + 1 if rate[i] < 88 else rate[i] - 1, i, f"{sum(r['success'] for r in groups_plot[n])}/{len(groups_plot[n])}",
                    va="center", ha="left" if rate[i] < 88 else "right", fontsize=7, color="black" if rate[i] < 88 else "white")
        ax.set_title(f"{world}: success per scenario")
        figs[f"success_{world}"] = fig_png(fig)
        plt.close(fig)

    # 2 pose / heading error, perpendicular scenarios
    pn = [n for n in sorted(groups) if n.startswith("perp_") and any(r["pos_err_cm"] is not None for r in groups[n])]
    for key, lab, tol in (("pos_err_cm", "final position error (cm)", 10), ("head_err_deg", "final heading error (deg)", 10)):
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ax.boxplot([[r[key] for r in groups[n] if r[key] is not None] for n in pn], showfliers=True, flierprops=dict(markersize=3))
        ax.axhline(tol, color=RED, lw=1, ls="--")
        ax.set_xticks(range(1, len(pn) + 1))
        ax.set_xticklabels([n[5:] for n in pn], rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(lab)
        ax.set_title("perpendicular (parked runs only); dashed = 10 tolerance")
        figs[f"err_{key}_perp"] = fig_png(fig)
        plt.close(fig)

    # 3 marker rotation -> heading error
    rot = [(0, "perp_nominal"), (5, "perp_rot5"), (10, "perp_rot10"), (20, "perp_rot20")]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xs, ys, lo, hi = [], [], [], []
    for d, n in rot:
        v = [r["head_err_deg"] for r in groups.get(n, []) if r["head_err_deg"] is not None]
        if v:
            xs.append(d)
            ys.append(np.median(v))
            lo.append(np.percentile(v, 10))
            hi.append(np.percentile(v, 90))
    ax.plot(xs, ys, "o-", color=BLUE)
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.18)
    ax.plot([0, 20], [0, 20], color=GREY, ls=":", label="1:1")
    ax.axhline(10, color=RED, lw=1, ls="--")
    ax.set_xlabel("marker rotation on the wall (deg)")
    ax.set_ylabel("final heading error (deg), median / 10-90 %")
    ax.legend(frameon=False)
    figs["marker_rotation"] = fig_png(fig)
    plt.close(fig)

    # 4 marker distance: detection and success
    dn = [(1.2, "perp_dist1.2"), (1.6, "perp_dist1.6"), (2.0, "perp_nominal"), (2.4, "perp_dist2.4"), (2.8, "perp_dist2.8"), (3.0, "perp_dist3.0")]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    det = [100 * np.mean([r["planned"] or 0 for r in groups[n]]) if n in groups else np.nan for _, n in dn]
    ax.plot([d for d, _ in dn], det, "o-", color=BLUE, label="marker detected and planned (%)")
    ax.set_xlabel("start distance from the wall (m)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8)
    figs["marker_distance"] = fig_png(fig)
    plt.close(fig)

    # 5 parallel: success vs target space length and lane
    par = [r for r in mc if r["group"] == "parallel" and r.get("target_len_cm") is not None and r["expect"] == "park"]
    if par:
        bins = [(0, 60, "narrow < 60"), (60, 90, "narrow 60-90"), (90, 140, "standard 90-140"), (140, 250, "wide > 140")]
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        w = 0.36
        for j, (lane, col, lab) in enumerate((("a", BLUE, "shipped start (70 cm from the edge)"), ("b", ORANGE, "start 90 cm from the edge"))):
            vals = []
            for lo_, hi_, _ in bins:
                sel = [r for r in par if lo_ <= r["target_len_cm"] < hi_ and (r["start_y"] is not None and (r["start_y"] > -0.25) == (lane == "a"))]
                vals.append(100 * np.mean([r["success"] for r in sel]) if sel else np.nan)
            ax.bar(np.arange(len(bins)) + (j - 0.5) * w, vals, w, color=col, label=lab)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels([b[2] for b in bins])
        ax.set_ylabel("success (%)")
        ax.set_xlabel("target space length (cm; car 33 cm)")
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, 105)
        figs["parallel_width"] = fig_png(fig)
        plt.close(fig)

    # 6 parallel error distributions (parked runs)
    pr = [r for r in mc if r["group"] == "parallel" and r["pos_err_cm"] is not None]
    if pr:
        fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.2))
        for ax, key, lab, tol in ((axs[0], "pos_err_cm", "position error (cm)", 10), (axs[1], "head_err_deg", "heading error (deg)", 10)):
            ax.hist([r[key] for r in pr], bins=24, color=BLUE)
            ax.axvline(tol, color=RED, ls="--", lw=1)
            ax.set_xlabel(lab)
            ax.set_ylabel("runs")
        fig.suptitle(f"parallel, {len(pr)} runs that parked", fontsize=9)
        figs["parallel_errors"] = fig_png(fig)
        plt.close(fig)

    # 7 time vs path length; corrections
    ok = [r for r in mc if r["time_s"] is not None]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.3))
    for g, col in (("perpendicular", BLUE), ("parallel", ORANGE)):
        s = [r for r in ok if r["group"] == g]
        axs[0].scatter([r["path_m"] for r in s], [r["time_s"] for r in s], s=8, alpha=0.5, color=col, label=g)
    axs[0].set_xlabel("path length (m)")
    axs[0].set_ylabel("time to complete (s)")
    axs[0].legend(frameon=False, fontsize=8)
    mx = int(max([r["corrections"] for r in ok] or [1]))
    for g, col, off in (("perpendicular", BLUE, -0.2), ("parallel", ORANGE, 0.2)):
        v = [r["corrections"] for r in ok if r["group"] == g]
        h = np.bincount(np.minimum(np.array(v, int), 8), minlength=9) if v else np.zeros(9)
        axs[1].bar(np.arange(9) + off, h, 0.4, color=col, label=g)
    axs[1].set_xticks(range(9))
    axs[1].set_xticklabels([str(i) for i in range(8)] + ["8+"])
    axs[1].set_xlabel("corrections (extra gear changes + re-plans + straightening strokes)")
    axs[1].set_ylabel("runs")
    figs["time_corrections"] = fig_png(fig)
    plt.close(fig)

    # 8 failure reasons
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    reasons = {}
    for r in mc:
        if not r["success"]:
            reasons.setdefault(r["group"], {}).setdefault(r["failure"].split(":")[0], 0)
            reasons[r["group"]][r["failure"].split(":")[0]] += 1
    allr = sorted({k for d in reasons.values() for k in d})
    left = np.zeros(2)
    for i, k in enumerate(allr):
        v = np.array([reasons.get(g, {}).get(k, 0) for g in ("perpendicular", "parallel")], float)
        ax.barh([0, 1], v, left=left, label=k, color=plt.cm.tab10(i))
        left += v
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["perpendicular", "parallel"])
    ax.set_xlabel("failed runs")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    figs["failures"] = fig_png(fig)
    plt.close(fig)

    # 9 trajectories
    if traces:
        fig, axs = plt.subplots(2, 1, figsize=(7.5, 6.2))
        for ax, world, names in ((axs[0], "perpendicular", ("perp_nominal", "perp_rot10", "perp_drift_strong")),
                                 (axs[1], "parallel", ("par_width_standard", "par_click_between", "par_wide_far_lane"))):
            for n, col in zip(names, (BLUE, ORANGE, GREEN)):
                tr = [t for t in traces if t["name"] == n]
                for j, t in enumerate(tr[:6]):
                    a = np.array(t["trace"])
                    if len(a):
                        ax.plot(a[:, 0], a[:, 1], color=col, lw=0.8, alpha=0.7, label=n if j == 0 else None)
            ax.set_aspect("equal")
            ax.set_title(world + " (up to 6 Monte Carlo runs each)", fontsize=8)
            ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        fig.tight_layout()
        figs["trajectories"] = fig_png(fig)
        plt.close(fig)
    for k, v in figs.items():
        with open(os.path.join(out_dir, k + ".png"), "wb") as f:
            f.write(v)
    return figs


def group_stats(rows):
    out = {}
    for g in ("perpendicular", "parallel"):
        rs = [r for r in rows if r["group"] == g]
        parked = [r for r in rs if r["time_s"] is not None]
        exp_park = [r for r in rs if r["expect"] == "park"]
        out[g] = {"runs": len(rs), "success_pct": 100 * np.mean([r["success"] for r in rs]) if rs else None,
                  "success_expect_park_pct": 100 * np.mean([r["success"] for r in exp_park]) if exp_park else None,
                  "collisions": int(sum(r["collisions"] for r in rs)), "parked": len(parked),
                  "pos_med": med([r["pos_err_cm"] for r in parked]), "pos_p95": pct([r["pos_err_cm"] for r in parked], 95),
                  "head_med": med([r["head_err_deg"] for r in parked]), "head_p95": pct([r["head_err_deg"] for r in parked], 95),
                  "corr_med": med([r["corrections"] for r in parked]), "time_med": med([r["time_s"] for r in parked]),
                  "path_med": med([r["path_m"] for r in parked])}
    return out


def scenario_table(rows):
    d = {}
    for r in rows:
        d.setdefault((r["group"], r["name"]), []).append(r)
    out = []
    for (g, n), rs in sorted(d.items()):
        parked = [r for r in rs if r["time_s"] is not None]
        fails = {}
        for r in rs:
            if not r["success"]:
                k = r["failure"].split(":")[0]
                fails[k] = fails.get(k, 0) + 1
        out.append({"group": g, "name": n, "expect": rs[0]["expect"], "runs": len(rs), "success": int(sum(r["success"] for r in rs)),
                    "success_pct": round(100 * sum(r["success"] for r in rs) / len(rs), 1), "collisions": int(sum(r["collisions"] for r in rs)),
                    "pos_med": med([r["pos_err_cm"] for r in parked]), "pos_p95": pct([r["pos_err_cm"] for r in parked], 95),
                    "lat_med": med([r["lat_err_cm"] for r in parked]), "long_med": med([r["long_err_cm"] for r in parked]),
                    "head_med": med([r["head_err_deg"] for r in parked]), "head_p95": pct([r["head_err_deg"] for r in parked], 95),
                    "corr_med": med([r["corrections"] for r in parked]), "time_med": med([r["time_s"] for r in parked]),
                    "path_med": med([r["path_m"] for r in parked]), "failures": "; ".join(f"{k} x{v}" for k, v in fails.items())})
    return out


def f1(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def build_html(rows, suite, figs, tbl, gs):
    def card(t, v, s=""):
        return f'<div class="card"><div class="k">{html.escape(t)}</div><div class="v">{v}</div><div class="s">{html.escape(s)}</div></div>'
    img = lambda k, w="100%": f'<img style="max-width:{w}" alt="{k}" src="data:image/png;base64,{base64.b64encode(figs[k]).decode()}">' if k in figs else ""
    total = len(rows)
    coll = sum(r["collisions"] for r in rows)
    head = "".join([
        card("Monte Carlo runs", f"{total}", f"{len(tbl)} scenarios; seeds 100+"),
        card("Contacts", f"{coll}", "0 required (twin stops at first contact)"),
        card("Perpendicular success", f"{f1(gs['perpendicular']['success_pct'])} %", f"{gs['perpendicular']['runs']} runs"),
        card("Parallel success", f"{f1(gs['parallel']['success_pct'])} %", f"{gs['parallel']['runs']} runs"),
        card("Median pose error", f"{f1(gs['perpendicular']['pos_med'])} / {f1(gs['parallel']['pos_med'])} cm", "perp / parallel"),
        card("Median heading error", f"{f1(gs['perpendicular']['head_med'])} / {f1(gs['parallel']['head_med'])} deg", "perp / parallel")])
    trs = []
    for t in tbl:
        cls = "ok" if t["success_pct"] >= 90 else ("warn" if t["success_pct"] >= 50 else "bad")
        trs.append(f'<tr class="{cls}"><td>{t["name"]}</td><td>{t["expect"]}</td><td>{t["success"]}/{t["runs"]}</td><td>{t["success_pct"]}</td>'
                   f'<td>{t["collisions"]}</td><td>{f1(t["pos_med"])} / {f1(t["pos_p95"])}</td><td>{f1(t["lat_med"])} / {f1(t["long_med"])}</td><td>{f1(t["head_med"])} / {f1(t["head_p95"])}</td>'
                   f'<td>{f1(t["corr_med"], 0)}</td><td>{f1(t["time_med"])}</td><td>{f1(t["path_med"], 2)}</td><td>{html.escape(t["failures"])}</td></tr>')
    table = ('<table><thead><tr><th>scenario</th><th>expect</th><th>success</th><th>%</th><th>contacts</th><th>pos cm med / p95</th><th>lateral / along cm (med)</th>'
             '<th>heading deg med / p95</th><th>corr.</th><th>time s</th><th>path m</th><th>failures</th></tr></thead><tbody>'
             + "".join(trs) + "</tbody></table>")
    sect = lambda title, *keys: f"<h2>{title}</h2><div class='grid'>" + "".join(f"<div>{img(k)}</div>" for k in keys) + "</div>"
    css = """
:root{--bg:#f6f7f9;--fg:#1c2330;--card:#fff;--mut:#5c6675;--line:#dfe3ea}
@media (prefers-color-scheme:dark){:root{--bg:#161b24;--fg:#e6e9ef;--card:#1e2530;--mut:#9aa4b2;--line:#2c3543}}
body{margin:0;padding:20px 16px 60px;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif}
main{max-width:1100px;margin:auto}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}p.m{color:var(--mut);margin:2px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.k{color:var(--mut);font-size:12px}
.v{font-size:20px;font-weight:600}.s{color:var(--mut);font-size:11px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px}
.grid img{width:100%;background:#fff;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:12px;background:var(--card)}th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
tr.bad td:nth-child(4){color:#c0392b;font-weight:600}tr.warn td:nth-child(4){color:#d9822b;font-weight:600}tr.ok td:nth-child(4){color:#3a8f5c}
.wrap{overflow-x:auto}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Parking Validation</title><style>{css}</style></head><body><main>
<h1>Parking validation dashboard</h1>
<p class="m">Both Parking Simulator worlds (park_perp, park_street) on the relay's own follower, brake gate and speed control in the digital twin.
Monte Carlo: randomised car (steering gain, servo offset, motor lag, delay), LiDAR noise / dropout / yaw error, camera pixel noise, start pose.
Success = 0 contacts, pose error &le; 10 cm, heading error &le; 10&deg; (and inside the space for parallel), or a correct refusal where the case demands one.</p>
<div class="cards">{head}</div>
{sect("Success per scenario", "success_perpendicular", "success_parallel")}
{sect("Perpendicular: errors", "err_pos_err_cm_perp", "err_head_err_deg_perp")}
{sect("Perpendicular: what the marker costs", "marker_rotation", "marker_distance")}
{sect("Parallel: space size and errors", "parallel_width", "parallel_errors")}
{sect("Time, path, corrections and failures", "time_corrections", "failures")}
{sect("Trajectories", "trajectories")}
<h2>All scenarios</h2><div class="wrap">{table}</div>
<p class="m">Files: results.csv (one row per run), summary.csv (per scenario), plots/. Regenerate: python -m sim.parking_validation mc; python -m sim.parking_dashboard.</p>
</main></body></html>"""


def main():
    suite = load(os.path.join(OUT, "results_suite.csv")) if os.path.exists(os.path.join(OUT, "results_suite.csv")) else []
    mc = load(os.path.join(OUT, "results_mc.csv"))
    rows = target_len_cm(suite + mc)
    suite, mc_rows = [r for r in rows if r["mc"] == 0], [r for r in rows if r["mc"] == 1]
    tp = os.path.join(OUT, "traces_mc.json")
    traces = json.load(open(tp, encoding="utf-8")) if os.path.exists(tp) else []
    figs = plots(rows, traces, suite, os.path.join(OUT, "plots"))
    tbl = scenario_table(mc_rows)
    gs = group_stats(mc_rows)
    cols = list(rows[0].keys()) + ["target_len_cm"]
    with open(os.path.join(OUT, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(cols)), restval="")
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tbl[0].keys()))
        w.writeheader()
        w.writerows(tbl)
    with open(os.path.join(OUT, "dashboard.html"), "w", encoding="utf-8") as f:
        f.write(build_html(mc_rows, suite, figs, tbl, gs))
    with open(os.path.join(OUT, "group_stats.json"), "w", encoding="utf-8") as f:
        json.dump(gs, f, indent=1)
    print(json.dumps(gs, indent=1))
    print("wrote", os.path.join(OUT, "dashboard.html"))


if __name__ == "__main__":
    main()
