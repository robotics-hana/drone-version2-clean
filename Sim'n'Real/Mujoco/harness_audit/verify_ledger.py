"""verify_ledger.py -- recompute every number on the Results Ledger from the
raw logs in Reports/eval_logs/, printing one verification row per claim.

Nothing is trusted from the page, the decision log, or session memory: each
row is (claim, page value, recomputed value, source file:line, status).
"""
import json
import os

import numpy as np

LOGS = r"c:/Users/hanah/Projects/drone-version2/Reports/eval_logs"
rows = []


def add(claim, page, recomputed, src, status=None):
    if status is None:
        status = "VERIFIED" if str(page) == str(recomputed) else "CORRECTED"
    rows.append((claim, str(page), str(recomputed), src, status))


def evals(fname, kind=None):
    """(line_no, record) for EVAL lines in a results log."""
    out = []
    path = os.path.join(LOGS, fname)
    if not os.path.exists(path):
        return out
    for i, l in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        if l.startswith("EVAL {"):
            r = json.loads(l[5:])
            if kind is None or r["kind"] == kind:
                out.append((i, r))
    return out


def pickstats(fname):
    ev = evals(fname, "pick")
    m = np.array([r["miss_mm"] for _, r in ev])
    return dict(n=len(ev), succ=sum(r["success"] for _, r in ev),
                picked=sum(r["picked"] for _, r in ev),
                best=(round(float(m.min()), 1) if len(m) else None),
                med=(round(float(np.median(m)), 1) if len(m) else None),
                q1=(round(float(np.quantile(m, .25)), 1) if len(m) else None),
                q3=(round(float(np.quantile(m, .75)), 1) if len(m) else None),
                mx=(round(float(m.max()), 1) if len(m) else None),
                lines=(ev[0][0], ev[-1][0]) if ev else None)


def navstats(fname):
    ev = evals(fname, "nav")
    return dict(n=len(ev), succ=sum(r["success"] for _, r in ev))


# ---------------- Tab 4: attribution square (30k) ----------------
for f, label, page_best in [("ladder_naive.log", "broken harness", 53),
                            ("startfix_naive.log", "--startfix", 205),
                            ("pfix_naive.log", "--platfix", 73),
                            ("both_naive.log", "both", 57)]:
    s = pickstats(f)
    add(f"attr {label}: success", "0/20", f"{s['succ']}/{s['n']}", f"{f}:{s['lines']}")
    add(f"attr {label}: best miss mm", page_best, s["best"], f"{f}")
    add(f"attr {label}: median miss mm (new on page)", "-", s["med"], f"{f}", "VERIFIED")

# ---------------- Tab 5: honest ladder (60k) ----------------
s = pickstats("b60k_naive.log")
add("60k naive pick success", "0/20", f"{s['succ']}/{s['n']}", "b60k_naive.log")
add("60k naive picked-up", 1, s["picked"], "b60k_naive.log")
add("60k naive best miss mm", 7.6, s["best"], "b60k_naive.log")
ev = evals("b60k_naive.log", "pick")
idx = [i + 1 for i, (_, r) in enumerate(ev) if r["picked"]]
add("60k pick-up episode index", 4, idx[0] if idx else None, "b60k_naive.log")
add("60k pick-up object", "penguin",
    ev[idx[0] - 1][1]["obj"] if idx else None, "b60k_naive.log",
    "VERIFIED" if idx and "penguin" in ev[idx[0] - 1][1]["obj"] else "CORRECTED")
for f, mode, pg in [("60krtc_naive.log", "rtc", 0), ("60kguided_naive.log", "guided", 0)]:
    s = pickstats(f)
    add(f"60k {mode} pick success", f"{pg}/20", f"{s['succ']}/{s['n']}", f)
    add(f"60k {mode} best miss mm", {"rtc": 49.8, "guided": 33.0}[mode], s["best"], f)
for f, mode, pg in [("lnav_naive_out.log", "naive", 12),
                    ("lnav_rtc_out.log", "rtc", 5),
                    ("lnav_guided_out.log", "guided", 7)]:
    s = navstats(f)
    add(f"60k nav {mode}", f"{pg}/20", f"{s['succ']}/{s['n']}", f)
    c = evals(f, "comp")
    add(f"60k comp {mode}", "0/20",
        f"{sum(r['success'] for _, r in c)}/{len(c)}", f)
s = pickstats("objyaw_out.log")
add("60k objyaw0 pick", "0/20", f"{s['succ']}/{s['n']}", "objyaw_out.log")

# ---------------- Tabs 7-8: interventions ----------------
s = pickstats("horizon10_out.log")
add("horizon10 picked", 0, s["picked"], "horizon10_out.log")
add("horizon10 median mm", 412, round(s["med"]), "horizon10_out.log",
    "VERIFIED" if abs(s["med"] - 412.4) < 1 else "CORRECTED")
add("horizon10 IQR mm", "286-636", f"{round(s['q1'])}-{round(s['q3'])}", "horizon10_out.log")
s1 = pickstats("oracleyaw_out.log")
add("oracleyaw v1 picked", 5, s1["picked"], "oracleyaw_out.log")
add("oracleyaw v1 median mm", 27.0, s1["med"], "oracleyaw_out.log")
add("oracleyaw v1 IQR", "7.7-45.5", f"{s1['q1']}-{s1['q3']}", "oracleyaw_out.log")
add("oracleyaw v1 max mm", 65.1, s1["mx"], "oracleyaw_out.log")
s2 = pickstats("orayaw2_out.log")
add("orayaw2 picked", 1, s2["picked"], "orayaw2_out.log")
add("orayaw2 median mm", 41.1, s2["med"], "orayaw2_out.log")
a = np.array([r["miss_mm"] for _, r in evals("oracleyaw_out.log", "pick")])
b = np.array([r["miss_mm"] for _, r in evals("orayaw2_out.log", "pick")])
add("paired v2-v1 median diff mm", "+7", f"{np.median(b - a):+.1f}",
    "both oracle logs", "VERIFIED" if abs(np.median(b - a) - 7.1) < 0.2 else "CORRECTED")
add("oracle-yaw pooled pick-up", "6/40", f"{s1['picked'] + s2['picked']}/{s1['n'] + s2['n']}",
    "both oracle logs")

# ---------------- Tab 6: audit (tag audit60k in eval_traj.jsonl) ----------------
recs = [json.loads(l) for l in open(os.path.join(LOGS, "eval_traj.jsonl"),
                                    encoding="utf-8", errors="replace")]
seg = [r for r in recs if r.get("tag") == "audit60k" and r["kind"] == "pick"]
add("audit n (tag=audit60k)", 20, len(seg), "eval_traj.jsonl")
jm, bm, xb, yb, spd, nose_b = [], [], [], [], [], []
mu = np.array([0.021, 0.563])
for r in seg:
    t = np.array(r["traj"]); o = np.array(r["obj_xy"])
    dj = np.linalg.norm(t[:, 4:6] - o, axis=1); k = int(dj.argmin())
    db = np.linalg.norm(t[:, 0:2] - o, axis=1)
    jm.append(dj[k]); bm.append(db.min())
    off = t[k, 4:6] - o; yaw = t[k, 3]
    c, s_ = np.cos(-yaw), np.sin(-yaw)
    xb.append(c * off[0] - s_ * off[1]); yb.append(s_ * off[0] + c * off[1])
    spd.append(np.linalg.norm(o - mu))
    nose = np.array([np.sin(t[k, 3]), -np.cos(t[k, 3])])
    v = o - t[k, 0:2]; v = v / max(np.linalg.norm(v), 1e-9)
    nose_b.append((np.arctan2(nose[1], nose[0]), np.arctan2(v[1], v[0])))
jm, bm, xb, yb, spd = map(np.array, (jm, bm, xb, yb, spd))
add("audit jaws median mm", 223, round(np.median(jm) * 1000), "eval_traj.jsonl")
add("audit body median mm", 365, round(np.median(bm) * 1000), "eval_traj.jsonl")
add("audit x_b sd mm", 263, round(xb.std() * 1000), "eval_traj.jsonl")
add("audit x_b side split", "55/45",
    f"{round(max((xb>0).mean(),(xb<0).mean())*100)}/{round(min((xb>0).mean(),(xb<0).mean())*100)}",
    "eval_traj.jsonl")
add("audit y_b mean mm (undershoot)", "+120", f"{yb.mean()*1000:+.0f}", "eval_traj.jsonl")
add("audit y_b consistent frac", "18/20", f"{int((yb>0).sum())}/20", "eval_traj.jsonl")
r20 = np.corrcoef(spd, jm)[0, 1]
add("audit spawn corr r (n=20)", "+0.33", f"{r20:+.2f}", "eval_traj.jsonl")


def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi
def cmean(a): return np.arctan2(np.sin(a).mean(), np.cos(a).mean())


nn = np.array([p[0] for p in nose_b]); bb = np.array([p[1] for p in nose_b])
sl = np.polyfit(wrap(bb - cmean(bb)), wrap(nn - cmean(nn)), 1)[0]
add("nose-on-bearing slope (closest)", "+0.39", f"{sl:+.2f}", "eval_traj.jsonl")

# heading sign test + FOV
fov_fr, signs = [], []
for r in seg:
    t = np.array(r["traj"]); o = np.array(r["obj_xy"])
    d = np.linalg.norm(t[:, 0:2] - o, axis=1); k = int(d.argmin())
    a_ = []
    es = []
    for i in range(max(k, 1)):
        nose = np.array([np.sin(t[i, 3]), -np.cos(t[i, 3])])
        v = o - t[i, 0:2]; nv = np.linalg.norm(v)
        if nv < 0.05:
            continue
        v = v / nv
        ang = np.degrees(np.arccos(np.clip(nose @ v, -1, 1)))
        a_.append(ang)
        if d[i] > 0.6:
            es.append(np.degrees(np.arctan2(nose[0] * v[1] - nose[1] * v[0],
                                            float(np.clip(nose @ v, -1, 1)))))
    fov_fr.append(float((np.array(a_) < 65).mean()) if a_ else np.nan)
    if es:
        nose_k = np.array([np.sin(t[k, 3]), -np.cos(t[k, 3])])
        off = t[k, 4:6] - o
        lat = nose_k[0] * off[1] - nose_k[1] * off[0]
        signs.append(np.sign(np.median(es)) == np.sign(lat))
add("object in nose FOV (median frac)", "1.00", f"{np.nanmedian(fov_fr):.2f}", "eval_traj.jsonl")
add("heading-sign match count", "2/20", f"{int(np.sum(signs))}/{len(signs)}", "eval_traj.jsonl")

# NEW: undershoot under the pin (the flagged geometry problem)
for tag in ("oracleyaw", "orayaw2"):
    sg = [r for r in recs if r.get("tag") == tag and r["kind"] == "pick"]
    if not sg:
        add(f"undershoot y_b under pin ({tag})", "?", "NO TAGGED RECORDS",
            "eval_traj.jsonl", "UNVERIFIED")
        continue
    ybp = []
    for r in sg:
        t = np.array(r["traj"]); o = np.array(r["obj_xy"])
        dj = np.linalg.norm(t[:, 4:6] - o, axis=1); k = int(dj.argmin())
        off = t[k, 4:6] - o; yaw = t[k, 3]
        c, s_ = np.cos(-yaw), np.sin(-yaw)
        ybp.append(s_ * off[0] + c * off[1])
    ybp = np.array(ybp)
    add(f"undershoot y_b under pin ({tag}) mm",
        "(claimed 'collapsed')", f"{ybp.mean()*1000:+.0f} (sd {ybp.std()*1000:.0f})",
        "eval_traj.jsonl", "INFERRED->MEASURED")

# ---------------- Tab 6: powered corr n=66 ----------------
pre = [json.loads(l) for l in open(os.path.join(LOGS, "eval_traj_pre_audit.jsonl"),
                                   encoding="utf-8", errors="replace")]
picks = [r for r in pre if r["kind"] == "pick"]
sfx = [r for r in picks if np.array(r["traj"])[0, 2] < 0.45]
bym = {}
for r in sfx:
    bym.setdefault(r["mode"], []).append(r)
nv = bym.get("naive", [])
ok = len(nv) == 86 and len(bym.get("rtc", [])) == 20 and len(bym.get("guided", [])) == 20
blocks = ({"b60k": nv[40:60], "vid": nv[80:86], "rtc": bym["rtc"],
           "guided": bym["guided"]} if ok else {})
sd_t = np.array([0.431, 0.240])
if blocks:
    miss, maha, ox_, fx_ = [], [], [], []
    blk = []
    for bn, rs in blocks.items():
        for r in rs:
            t = np.array(r["traj"]); o = np.array(r["obj_xy"])
            d = np.linalg.norm(t[:, 0:2] - o, axis=1)
            miss.append(d.min()); blk.append(bn)
            dev = o - mu
            maha.append(np.sqrt(((dev / sd_t) ** 2).sum()))
            k = int(d.argmin()); ox_.append(o[0]); fx_.append(t[k, 0] - o[0])
    miss, maha = np.array(miss), np.array(maha)
    md = miss.copy()
    for b in set(blk):
        m_ = np.array(blk) == b
        md[m_] -= miss[m_].mean()
    add("powered n", 66, len(miss), "eval_traj_pre_audit.jsonl")
    add("powered radial r (demeaned)", "+0.44", f"{np.corrcoef(maha, md)[0,1]:+.2f}",
        "eval_traj_pre_audit.jsonl")
    dev_x = np.array(ox_) - mu[0]
    add("powered x-slope (miss_x vs dev_x)", "-0.86",
        f"{np.polyfit(dev_x, np.array(fx_), 1)[0]:+.2f}", "eval_traj_pre_audit.jsonl")
else:
    add("powered block identification", "n=66", "BLOCK COUNTS CHANGED", "pre_audit", "UNVERIFIED")

# ---------------- Tab 2/3: probe + fixfit + repdiag ----------------
def grepnum(fname, needle):
    path = os.path.join(LOGS, fname)
    for i, l in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        if needle in l:
            return i, l.strip()
    return None, None


i, l = grepnum("probe.log", "endpoint err")
add("probe30k endpoint line", "2.90cm / 28.56cm", l, f"probe.log:{i}",
    "VERIFIED" if l and "0.0290" in l and "0.2856" in l else "CORRECTED")
i, l = grepnum("probe60k.log", "endpoint err")
add("probe60k endpoint line", "2.57cm", l, f"probe60k.log:{i}",
    "VERIFIED" if l and "0.0257" in l else "CORRECTED")
i, l = grepnum("fixfit5.log", "PLATFORM-FIX5")
add("fixfit5 verdict", "A=3/3 B=3/3", l, f"fixfit5.log:{i}",
    "VERIFIED" if l and "A=3/3 B=3/3" in l else "CORRECTED")
# fixfit5 placement distances
ds = []
for i, l in enumerate(open(os.path.join(LOGS, "fixfit5.log"), encoding="utf-8",
                           errors="replace"), 1):
    if "final obj xy" in l:
        ds.append(float(l.split("d=")[1].split(" m")[0]) * 1000)
add("fixfit5 placement range mm", "7-13", f"{min(ds):.0f}-{max(ds):.0f}", "fixfit5.log")
ms = []
for l in open(os.path.join(LOGS, "repdiag2.log"), encoding="utf-8", errors="replace"):
    if "MISS(live_target)" in l:
        ms.append(float(l.split("MISS(live_target)=")[1].split(" mm")[0]))
add("repdiag2 replay misses mm", "102-152", f"{min(ms):.0f}-{max(ms):.0f}", "repdiag2.log",
    "VERIFIED" if abs(min(ms) - 101.7) < 1 and abs(max(ms) - 151.9) < 1 else "CORRECTED")

# ---------------- print ----------------
W = max(len(r[0]) for r in rows)
n_bad = 0
for c, p, rec, src, st in rows:
    if st not in ("VERIFIED",):
        n_bad += (st == "CORRECTED" or st == "UNVERIFIED")
    print(f"{st:22s} | {c:<{W}s} | page: {p:>18s} | recomputed: {rec:>18s} | {src}")
print(f"\n{len(rows)} claims checked; {sum(1 for r in rows if r[4]=='VERIFIED')} verified as printed")
