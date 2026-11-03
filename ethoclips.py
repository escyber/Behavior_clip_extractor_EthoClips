#!/usr/bin/env python3
"""
ethoclips.py - find moments in scored behavior videos and review them. One file, one command.

Start it (a page opens in your browser):

    python3 ethoclips.py

The page lets you pick the video (AVI, MP4, MOV, MKV ...) and the scoring files (BORIS TSV, one per
scorer), choose what to find, and review each moment with every scorer's labels next to it.
The video is read directly from the file, frame by frame. Nothing needs converting and nothing is uploaded.

Needs: Python 3.8+, and `pip install opencv-python numpy`. ffmpeg is only used (if present) to save clips.

Useful options:
    --dir FOLDER      folder the file chooser opens in (default: the current folder)
    --port 8765       port to use
    --no-browser      do not open a browser (for example on Sherlock, see below)

Running on Sherlock without copying videos to your Mac:
    on Sherlock:   python3 ethoclips.py --no-browser      (prints a link)
    on your Mac:   ssh -L 8765:localhost:8765 <your sherlock login>
    then open the link in your Mac's browser. The video stays on Sherlock; only frames are sent.
    (Run it inside an interactive allocation, not on a login node.)

Save clips without opening a page (for example in a batch job):

    python3 ethoclips.py cut overlap --a rearing --b resting \\
        --video m1917.avi --tsv scoring/*.tsv --out clips_rearing_resting

    rules:  behavior --behavior face_swipe  |  overlap --a A --b B [--scope same|different]  |  multi
    options: --min-frames 5  --pad 1  --merge-gap 1  --frame-offset 0  --names Yash Sophie
"""
import argparse, csv, glob, json, os, re, secrets, shutil, struct, subprocess, sys, threading, time, webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

try:
    import numpy as np
    import cv2
except ImportError:
    sys.exit("This tool needs numpy and OpenCV. Install them with:  pip install opencv-python numpy")

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".webm"}
TEXT_EXT = {".tsv", ".txt", ".csv"}


# ============================================================== scoring files
def norm(name):
    """Behavior names are compared lowercase, with underscores (same as the Behavior Agreement viewer)."""
    return re.sub(r"[\s\-]+", "_", name.strip().lower()).replace("locomotor", "locomotive")


def guess_names(paths):
    """Guess who scored each file: the word that appears in only one file name."""
    stems = [re.split(r"[_\-\s.]+", os.path.splitext(os.path.basename(p))[0]) for p in paths]
    out = []
    for i, toks in enumerate(stems):
        if len(paths) < 2:
            out.append(os.path.splitext(os.path.basename(paths[i]))[0]); continue
        others = {t for j, s in enumerate(stems) if j != i for t in s}
        uniq = [t for t in toks if t not in others and t.isalpha()]
        out.append(uniq[0] if uniq else os.path.splitext(os.path.basename(paths[i]))[0])
    return out


def read_boris(path, warnings, who):
    """-> dict(intervals=[(behavior, start, end)] end inclusive, fps, media, mdur)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"{os.path.basename(path)} is empty")
    head = rows[0]
    for c in ("Behavior", "Behavior type", "Image index"):
        if c not in head:
            raise ValueError(f'{os.path.basename(path)}: column "{c}" not found. Is this a BORIS event-log TSV?')
    ix = {c: head.index(c) for c in head}
    open_ev, out = defaultdict(list), []
    fps = media = mdur = None
    for r in rows[1:]:
        if len(r) < len(head):
            continue
        beh = norm(r[ix["Behavior"]])
        typ = r[ix["Behavior type"]].strip().upper()
        try:
            fr = int(float(r[ix["Image index"]]))
        except ValueError:
            warnings.append(f"{who}: a row with a bad frame number was skipped ({beh}, {typ})"); continue
        if fps is None and "FPS" in ix:
            try: fps = float(r[ix["FPS"]])
            except ValueError: pass
        if media is None and "Media file name" in ix and r[ix["Media file name"]].strip().upper() not in ("", "NA"):
            media = re.split(r"[\\/]", r[ix["Media file name"]])[-1]
        if mdur is None and "Media duration (s)" in ix:
            try: mdur = float(r[ix["Media duration (s)"]])
            except ValueError: pass
        if typ == "START":
            open_ev[beh].append(fr)
        elif typ == "STOP":
            if open_ev[beh]: out.append((beh, open_ev[beh].pop(0), fr))
            else: warnings.append(f"{who}: STOP without START for {beh} at frame {fr} (skipped)")
        elif typ == "POINT":
            out.append((beh, fr, fr))
    for beh, starts in open_ev.items():
        for s in starts:
            warnings.append(f"{who}: START without STOP for {beh} at frame {s} (skipped)")
    return dict(intervals=out, fps=fps, media=media, mdur=mdur)


def describe_scoring(paths):
    names, out = guess_names(paths), []
    for p, n in zip(paths, names):
        try:
            w = []
            r = read_boris(p, w, n)
            out.append(dict(path=p, name=n, file=os.path.basename(p), events=len(r["intervals"]), fps=r["fps"],
                            media=r["media"], mdur=r["mdur"], behaviors=sorted({b for b, _, _ in r["intervals"]}), warnings=w))
        except Exception as e:
            out.append(dict(path=p, file=os.path.basename(p), error=str(e)))
    return out


# ============================================================== finding moments
def runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def build_masks(per, total):
    masks = {}
    for s, ivs in per.items():
        d = {}
        for beh, a, b in ivs:
            d.setdefault(beh, np.zeros(total, bool))[max(a, 0):min(b, total - 1) + 1] = True
        masks[s] = d
    return masks


def labels_in_window(masks, scorer, a, b):
    cnt = [(int(m[a:b + 1].sum()), beh) for beh, m in masks[scorer].items()]
    return [beh for n, beh in sorted((c for c in cnt if c[0]), key=lambda x: (-x[0], x[1]))]


def find_events(masks, total, P):
    ev = []
    if P["rule"] == "behavior":
        for s, d in masks.items():
            if P["beh"] in d:
                for a, b in runs(d[P["beh"]]):
                    ev.append(dict(start=int(a), end=int(b), scorers={s}, behaviors={P["beh"]}))
    elif P["rule"] == "overlap":
        for s1, d1 in masks.items():
            for s2, d2 in masks.items():
                if (P["scope"] == "same") != (s1 == s2) or P["a"] not in d1 or P["b"] not in d2:
                    continue
                for a, b in runs(d1[P["a"]] & d2[P["b"]]):
                    if b - a + 1 >= P["min"]:
                        ev.append(dict(start=int(a), end=int(b), scorers={s1, s2}, behaviors={P["a"], P["b"]}))
    else:
        for s, d in masks.items():
            if len(d) < 2: continue
            names = sorted(d)
            stack = np.stack([d[n] for n in names])
            code = np.zeros(total, np.int64)
            for k in range(len(names)):
                code += stack[k].astype(np.int64) << k
            for a, b in runs(stack.sum(axis=0) >= 2):
                seg = code[a:b + 1]
                bounds = np.concatenate(([0], np.where(np.diff(seg) != 0)[0] + 1, [len(seg)]))
                for i in range(len(bounds) - 1):
                    x, y = a + bounds[i], a + bounds[i + 1] - 1
                    if y - x + 1 >= P["min"]:
                        ev.append(dict(start=int(x), end=int(y), scorers={s},
                                       behaviors={names[k] for k in range(len(names)) if (code[x] >> k) & 1}))
    return ev


def merge_events(events, gap):
    out = []
    for e in sorted(events, key=lambda e: (e["start"], e["end"])):
        if out and e["start"] <= out[-1]["end"] + gap:
            c = out[-1]; c["end"] = max(c["end"], e["end"]); c["scorers"] |= e["scorers"]; c["behaviors"] |= e["behaviors"]; c["n"] += 1
        else:
            out.append(dict(e, scorers=set(e["scorers"]), behaviors=set(e["behaviors"]), n=1))
    return out


def rule_name(P):
    if P["rule"] == "behavior": return f"{P['beh']}_any_scorer"
    if P["rule"] == "overlap": return f"{P['a']}+{P['b']}_{P['scope']}_scorer"
    return "multiple_behaviors_same_scorer"


def find_moments(video_name, vframes, vfps, files, P):
    """files: [{path, name}]. P: rule, beh, a, b, scope, min, pad (s), gap (s), offset, fps (optional)."""
    warnings, per, last, fps_tsv = [], {}, 0, None
    names = [f["name"] for f in files]
    if len(set(names)) != len(names):
        raise ValueError("Two scorers have the same name. Give each a different name.")
    for f in files:
        r = read_boris(f["path"], warnings, f["name"])
        per[f["name"]] = [(b, s + P["offset"], e + P["offset"]) for b, s, e in r["intervals"]]
        fps_tsv = fps_tsv or r["fps"]
        for _, _, e in per[f["name"]]: last = max(last, e)
    fps = P.get("fps") or vfps or fps_tsv
    if not fps: raise ValueError("Could not read the frame rate. Enter it by hand.")
    total = max(vframes or 0, last + 1)
    if vframes and last + 1 > vframes + max(5, int(fps)):
        warnings.append("Some scored events end after the last frame of the video.")
    for k in ("beh", "a", "b"):
        if P.get(k): P[k] = norm(P[k])
    if P["rule"] == "behavior" and not P.get("beh"): raise ValueError("Pick a behavior.")
    if P["rule"] == "overlap" and not (P.get("a") and P.get("b")): raise ValueError("Pick both behaviors.")
    masks = build_masks(per, total)
    merged = merge_events(find_events(masks, total, P), int(round(P["gap"] * fps)))
    pad, rn, clips = int(round(P["pad"] * fps)), rule_name(P), []
    stem = re.sub(r"[^\w\-]+", "_", os.path.splitext(video_name)[0]).strip("_")[:60]
    w = max(3, len(str(len(merged))))
    for i, c in enumerate(merged, 1):
        cs, ce, behs = max(0, c["start"] - pad), min(total - 1, c["end"] + pad), sorted(c["behaviors"])
        lab = {n: labels_in_window(masks, n, c["start"], c["end"]) for n in names}
        clips.append(dict(
            id=f"CLIP{i:0{w}d}", rule=rn, es=c["start"], ee=c["end"], cs=cs, ce=ce, fps=round(float(fps), 3),
            scorers=sorted(c["scorers"]), behaviors=behs,
            sb="; ".join(f"{n}:{'+'.join(lab[n]) or 'none'}" for n in names),
            disagree="yes" if len({tuple(sorted(v)) for v in lab.values()}) > 1 else "no", merged=c["n"],
            file=f"CLIP{i:0{w}d}_{stem}_frames_{cs}-{ce}.mp4",
            tracks={n: [[b, x, y] for b, x, y in per[n] if y >= cs and x <= ce] for n in names}))
    return dict(clips=clips, warnings=warnings, fps=fps, total=total, rule_name=rn)


# ============================================================== reading video frames
class Video:
    """A video opened with OpenCV. Frames are read by exact frame number, whatever the format."""
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open the video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0)
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = 0
        self.lock = threading.Lock()

    def _seek(self, n):
        if n == self.pos: return
        if self.pos < n <= self.pos + 40:
            for _ in range(n - self.pos): self.cap.grab()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        self.pos = n

    def raw(self, start, count):
        """Yield (index, BGR frame)."""
        with self.lock:
            self._seek(start)
            for i in range(count):
                if self.n and start + i >= self.n: break
                ok, fr = self.cap.read()
                if not ok: break
                self.pos = start + i + 1
                yield start + i, fr

    def jpegs(self, start, count, maxw, quality):
        out = []
        for idx, fr in self.raw(start, count):
            if maxw and fr.shape[1] > maxw:
                fr = cv2.resize(fr, (maxw, int(fr.shape[0] * maxw / fr.shape[1])), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok: out.append((idx, buf.tobytes()))
        return out


_videos, _vlock = {}, threading.Lock()
def get_video(path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.splitext(path)[1].lower() not in VIDEO_EXT:
        raise ValueError("That does not look like a video file.")
    with _vlock:
        v = _videos.get(path)
        if v is None:
            if len(_videos) >= 4:
                old = next(iter(_videos)); _videos.pop(old).cap.release()
            v = _videos[path] = Video(path)
        return v


# ============================================================== saving clips
def find_font():
    for p in ["/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]:
        if os.path.exists(p): return p


def ffmpeg_can_draw():
    if not shutil.which("ffmpeg"): return False
    try: return b"drawtext" in subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.DEVNULL)
    except Exception: return False


def cut_ffmpeg(video, out, c0, c1, fps, ev_a, ev_b, overlay):
    font, vf = find_font(), []
    if overlay and font:
        ff = font.replace(":", "\\:")
        vf = ["-vf", (f"drawtext=fontfile='{ff}':text='frame %{{eif\\:n+{c0}\\:d}}':x=10:y=10:fontsize=h/22:fontcolor=white:"
                      f"box=1:boxcolor=black@0.55:boxborderw=6,drawtext=fontfile='{ff}':text='EVENT':x=w-tw-10:y=10:"
                      f"fontsize=h/22:fontcolor=white:box=1:boxcolor=red@0.7:boxborderw=6:enable='between(n,{ev_a - c0},{ev_b - c0})'")]
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{c0 / fps:.6f}", "-i", video, "-frames:v", str(c1 - c0 + 1), "-an", *vf,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()[:200]


def cut_opencv(video, out, c0, c1, fps):
    v = get_video(video)
    w = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (v.w, v.h))
    n = 0
    for _, fr in v.raw(c0, c1 - c0 + 1):
        w.write(fr); n += 1
    w.release()
    return n == c1 - c0 + 1, "" if n == c1 - c0 + 1 else f"only {n} frames could be read"


CLIP_COLS = ["clip_id", "source_video", "clip_filename", "rule_name", "event_start_frame", "event_end_frame", "clip_start_frame",
             "clip_end_frame", "fps", "event_start_sec", "event_end_sec", "scorers", "behaviors", "scorer_behaviors",
             "scorers_disagree", "events_merged", "notes"]


def save_package(video, out_dir, clips, settings, progress=None):
    """Cut every clip and write clips.csv, labels.json, settings.json. Returns a list of warnings."""
    os.makedirs(os.path.join(out_dir, "clips"), exist_ok=True)
    warnings, overlay = [], ffmpeg_can_draw()
    use_ff = bool(shutil.which("ffmpeg"))
    if use_ff and not overlay:
        warnings.append("Note: this ffmpeg cannot draw frame numbers on the clips, so they have none. The frame numbers are in the file names and in clips.csv.")
    if not use_ff:
        warnings.append("Note: ffmpeg was not found, so the clips were made with OpenCV. They play in this tool and in VLC or QuickTime, but may not play in a web browser.")
    rows = []
    for i, c in enumerate(clips):
        path = os.path.join(out_dir, "clips", c["file"])
        fps = c["fps"]
        ok, err = (cut_ffmpeg(video, path, c["cs"], c["ce"], fps, c["es"], c["ee"], overlay) if use_ff
                   else cut_opencv(video, path, c["cs"], c["ce"], fps))
        if not ok:
            ok, err = cut_opencv(video, path, c["cs"], c["ce"], fps)
        note = "" if ok else "clip not created"
        if not ok: warnings.append(f"{c['id']}: {err}")
        rows.append(dict(clip_id=c["id"], source_video=os.path.basename(video), clip_filename=c["file"], rule_name=c["rule"],
                         event_start_frame=c["es"], event_end_frame=c["ee"], clip_start_frame=c["cs"], clip_end_frame=c["ce"],
                         fps=c["fps"], event_start_sec=round(c["es"] / fps, 3), event_end_sec=round((c["ee"] + 1) / fps, 3),
                         scorers=";".join(c["scorers"]), behaviors=";".join(c["behaviors"]), scorer_behaviors=c["sb"],
                         scorers_disagree=c["disagree"], events_merged=c["merged"], notes=note))
        if progress: progress(i + 1, len(clips))
    with open(os.path.join(out_dir, "clips.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLIP_COLS); w.writeheader(); w.writerows(rows)
    fps0 = clips[0]["fps"] if clips else None
    with open(os.path.join(out_dir, "labels.json"), "w") as f:
        json.dump(dict(fps=fps0, scorers=sorted({s for c in clips for s in c["tracks"]}),
                       clips={c["id"]: dict(triggers=c["scorers"], tracks=c["tracks"]) for c in clips}), f)
    with open(os.path.join(out_dir, "settings.json"), "w") as f:
        json.dump(dict(settings, video=os.path.basename(video), clip_count=len(clips), warnings=warnings), f, indent=2)
    return warnings


def open_package(folder):
    folder = os.path.abspath(os.path.expanduser(folder))
    cf = os.path.join(folder, "clips.csv")
    if not os.path.exists(cf):
        raise ValueError("clips.csv was not found. Choose the folder that contains clips.csv and the clips folder.")
    with open(cf, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    labels = {}
    lf = os.path.join(folder, "labels.json")
    if os.path.exists(lf):
        with open(lf) as f: labels = json.load(f).get("clips", {})
    clips = []
    for r in rows:
        lab = labels.get(r["clip_id"], {})
        clips.append(dict(id=r["clip_id"], rule=r["rule_name"], es=int(r["event_start_frame"]), ee=int(r["event_end_frame"]),
                          cs=int(r["clip_start_frame"]), ce=int(r["clip_end_frame"]), fps=float(r["fps"]),
                          scorers=[x for x in r["scorers"].split(";") if x], behaviors=[x for x in r["behaviors"].split(";") if x],
                          sb=r["scorer_behaviors"], disagree=r.get("scorers_disagree", ""), merged=int(r.get("events_merged") or 1),
                          file=r["clip_filename"], path=os.path.join(folder, "clips", r["clip_filename"]),
                          tracks=lab.get("tracks") if lab else None))
    return dict(clips=clips, folder=folder, reviews=read_reviews(folder))


def read_reviews(folder):
    """Saved review files (review_<name>.csv) inside a folder, oldest first."""
    folder = os.path.abspath(os.path.expanduser(folder or ""))
    if not folder or not os.path.isdir(folder): return []
    out = []
    for p in sorted(glob.glob(os.path.join(folder, "review_*.csv")), key=os.path.getmtime):
        try:
            with open(p, encoding="utf-8-sig") as f: out.append(dict(file=os.path.basename(p), text=f.read()))
        except OSError: pass
    return out


def save_review(folder, filename, text):
    folder = os.path.abspath(os.path.expanduser(folder or ""))
    name = re.sub(r"[^\w\-.]+", "_", os.path.basename(filename or ""))
    if not (name.startswith("review_") and name.endswith(".csv")): raise ValueError("Bad review file name.")
    try:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, name), "w", encoding="utf-8", newline="") as f: f.write(text)
    except OSError as err:
        raise ValueError(f"Could not save in this folder: {err}")
    return dict(file=name, folder=folder)


SKIP_DIRS = {"library", "node_modules", "__pycache__", "site-packages", "$recycle.bin", "appdata", "system volume information", "proc", "sys", "dev"}


def search_files(root, query, kind, exts, limit=300, tmax=12.0):
    """Find files (or folders) whose name contains every word of the query. Newest first."""
    root = os.path.abspath(os.path.expanduser(root or "~"))
    if not os.path.isdir(root): raise ValueError("Cannot search there.")
    words = [w for w in query.lower().split() if w]
    if not words: return dict(items=[], truncated=False, root=root)
    exts = [e.lower() for e in exts if e]
    t0, items, truncated = time.time(), [], False
    for dp, dns, fns in os.walk(root, onerror=lambda e: None):
        dns[:] = [d for d in dns if not d.startswith(".") and d.lower() not in SKIP_DIRS and not d.lower().endswith((".app", ".photoslibrary"))]
        for n in (dns if kind == "dir" else fns):
            low = n.lower()
            if n.startswith(".") or not all(w in low for w in words): continue
            if kind == "file" and exts and not low.endswith(tuple(exts)): continue
            p = os.path.join(dp, n)
            try: st = os.stat(p)
            except OSError: continue
            items.append(dict(name=n, path=p, dir=dp, size=st.st_size, mtime=st.st_mtime))
        if len(items) >= limit or time.time() - t0 > tmax: truncated = True; break
    items.sort(key=lambda x: -x["mtime"])
    return dict(items=items[:limit], truncated=truncated, root=root)


# ============================================================== web server
HTML = None  # filled in at the bottom of the file
TOKEN = secrets.token_urlsafe(16)
JOBS = {}


def list_dir(path):
    path = os.path.abspath(os.path.expanduser(path or "~"))
    if not os.path.isdir(path): path = os.path.dirname(path) if os.path.isdir(os.path.dirname(path)) else os.path.expanduser("~")
    dirs, files = [], []
    try:
        for e in sorted(os.scandir(path), key=lambda e: e.name.lower()):
            if e.name.startswith("."): continue
            try:
                if e.is_dir(): dirs.append(e.name)
                else: files.append(dict(name=e.name, size=e.stat().st_size))
            except OSError: pass
    except OSError as err:
        raise ValueError(f"Cannot open this folder: {err}")
    return dict(path=path, parent=os.path.dirname(path), dirs=dirs, files=files, home=os.path.expanduser("~"), cwd=os.getcwd())


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    start_dir = "."

    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def _ok(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host not in ("127.0.0.1", "localhost"): self._send(403, {"error": "forbidden"}); return False
        q = parse_qs(urlparse(self.path).query)
        tok = self.headers.get("X-Token") or (q.get("t") or [""])[0]
        if not secrets.compare_digest(tok, TOKEN): self._send(403, {"error": "forbidden"}); return False
        return True

    def do_GET(self):
        if not self._ok(): return
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                return self._send(200, HTML.replace("__START_DIR__", json.dumps(os.path.abspath(Handler.start_dir))), "text/html; charset=utf-8")
            if u.path == "/api/ls": return self._send(200, list_dir(q.get("path")))
            if u.path == "/api/search":
                return self._send(200, search_files(q.get("path"), q.get("q", ""), q.get("kind", "file"), (q.get("exts") or "").split(",")))
            if u.path == "/api/open":
                v = get_video(q["path"])
                return self._send(200, dict(path=v.path, frames=v.n, fps=v.fps, w=v.w, h=v.h, name=os.path.basename(v.path)))
            if u.path == "/api/frames":
                v = get_video(q["path"])
                fr = v.jpegs(int(q["start"]), min(int(q.get("count", 30)), 200), int(q.get("maxw", 960)), int(q.get("q", 82)))
                return self._send(200, b"".join(struct.pack("<II", i, len(b)) + b for i, b in fr), "application/octet-stream")
            if u.path == "/api/job":
                return self._send(200, JOBS.get(q.get("id"), {"error": "unknown job"}))
            self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(400, {"error": str(e)})

    def do_POST(self):
        if not self._ok(): return
        u = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            if u.path == "/api/scoring": return self._send(200, describe_scoring(body["paths"]))
            if u.path == "/api/find":
                v = get_video(body["video"])
                P = dict(rule=body["rule"], beh=body.get("beh"), a=body.get("a"), b=body.get("b"), scope=body.get("scope", "same"),
                         min=max(1, int(body.get("min", 5))), pad=float(body.get("pad", 1)), gap=float(body.get("gap", 1)),
                         offset=int(body.get("offset", 0)), fps=float(body["fps"]) if body.get("fps") else None)
                return self._send(200, find_moments(os.path.basename(v.path), v.n, v.fps, body["files"], P))
            if u.path == "/api/folder": return self._send(200, open_package(body["path"]))
            if u.path == "/api/reviews": return self._send(200, dict(files=read_reviews(body.get("folder"))))
            if u.path == "/api/savereview": return self._send(200, save_review(body.get("folder"), body.get("file"), body.get("text", "")))
            if u.path == "/api/export":
                outp = os.path.abspath(os.path.expanduser(body.get("out") or ""))
                if not body.get("out"): raise ValueError("Choose a folder and a name for the new review folder first.")
                if os.path.exists(os.path.join(outp, "clips.csv")):
                    raise ValueError("A review folder with this name already exists. Change the name, or delete that folder first.")
                jid = secrets.token_hex(4); JOBS[jid] = dict(done=0, total=len(body["clips"]), finished=False)
                def work():
                    try:
                        def prog(i, n): JOBS[jid].update(done=i, total=n)
                        w = save_package(os.path.abspath(os.path.expanduser(body["video"])), os.path.abspath(os.path.expanduser(body["out"])),
                                         body["clips"], body.get("settings", {}), prog)
                        JOBS[jid].update(finished=True, warnings=w, out=os.path.abspath(os.path.expanduser(body["out"])))
                    except Exception as e:
                        JOBS[jid].update(finished=True, error=str(e))
                threading.Thread(target=work, daemon=True).start()
                return self._send(200, dict(id=jid))
            self._send(404, {"error": "not found"})
        except KeyError as e:
            self._send(400, {"error": f"missing {e}"})
        except Exception as e:
            self._send(400, {"error": str(e)})


def serve(args):
    Handler.start_dir = args.dir
    port = args.port
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler); port = p; break
        except OSError: continue
    else:
        sys.exit("Could not find a free port.")
    url = f"http://127.0.0.1:{port}/?t={TOKEN}"
    print(f"\nEthoClips is running.\nOpen this link in your browser:\n\n  {url}\n\nPress Ctrl+C here to stop.\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: print("\nStopped.")


def cut_command(a):
    tsvs = []
    for p in a.tsv: tsvs += sorted(glob.glob(os.path.expanduser(p))) or [p]
    names = a.names or guess_names(tsvs)
    if len(names) != len(tsvs): sys.exit("--names must have one name per TSV file.")
    v = get_video(a.video)
    P = dict(rule=a.rule, beh=a.behavior, a=a.a, b=a.b, scope=a.scope, min=a.min_frames, pad=a.pad, gap=a.merge_gap, offset=a.frame_offset, fps=None)
    res = find_moments(os.path.basename(v.path), v.n, v.fps, [dict(path=p, name=n) for p, n in zip(tsvs, names)], P)
    print(f"Rule: {res['rule_name']}\n  {len(res['clips'])} moments")
    if not res["clips"]: return
    w = save_package(v.path, a.out, res["clips"], dict(P, scorers=dict(zip(names, map(os.path.basename, tsvs)))),
                     lambda i, n: print(f"  clip {i}/{n}", end="\r"))
    for x in res["warnings"] + w: print("Warning:", x)
    print(f"\nSaved to {os.path.abspath(a.out)}")


def main():
    ap = argparse.ArgumentParser(description="Find moments in scored videos and review them.", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--dir", default=os.getcwd()); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--no-browser", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("cut", help="save clips without opening a page")
    c.add_argument("rule", choices=["behavior", "overlap", "multi"]); c.add_argument("--video", required=True); c.add_argument("--tsv", nargs="+", required=True)
    c.add_argument("--out", required=True); c.add_argument("--names", nargs="+"); c.add_argument("--behavior"); c.add_argument("--a"); c.add_argument("--b")
    c.add_argument("--scope", choices=["same", "different"], default="same"); c.add_argument("--min-frames", type=int, default=5)
    c.add_argument("--pad", type=float, default=1.0); c.add_argument("--merge-gap", type=float, default=1.0); c.add_argument("--frame-offset", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "cut":
        if a.rule == "behavior" and not a.behavior: sys.exit("rule 'behavior' needs --behavior")
        if a.rule == "overlap" and not (a.a and a.b): sys.exit("rule 'overlap' needs --a and --b")
        if a.behavior: a.behavior = norm(a.behavior)
        if a.a: a.a, a.b = norm(a.a), norm(a.b)
        cut_command(a)
    else:
        serve(a)


# ============================================================== the page
HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EthoClips</title>
<style>
  :root { --bg:#f6f7f9; --panel:#fff; --text:#1d2330; --muted:#667085; --line:#dde1e8; --accent:#0284c7; --good:#1f8a4c; --bad:#c0392b; --warn:#b7791f; --sel:#e3f3fc; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#14171c; --panel:#1c2027; --text:#e8eaee; --muted:#9aa3b2; --line:#2e343e; --accent:#4cb6ff; --good:#4cc384; --bad:#ef7b6e; --warn:#e0b050; --sel:#1b2a38; } }
  :root[data-theme="dark"] { --bg:#14171c; --panel:#1c2027; --text:#e8eaee; --muted:#9aa3b2; --line:#2e343e; --accent:#4cb6ff; --good:#4cc384; --bad:#ef7b6e; --warn:#e0b050; --sel:#1b2a38; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; padding:10px 16px; background:var(--panel); border-bottom:1px solid var(--line); }
  header h1 { font-size:16px; margin:0; }
  button, select, input[type=text], input[type=number] { font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:6px 10px; }
  button { cursor:pointer; } button:hover:not(:disabled) { border-color:var(--accent); } button:disabled { opacity:.45; cursor:default; }
  button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
  .spacer { flex:1; } .hint { color:var(--muted); font-size:12.5px; } .err { color:var(--bad); } .ok { color:var(--good); }
  .row { display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; align-items:center; }

  details#setup { margin:12px 16px; background:var(--panel); border:1px solid var(--line); border-radius:10px; }
  details#setup > summary { cursor:pointer; padding:10px 14px; font-weight:600; list-style:none; }
  details#setup > summary::before { content:"›"; display:inline-block; margin-right:8px; transition:transform .15s; }
  details#setup[open] > summary::before { transform:rotate(90deg); }
  #sumline { font-weight:400; color:var(--muted); margin-left:8px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; padding:0 14px 14px; }
  .card { border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--bg); }
  .card h3 { margin:0 0 8px; font-size:13.5px; }
  .card label { display:block; margin:8px 0 3px; font-size:12.5px; color:var(--muted); }
  .card select, .card input[type=number] { width:100%; }
  .scorer { display:flex; gap:6px; align-items:center; margin:5px 0; }
  .scorer input { width:96px; } .scorer .fn { flex:1; color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .scorer button { padding:2px 8px; }
  .two { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .seg2 { display:inline-flex; } .seg2 button { border-radius:0; padding:5px 10px; font-size:12.5px; margin-left:-1px; } .seg2 button:first-child { border-radius:6px 0 0 6px; margin-left:0; } .seg2 button:last-child { border-radius:0 6px 6px 0; }
  .seg2 button.on { background:var(--accent); color:#fff; border-color:var(--accent); position:relative; }
  #ruleSentence { margin:10px 0 0; padding:8px 10px; background:var(--panel); border-radius:6px; font-size:12.5px; border:1px solid var(--line); }

  #app { display:none; grid-template-columns:340px 1fr; height:calc(100vh - 130px); min-height:520px; border-top:1px solid var(--line); }
  #list { overflow:auto; border-right:1px solid var(--line); background:var(--panel); }
  #filters { display:flex; flex-wrap:wrap; gap:6px; padding:8px; border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--panel); z-index:1; }
  #filters select { flex:1; min-width:100px; }
  .item { padding:8px 12px; border-bottom:1px solid var(--line); cursor:pointer; display:flex; gap:8px; }
  .item:hover { background:var(--bg); } .item.sel { background:var(--sel); box-shadow:inset 3px 0 0 var(--accent); }
  .dot { width:10px; height:10px; border-radius:50%; margin-top:5px; flex:none; background:var(--line); }
  .dot.real { background:var(--good); } .dot.conflict { background:var(--bad); } .dot.mislabeled { background:#8e6bd0; } .dot.unsure { background:var(--warn); } .dot.note { background:var(--muted); } .item.gone { opacity:.45; } .item.gone .t { text-decoration:line-through; }
  .item .t { font-weight:600; } .item .s { color:var(--muted); font-size:12.5px; }
  #main { overflow:auto; padding:16px; }
  #stage { position:relative; background:#000; border-radius:8px; overflow:hidden; text-align:center; }
  #cv { max-width:100%; max-height:46vh; display:block; margin:0 auto; }
  #stageMsg { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#ddd; font-size:13px; padding:20px; text-align:center; pointer-events:none; }
  #ctl { display:flex; align-items:center; gap:10px; margin-top:8px; font-variant-numeric:tabular-nums; }
  #scrub { flex:1; accent-color:var(--accent); }
  #chip { padding:1px 9px; border-radius:10px; font-size:12px; font-weight:700; background:var(--line); color:var(--muted); }
  #chip.on { background:var(--bad); color:#fff; }
  .speed button.on { background:var(--accent); color:#fff; border-color:var(--accent); }
  #overview { position:relative; height:26px; background:var(--panel); border:1px solid var(--line); border-radius:6px; margin:6px 0 2px; cursor:pointer; overflow:hidden; }
  #overview .tick { position:absolute; top:5px; bottom:5px; background:var(--accent); opacity:.55; min-width:2px; border-radius:1px; }
  #overview .tick.cur { opacity:1; top:2px; bottom:2px; }
  #overview .win { position:absolute; top:0; bottom:0; border:1.5px solid var(--accent); border-radius:3px; background:color-mix(in srgb, var(--accent) 10%, transparent); }
  #overview .head { position:absolute; top:0; bottom:0; width:2px; background:var(--text); }
  .tl { position:relative; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:24px 10px 8px; margin-top:6px; cursor:pointer; user-select:none; }
  .tlrow { display:flex; align-items:center; margin:3px 0; }
  .tlname { width:78px; flex:none; font-size:12.5px; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .tlname.trig::before { content:""; display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--accent); margin-right:5px; }
  .tltrack { position:relative; flex:1; background:var(--bg); border-radius:5px; }
  .seg { position:absolute; border-radius:3px; font-size:10.5px; line-height:14px; padding:0 4px; overflow:hidden; white-space:nowrap; }
  .tllayer { position:absolute; top:24px; bottom:8px; left:88px; right:10px; pointer-events:none; }
  .evband { position:absolute; top:0; bottom:0; background:color-mix(in srgb, var(--accent) 14%, transparent); border-left:1.5px dashed var(--accent); border-right:1.5px dashed var(--accent); }
  .evtag { position:absolute; top:-18px; font-size:11.5px; color:var(--accent); font-weight:600; white-space:nowrap; }
  .playhead { position:absolute; top:-4px; bottom:-2px; width:2px; background:var(--text); }
  .tlaxis { display:flex; justify-content:space-between; margin:4px 0 0 78px; color:var(--muted); font-size:11.5px; }
  #legend { display:flex; flex-wrap:wrap; gap:4px 12px; margin:8px 0 0; font-size:12px; color:var(--muted); }
  #legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:5px; vertical-align:-1px; }
  #info { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px 14px; margin-top:12px; }
  #info dl { display:grid; grid-template-columns:130px 1fr; gap:4px 12px; margin:0; } #info dt { color:var(--muted); } #info dd { margin:0; word-break:break-word; }
  .badge { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600; }
  .badge.dis { background:color-mix(in srgb, var(--bad) 16%, transparent); color:var(--bad); } .badge.agr { background:color-mix(in srgb, var(--good) 16%, transparent); color:var(--good); }
  .labels button { padding:8px 12px; } .labels button.on { color:#fff; border-color:transparent; }
  .labels button.on.real { background:var(--good); } .labels button.on.conflict { background:var(--bad); } .labels button.on.mislabeled { background:#8e6bd0; } .labels button.on.unsure { background:var(--warn); }
  textarea { width:100%; min-height:64px; font:inherit; color:var(--text); background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px; }
  details.share { margin-top:14px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 14px; }
  details.share summary { cursor:pointer; font-weight:600; }
  details.share input[type=text] { width:100%; }

  .tip { position:relative; display:inline-block; }
  .ib { display:inline-flex; width:18px; height:18px; border-radius:50%; border:1px solid var(--muted); color:var(--muted); align-items:center; justify-content:center; font:italic 12px/1 Georgia,serif; cursor:help; margin-left:6px; vertical-align:middle; }
  .tip .pop { display:none; position:absolute; z-index:20; top:26px; left:0; width:290px; background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:8px 10px; font:12.5px/1.45 -apple-system,"Segoe UI",Roboto,sans-serif; color:var(--text); box-shadow:0 4px 14px rgba(0,0,0,.22); text-align:left; }
  .tip.r .pop { left:auto; right:0; }
  .tip:hover .pop, .tip.open .pop { display:block; }
  #loadNote { padding:6px 12px; } #loadNote:empty { display:none; }
  #boxQrow { display:flex; gap:6px; padding:0 14px 8px; } #boxQ { flex:1; }
  .ent .dir { margin-left:auto; color:var(--muted); font-size:12px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:48%; }
  .ent .dir + .sz { margin-left:8px; }
  /* file chooser */
  #modal { position:fixed; inset:0; background:rgba(0,0,0,.45); display:none; align-items:center; justify-content:center; z-index:10; }
  #box { width:min(640px,94vw); max-height:84vh; background:var(--panel); border-radius:10px; display:flex; flex-direction:column; border:1px solid var(--line); }
  #box header { border:0; padding:12px 14px 6px; } #boxPath { flex:1; min-width:200px; }
  #boxList { overflow:auto; border-top:1px solid var(--line); border-bottom:1px solid var(--line); min-height:240px; flex:1; }
  .ent { padding:6px 14px; cursor:pointer; display:flex; gap:8px; align-items:center; }
  .ent:hover { background:var(--bg); } .ent.sel { background:var(--sel); } .ent .sz { margin-left:auto; color:var(--muted); font-size:12px; }
  .ent.dim { color:var(--muted); }
  #boxFoot { display:flex; gap:8px; align-items:center; padding:10px 14px; }
  @media (max-width:800px) { #app { grid-template-columns:1fr; height:auto; } #list { max-height:240px; } }
</style>
</head>
<body>
<header>
  <h1>EthoClips</h1>
  <span id="counts" class="hint"></span>
  <span class="spacer"></span>
  <span id="saveNote" class="hint"></span>
  <span class="tip r"><button id="pickFolder">Open review folder…</button><span class="ib" tabindex="0">i</span><span class="pop">Open a review folder: a folder of clips made by this tool, one you created earlier or one someone sent you. The review saved in it (review_&lt;name&gt;.csv) is loaded too. You don't need the video or the scoring files.</span></span>
  <button id="export" class="primary" style="display:none" title="Saves review_&lt;name&gt;.csv inside the review folder">Save review</button>
  <button id="theme" title="Light / dark">◐</button>
</header>

<details id="setup" open>
  <summary>Set up <span id="sumline"></span></summary>
  <div class="cards">
    <div class="card">
      <h3>1 · Video and scoring files</h3>
      <button id="pickVideo">Choose video…</button> <span id="videoName" class="hint">no video yet</span>
      <p id="videoInfo" class="hint" style="margin:6px 0 0"></p>
      <div style="height:8px"></div>
      <button id="pickTsv">Add scoring files (TSV)…</button>
      <div id="scorers"></div>
      <p id="loadMsg" class="hint"></p>
    </div>
    <div class="card">
      <h3>2 · What to find</h3>
      <div class="seg2" id="ruleTabs">
        <button data-r="behavior" class="on">A behavior</button><button data-r="overlap">Two behaviors together</button><button data-r="multi">Several at once</button>
      </div>
      <div id="rBehavior"><label>Behavior marked by any scorer</label><select id="selBeh"></select></div>
      <div id="rOverlap" style="display:none">
        <div class="two"><div><label>Behavior A</label><select id="selA"></select></div><div><label>Behavior B</label><select id="selB"></select></div></div>
        <label>Who marked them</label>
        <select id="selScope"><option value="same">The same scorer marked both</option><option value="different">Different scorers (one marked A, another B)</option></select>
      </div>
      <div id="rMulti" style="display:none"><p class="hint" style="margin:10px 0 0">Finds frames where any scorer has two or more behaviors at the same time.</p></div>
      <div class="two">
        <div><label>Shortest overlap (frames)</label><input type="number" id="minFrames" value="5" min="1"></div>
        <div><label>Context before/after (s)</label><input type="number" id="pad" value="1" min="0" step="0.5"></div>
      </div>
      <details style="margin-top:8px"><summary class="hint" style="cursor:pointer">More options</summary>
        <label>Join moments closer than (s)</label><input type="number" id="gap" value="1" min="0" step="0.5">
        <label>Frame offset (add to every scoring frame number)</label><input type="number" id="offset" value="0" step="1">
      </details>
      <div id="ruleSentence"></div>
      <div class="row"><button class="primary" id="find" disabled>Find moments</button><span id="findMsg" class="hint"></span></div>
    </div>
  </div>
</details>

<div id="app">
  <div id="list">
    <div id="filters">
      <select id="fScorer"><option value="">All scorers</option></select>
      <select id="fStatus"><option value="">All moments</option><option value="todo">Not reviewed</option><option value="done">Reviewed</option><option value="dis">Scorers disagree</option><option value="gone">Removed from export</option></select>
    </div>
    <div id="loadNote" class="hint"></div>
    <div id="items"></div>
  </div>
  <div id="main">
    <div id="stage"><canvas id="cv" width="640" height="360"></canvas><div id="stageMsg"></div></div>
    <div id="ctl">
      <button id="play" style="width:42px">⏸</button>
      <input type="range" id="scrub" min="0" max="1" value="0" step="1">
      <span id="frameTxt" class="hint" style="min-width:150px;text-align:right"></span><span id="chip" title="Red while the video is inside the moment the rule found. Before and after it you see the extra seconds of context.">EVENT</span>
    </div>
    <div class="row" style="margin:6px 0 0">
      <span class="speed"><button data-s="1" class="on">1x</button> <button data-s="0.5">0.5x</button> <button data-s="0.25">0.25x</button></span>
      <span class="spacer"></span>
      <button id="rm" title="Leave this moment out of the saved clips folder (X)">Remove from export</button><button id="prev">◀ Previous (K)</button><button id="next">Next (J) ▶</button><span class="tip r"><span class="ib" tabindex="0">i</span><span class="pop">Keys: J next, K previous, Space play/pause, ← → one frame back/forward, 1–4 mark, X remove from export.<br>Click the timeline to jump.</span></span>
    </div>
    <div id="ovwrap"><div class="hint" style="margin-top:8px">Where in the video</div><div id="overview"></div></div>
    <div class="row" style="margin-bottom:0"><span class="hint">What each scorer marked</span><span class="spacer"></span>
      <span class="seg2"><button id="tlAll" class="on">All behaviors</button><button id="tlRule">Rule behaviors only</button></span></div>
    <div class="tl" id="tl"></div>
    <div id="legend"></div>
    <div id="info"><dl id="dl"></dl></div>
    <div class="row" style="margin-bottom:0"><label class="hint">Reviewer <input type="text" id="reviewer" placeholder="your name" size="16"></label></div>
    <div class="row labels" id="labels"></div>
    <textarea id="note" placeholder="Note (optional)"></textarea>
    <details class="share" id="shareBox"><summary>Create a review folder to share (optional)</summary>
      <p class="hint">Makes a review folder with the clips, <code>clips.csv</code> and the scorers' labels. Needs ffmpeg for the best quality.</p>
      <label class="hint">Save inside this folder</label>
      <div class="row" style="margin:2px 0 8px"><input type="text" id="outDir" style="flex:1;min-width:200px"><button id="browseOut">Browse…</button></div>
      <label class="hint">Name of the new review folder (filled in from the rule you chose, you can change it)</label>
      <input type="text" id="outName" style="width:100%;margin-top:2px">
      <p class="hint" id="outFull" style="margin:6px 0"></p>
      <div class="row"><button id="saveClips" class="primary">Create review folder</button><span id="saveMsg" class="hint"></span></div>
      <p class="hint" id="saveWarn" style="margin:4px 0 0"></p>
    </details>
  </div>
</div>

<div id="modal"><div id="box">
  <header><strong id="boxTitle">Choose</strong><input type="text" id="boxPath"><button id="boxUp">↑ Up</button><button id="boxHome">Home</button><button id="boxCwd">Start folder</button></header>
  <div id="boxQrow"><input type="text" id="boxQ" placeholder="Search by name, e.g. m1917"><select id="boxWhere"><option value="home">In my home folder</option><option value="here">In this folder</option></select></div>
  <div id="boxList"></div>
  <div id="boxFoot"><span class="hint" id="boxHint"></span><span class="spacer"></span><button id="boxCancel">Cancel</button><button class="primary" id="boxOk">Select</button></div>
</div></div>

<script>
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[c]));
  const TOKEN = new URLSearchParams(location.search).get("t") || "";
  const START_DIR = __START_DIR__;
  const LABELS = [["real", "Valid event", "The behaviors really happen together in the video."], ["conflict", "Contradictory labels", "The labels cannot both be true at once (for example rearing and locomotion)."],
    ["mislabeled", "Mislabeled", "What a scorer marked does not match what the video shows."], ["unsure", "Unsure", "Not sure. Look again later."]];
  const OLD_LABELS = { "Real event": "real", "Wrong behavior": "mislabeled" };
  const COLORS = { rearing:"#6C63FF", supported_rearing:"#6C63FF", unsupported_rearing:"#8E88FF", exploration_locomotive:"#5AC8FA", exploration_stationary:"#FFD60A",
    face_swipe:"#FF2D92", walking:"#34C759", resting:"#AEAEB2", face_grooming:"#BF5AF2", body_grooming:"#FF9F0A", digging:"#30B0C7", push_digging:"#1D8FA3",
    unchained_grooming:"#D98BFF", freezing:"#8E8E93", unclear:"#E5484D", other:"#C7A27C", climbing:"#8D6E63", body_flick:"#FF7043", ear_flick:"#F06292", defecation:"#795548" };
  const FALL = ["#4F8EF7","#F78C4F","#45B58A","#B5457F","#7F9A2E","#A3702F","#3B7A8C","#9B59B6"], fallMap = {};
  const colorOf = b => COLORS[b] || (fallMap[b] = fallMap[b] || FALL[Object.keys(fallMap).length % FALL.length]);
  const ink = hex => { const n = parseInt(hex.slice(1), 16); return (0.299*(n >> 16) + 0.587*((n >> 8) & 255) + 0.114*(n & 255)) > 150 ? "#1d2330" : "#fff"; };
  const pretty = b => String(b).replace(/_/g, " ");
  const dirname = p => p.replace(/[\\/][^\\/]*$/, "") || "/";
  const basename = p => p.split(/[\\/]/).pop();
  const sepOf = p => p.includes("\\") && !p.includes("/") ? "\\" : "/";

  async function api(path, body) {
    const opt = body === undefined ? { headers: { "X-Token": TOKEN } } : { method: "POST", headers: { "X-Token": TOKEN, "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(path, opt); const j = await r.json().catch(() => ({ error: "bad reply" }));
    if (!r.ok || j.error) throw new Error(j.error || "request failed"); return j;
  }

  // ------------------------------------------------------------ state
  let video = null;            // { path, name, frames, fps, w, h }
  let scorers = [];            // { path, name, file, events, fps, media, mdur, behaviors }
  let removed = new Set(), clips = [], mode = "video", folderDir = "", cur = -1, review = {}, showAll = true, rule = "behavior", shown = [], lastRule = null;

  try { const t = localStorage.getItem("clipTheme"); if (t) document.documentElement.dataset.theme = t; $("reviewer").value = localStorage.getItem("clipReviewer") || ""; } catch (e) {}
  $("theme").onclick = () => { const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() === "#14171c";
    const t = dark ? "light" : "dark"; document.documentElement.dataset.theme = t; try { localStorage.setItem("clipTheme", t); } catch (e) {} };
  $("reviewer").oninput = () => { try { localStorage.setItem("clipReviewer", $("reviewer").value); } catch (e) {} };

  // ------------------------------------------------------------ file chooser (runs on the computer that runs the tool)
  let boxState = null, lastDir = START_DIR, searchSeq = 0, searchTimer = 0;
  const joinP = (d, n) => d.replace(/[\\/]$/, "") + sepOf(d) + n;
  const mb = s => (s / 1048576).toFixed(s > 1e7 ? 0 : 1) + " MB";
  function chooser({ title, exts, multi, dirMode, start }) {
    return new Promise(resolve => {
      boxState = { exts, multi, dirMode, sel: new Set(), path: start || lastDir, resolve, data: null, results: null };
      boxState.hint = dirMode ? "Open the folder you want, then click Use this folder. Or search for it by name." : multi ? "Click files to select them (several allowed). Or search by name." : "Click a file to select it. Or search by name, for example m1917.";
      $("boxTitle").textContent = title; $("modal").style.display = "flex"; $("boxQ").value = ""; $("boxHint").textContent = boxState.hint;
      $("boxOk").textContent = dirMode ? "Use this folder" : "Select"; goDir(boxState.path);
    });
  }
  async function goDir(path) {
    try { const d = await api("/api/ls?path=" + encodeURIComponent(path)); boxState.data = d; boxState.path = d.path; lastDir = d.path; boxState.results = null; $("boxQ").value = ""; $("boxHint").textContent = boxState.hint; $("boxPath").value = d.path; renderBox(); }
    catch (e) { $("boxHint").textContent = e.message; }
  }
  function refreshOk() { $("boxOk").textContent = boxState.dirMode ? "Use this folder" : "Select" + (boxState.sel.size ? ` (${boxState.sel.size})` : ""); }
  function renderBox() {
    const d = boxState.data, R = boxState.results, ok = f => !boxState.exts || boxState.exts.some(x => f.toLowerCase().endsWith(x)); let h = "";
    const file = (p, name, extra) => `<div class="ent ${ok(name) ? "" : "dim"} ${boxState.sel.has(p) ? "sel" : ""}" data-p="${esc(p)}">📄 ${esc(name)}${extra}</div>`;
    if (R) {
      h = R.items.map(x => boxState.dirMode ? `<div class="ent" data-d="${esc(x.path)}">📁 ${esc(x.name)}<span class="dir" title="${esc(x.dir)}">${esc(x.dir)}</span></div>`
        : file(x.path, x.name, `<span class="dir" title="${esc(x.dir)}">${esc(x.dir)}</span><span class="sz">${mb(x.size)}</span>`)).join("");
      if (!h) h = '<div class="ent dim">Nothing found. Try a shorter word, or search in your home folder.</div>';
      if (R.truncated) h += '<div class="ent dim">Showing the first results only. Type more of the name to narrow it down.</div>';
    } else {
      h = d.dirs.map(n => `<div class="ent" data-d="${esc(joinP(d.path, n))}">📁 ${esc(n)}</div>`).join("");
      if (!boxState.dirMode) h += d.files.map(f => file(joinP(d.path, f.name), f.name, `<span class="sz">${mb(f.size)}</span>`)).join("");
      if (!h) h = '<div class="ent dim">(empty)</div>';
    }
    $("boxList").innerHTML = h;
    $("boxList").querySelectorAll(".ent[data-d]").forEach(el => el.onclick = () => goDir(el.dataset.d));
    $("boxList").querySelectorAll(".ent[data-p]").forEach(el => {
      el.onclick = () => { const p = el.dataset.p; if (!boxState.multi) boxState.sel.clear(); boxState.sel.has(p) ? boxState.sel.delete(p) : boxState.sel.add(p);
        $("boxList").querySelectorAll(".ent[data-p]").forEach(e2 => e2.classList.toggle("sel", boxState.sel.has(e2.dataset.p))); refreshOk(); };
      el.ondblclick = () => { if (!boxState.multi) { boxState.sel = new Set([el.dataset.p]); $("boxOk").click(); } };
    });
    refreshOk();
  }
  async function runSearch() {
    if (!boxState || !boxState.data) return; const q = $("boxQ").value.trim();
    if (q.length < 2) { boxState.results = null; $("boxHint").textContent = boxState.hint; renderBox(); return; }
    const seq = ++searchSeq, root = $("boxWhere").value === "home" ? boxState.data.home : boxState.data.path; $("boxHint").textContent = "Searching…";
    try {
      const r = await api("/api/search?path=" + encodeURIComponent(root) + "&q=" + encodeURIComponent(q) + "&kind=" + (boxState.dirMode ? "dir" : "file") + "&exts=" + encodeURIComponent((boxState.exts || []).join(",")));
      if (seq !== searchSeq || !boxState) return;
      boxState.results = r; $("boxHint").textContent = `${r.items.length}${r.truncated ? "+" : ""} found in ${r.root}`; renderBox();
    } catch (e) { if (seq === searchSeq && boxState) $("boxHint").textContent = e.message; }
  }
  $("boxQ").oninput = () => { clearTimeout(searchTimer); searchTimer = setTimeout(runSearch, 450); };
  $("boxQ").onkeydown = e => { if (e.key === "Enter") { clearTimeout(searchTimer); runSearch(); } };
  $("boxWhere").onchange = runSearch;
  $("boxPath").onkeydown = e => { if (e.key === "Enter") goDir($("boxPath").value); };
  $("boxUp").onclick = () => goDir(boxState.data.parent); $("boxHome").onclick = () => goDir(boxState.data.home); $("boxCwd").onclick = () => goDir(START_DIR);
  const closeBox = v => { $("modal").style.display = "none"; const r = boxState.resolve; boxState = null; r(v); };
  $("boxCancel").onclick = () => closeBox(null);
  $("boxOk").onclick = () => {
    if (boxState.dirMode) return closeBox(boxState.data.path);
    if (!boxState.sel.size) { $("boxHint").textContent = "Click a file first."; return; }
    closeBox([...boxState.sel]);
  };
  document.addEventListener("keydown", e => { if (e.key === "Escape" && boxState) closeBox(null); }, true);
  document.addEventListener("click", e => { document.querySelectorAll(".tip.open").forEach(t => { if (!t.contains(e.target)) t.classList.remove("open"); }); const ib = e.target.closest && e.target.closest(".ib"); if (ib) ib.parentElement.classList.toggle("open"); });

  // ------------------------------------------------------------ step 1
  $("pickVideo").onclick = async () => {
    const r = await chooser({ title: "Choose the video", exts: [".avi", ".mp4", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".webm"], multi: false }); if (!r) return;
    $("videoName").textContent = "opening…"; $("videoName").className = "hint";
    try {
      const v = await api("/api/open?path=" + encodeURIComponent(r[0])); video = v; clearBuffers();
      $("videoName").textContent = v.name; $("videoInfo").textContent = `${v.frames} frames · ${v.fps.toFixed(3)} fps · ${v.w}×${v.h} · ${(v.frames / v.fps).toFixed(1)} s`;
      review = {}; removed = new Set();
      $("outDir").value = dirname(v.path) + sepOf(v.path) + "clips_" + stemOf(v.name).slice(0, 40); $("outName").value = ""; updateOutFull();
    } catch (e) { video = null; $("videoName").textContent = ""; $("loadMsg").className = "err"; $("loadMsg").textContent = e.message; }
    renderScorers(); updateReady();
  };
  $("pickTsv").onclick = async () => {
    const r = await chooser({ title: "Choose the scoring files (BORIS TSV), one per scorer", exts: [".tsv", ".txt", ".csv"], multi: true, start: video ? dirname(video.path) : undefined }); if (!r) return;
    $("loadMsg").className = "hint"; $("loadMsg").textContent = "";
    try {
      const d = await api("/api/scoring", { paths: r.filter(p => !scorers.some(s => s.path === p)) });
      const bad = d.filter(x => x.error); if (bad.length) { $("loadMsg").className = "err"; $("loadMsg").textContent = bad.map(x => x.error).join(" "); }
      scorers = scorers.concat(d.filter(x => !x.error));
    } catch (e) { $("loadMsg").className = "err"; $("loadMsg").textContent = e.message; }
    renderScorers(); updateBehaviorLists(); updateReady();
  };
  const stemOf = n => String(n || "").replace(/\.[^.]+$/, "").toLowerCase();
  function checkScorer(s) {   // does this scoring file belong to the loaded video?
    if (!video) return null; const bad = [];
    if (s.media && stemOf(s.media) !== stemOf(video.name)) bad.push(`scored a video called "${s.media}" (fine if it is the same video renamed or converted)`);
    const dur = video.fps ? video.frames / video.fps : 0;
    if (s.mdur && dur && Math.abs(s.mdur - dur) > 1) bad.push(`video length ${s.mdur.toFixed(1)} s vs ${dur.toFixed(1)} s`);
    if (s.fps && video.fps && Math.abs(s.fps - video.fps) > 0.05) bad.push(`frame rate ${s.fps} vs ${video.fps.toFixed(3)}`);
    if (!s.media && !s.mdur) return { ok: null, text: "can't tell which video this file is for" };
    return bad.length ? { ok: false, text: bad.join("; ") } : { ok: true, text: "matches the video" };
  }
  function renderScorers() {
    $("scorers").innerHTML = scorers.map((s, i) => { const c = checkScorer(s);
      const st = c ? `<div class="hint ${c.ok === false ? "err" : c.ok ? "ok" : ""}" style="margin:0 0 4px 2px">${c.ok === false ? "⚠ " : c.ok ? "✓ " : ""}${esc(c.text)}</div>` : "";
      return `<div class="scorer"><input type="text" value="${esc(s.name)}" data-i="${i}" title="Scorer name"><span class="fn" title="${esc(s.path)}">${esc(s.file)} · ${s.events} events</span><button data-x="${i}" title="Remove">✕</button></div>${st}`; }).join("");
    $("scorers").querySelectorAll("input").forEach(el => el.oninput = () => { scorers[+el.dataset.i].name = el.value.trim() || ("scorer" + (+el.dataset.i + 1)); updateSentence(); });
    $("scorers").querySelectorAll("button[data-x]").forEach(b => b.onclick = () => { scorers.splice(+b.dataset.x, 1); renderScorers(); updateBehaviorLists(); updateReady(); });
  }
  function updateBehaviorLists() {
    const b = [...new Set(scorers.flatMap(s => s.behaviors))].sort(), opts = b.map(x => `<option value="${esc(x)}">${esc(pretty(x))}</option>`).join("");
    const keep = [$("selBeh").value, $("selA").value, $("selB").value];
    $("selBeh").innerHTML = opts; $("selA").innerHTML = opts; $("selB").innerHTML = opts;
    if (b.includes(keep[0])) $("selBeh").value = keep[0]; if (b.includes(keep[1])) $("selA").value = keep[1];
    if (b.length > 1) $("selB").value = b.includes(keep[2]) ? keep[2] : (b.includes("rearing") ? "rearing" : b[1]);
    updateSentence();
  }
  function updateReady() { const ok = video && scorers.length; $("find").disabled = !ok; $("findMsg").className = "hint"; $("findMsg").textContent = ok ? "" : "Choose a video and at least one scoring file first."; }

  // ------------------------------------------------------------ step 2
  document.querySelectorAll("#ruleTabs button").forEach(b => b.onclick = () => {
    rule = b.dataset.r; document.querySelectorAll("#ruleTabs button").forEach(x => x.classList.toggle("on", x === b));
    $("rBehavior").style.display = rule === "behavior" ? "" : "none"; $("rOverlap").style.display = rule === "overlap" ? "" : "none"; $("rMulti").style.display = rule === "multi" ? "" : "none"; updateSentence(); });
  ["selBeh", "selA", "selB", "selScope", "minFrames", "pad", "gap"].forEach(id => $(id).addEventListener("input", updateSentence));
  function updateSentence() {
    const who = scorers.map(s => s.name).join(", ") || "the scorers"; let t;
    if (rule === "behavior") t = `Every time ${esc(who)} marked <b>${esc(pretty($("selBeh").value || "…"))}</b>. Moments marked by several scorers become one.`;
    else if (rule === "overlap") t = $("selScope").value === "same" ? `Frames where one scorer marked both <b>${esc(pretty($("selA").value || "…"))}</b> and <b>${esc(pretty($("selB").value || "…"))}</b> at the same time.`
      : `Frames where one scorer marked <b>${esc(pretty($("selA").value || "…"))}</b> and a different scorer marked <b>${esc(pretty($("selB").value || "…"))}</b>.`;
    else t = "Frames where any scorer has two or more behaviors at the same time.";
    if (rule !== "behavior") t += ` Overlaps shorter than ${$("minFrames").value || 1} frames are ignored.`;
    $("ruleSentence").innerHTML = t + " Everyone's labels for those frames are shown, whether or not they match the rule.";
  }
  $("find").onclick = async () => {
    $("findMsg").className = "hint"; $("findMsg").textContent = "Working…";
    try {
      const names = scorers.map(s => s.name); if (new Set(names).size !== names.length) throw new Error("Two scorers have the same name. Rename one.");
      const body = { video: video.path, files: scorers.map(s => ({ path: s.path, name: s.name })), rule, beh: $("selBeh").value, a: $("selA").value, b: $("selB").value, scope: $("selScope").value,
        min: +$("minFrames").value || 1, pad: +$("pad").value || 0, gap: +$("gap").value || 0, offset: parseInt($("offset").value) || 0 };
      const r = await api("/api/find", body); clips = r.clips; mode = "video"; lastRule = body; $("outName").value = folderName(r.rule_name); updateOutFull(); $("saveMsg").textContent = ""; $("saveWarn").textContent = "";
      const w = r.warnings.concat(scorers.filter(s => (checkScorer(s) || {}).ok === false).map(s => `Check ${s.name}: it may be for a different video.`));
      let revs = []; try { revs = (await api("/api/reviews", { folder: outPath() })).files; } catch (e) {}
      startResults(`${clips.length} moments · ${r.rule_name.replace(/_/g, " ")}`, revs);
      $("findMsg").className = w.length ? "err" : "hint"; $("findMsg").textContent = w.join(" ");
      if (w.length) { $("setup").open = true; $("sumline").textContent += " · ⚠ " + w[0]; }
    } catch (e) { $("findMsg").className = "err"; $("findMsg").textContent = e.message; }
  };
  $("pickFolder").onclick = async () => {
    const r = await chooser({ title: "Choose the review folder (the one with clips.csv)", dirMode: true }); if (!r) return;
    try { const d = await api("/api/folder", { path: r }); clips = d.clips; mode = "folder"; folderDir = d.folder; lastRule = null; review = {}; removed = new Set(); startResults(`${clips.length} clips from ${basename(d.folder)}`, d.reviews); note("", ""); }
    catch (e) { note(e.message, "err"); }
  };
  $("browseOut").onclick = async () => {
    const cur = $("outDir").value.trim();
    const r = await chooser({ title: "Choose the folder to save into", dirMode: true, start: cur || (video ? dirname(video.path) : undefined) }); if (!r) return;
    $("outDir").value = r; updateOutFull();
  };
  const folderName = n => String(n).replace(/[^\w\-]+/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "");
  const outPath = () => { const a = $("outDir").value.trim(), b = $("outName").value.trim(); return a && b ? joinP(a, b) : ""; };
  function updateOutFull() { const p = outPath(); $("outFull").textContent = p ? "Will create: " + p : "The folder name is filled in after you click Find moments."; }
  $("outDir").oninput = $("outName").oninput = updateOutFull;
  let noteTimer = 0, dirty = false;
  function note(t, cls) { const n = $("saveNote"); n.textContent = t; n.className = "hint " + (cls || ""); clearTimeout(noteTimer); if (t && cls !== "err") noteTimer = setTimeout(() => { n.textContent = ""; }, 9000); }

  // ------------------------------------------------------------ player (frames come from the server, any video format)
  const P = { path: null, info: null, buf: new Map(), cur: 0, playing: false, speed: 1, lo: 0, hi: 0, t0: 0, f0: 0, pending: new Set(), draw: 0, base: 0 };
  const cv = $("cv"), ctx = cv.getContext("2d");
  function clearBuffers() { P.buf.clear(); P.pending.clear(); P.path = null; }
  async function fetchFrames(start, count) {
    const path = P.path, key = start + ":" + count; if (!path || P.pending.has(key)) return;
    start = Math.max(0, start); let need = false; for (let i = start; i < start + count; i++) if (!P.buf.has(i)) { need = true; break; } if (!need) return;
    P.pending.add(key);
    try {
      const r = await fetch(`/api/frames?path=${encodeURIComponent(path)}&start=${start}&count=${count}&maxw=960&q=82`, { headers: { "X-Token": TOKEN } });
      if (!r.ok) throw new Error("frames"); const ab = await r.arrayBuffer(); if (P.path !== path) return;
      const dv = new DataView(ab); let o = 0;
      while (o + 8 <= ab.byteLength) { const idx = dv.getUint32(o, true), len = dv.getUint32(o + 4, true); P.buf.set(idx, new Blob([ab.slice(o + 8, o + 8 + len)], { type: "image/jpeg" })); o += 8 + len; }
      if (P.buf.size > 700) { const far = [...P.buf.keys()].sort((a, b) => Math.abs(b - P.cur) - Math.abs(a - P.cur)); for (const k of far.slice(0, P.buf.size - 600)) P.buf.delete(k); }
    } catch (e) {} finally { P.pending.delete(key); }
  }
  async function drawFrame(n) {
    const b = P.buf.get(n); if (!b) return false; const tok = ++P.draw;
    const bm = await createImageBitmap(b); if (tok !== P.draw) { bm.close(); return true; }
    if (cv.width !== bm.width || cv.height !== bm.height) { cv.width = bm.width; cv.height = bm.height; }
    ctx.drawImage(bm, 0, 0); bm.close(); return true;
  }
  function seek(n) {
    n = Math.max(P.lo, Math.min(P.hi, n)); P.cur = n; P.f0 = n; P.t0 = performance.now();
    if (P.buf.has(n)) drawFrame(n); else fetchFrames(Math.max(P.lo, n - 8), 16).then(() => { if (P.cur === n) drawFrame(n); });
  }
  function setPlaying(p) { P.playing = p; P.f0 = P.cur; P.t0 = performance.now(); $("play").textContent = p ? "⏸" : "▶"; }
  function tick(now) {
    requestAnimationFrame(tick); if (!P.path || !P.info || cur < 0) return; const c = clips[cur], fps = P.info.fps || c.fps;
    if (P.playing) {
      let t = P.f0 + Math.floor((now - P.t0) / 1000 * fps * P.speed);
      if (t > P.hi) { t = P.lo; P.f0 = P.lo; P.t0 = now; }
      if (t !== P.cur) { if (P.buf.has(t)) { P.cur = t; drawFrame(t); } else { P.f0 = P.cur; P.t0 = now; fetchFrames(t, 30); } }
      let ahead = P.cur; while (ahead < P.hi && P.buf.has(ahead + 1) && ahead - P.cur < 90) ahead++;
      if (ahead < P.hi && ahead - P.cur < 45) fetchFrames(ahead + 1, Math.min(30, P.hi - ahead));
    }
    const abs = P.base + P.cur, n = c.ce - c.cs + 1;
    const ph = $("ph"); if (ph) ph.style.left = (Math.min(n, Math.max(0, abs - c.cs + 0.5)) / n * 100) + "%";
    const oh = $("ovHead"); if (oh && mode === "video" && video) oh.style.left = (abs / Math.max(1, +$("overview").dataset.t) * 100) + "%";
    $("frameTxt").textContent = `frame ${abs} · ${(abs / c.fps).toFixed(2)} s`; $("chip").classList.toggle("on", abs >= c.es && abs <= c.ee);
    if (document.activeElement !== $("scrub")) $("scrub").value = P.cur;
  }
  requestAnimationFrame(tick);
  $("scrub").oninput = () => { setPlaying(false); seek(+$("scrub").value); };
  $("play").onclick = () => setPlaying(!P.playing);
  function setSpeed(s) { P.speed = s; P.f0 = P.cur; P.t0 = performance.now(); document.querySelectorAll(".speed button").forEach(b => b.classList.toggle("on", +b.dataset.s === s)); }
  document.querySelectorAll(".speed button").forEach(b => b.onclick = () => setSpeed(+b.dataset.s));

  // ------------------------------------------------------------ results
  function startResults(summary, revFiles) {
    $("sumline").textContent = "· " + summary; $("setup").open = false;
    $("app").style.display = "grid"; $("export").style.display = ""; $("loadNote").textContent = ""; dirty = false;
    const sc = [...new Set(clips.flatMap(c => c.scorers))].sort();
    $("fScorer").innerHTML = '<option value="">All scorers</option>' + sc.map(s => `<option>${esc(s)}</option>`).join("");
    $("ovwrap").style.display = mode === "video" ? "" : "none"; $("shareBox").style.display = mode === "video" ? "" : "none";
    if (!clips.length) { $("items").innerHTML = '<div class="item hint">Nothing matched this rule. Try a shorter overlap or a different behavior.</div>'; $("main").style.visibility = "hidden"; $("counts").textContent = "0 moments"; return; }
    $("main").style.visibility = "visible"; cur = -1; applyReviews(revFiles); renderList(); select(0);
  }
  const keyOf = c => `${c.rule}|${c.es}-${c.ee}`;
  const statusOf = c => (review[keyOf(c)] && (review[keyOf(c)].label || review[keyOf(c)].note)) ? "done" : "todo";
  function renderList() {
    const fs = $("fScorer").value, fst = $("fStatus").value;
    shown = clips.map((c, i) => i).filter(i => { const c = clips[i]; return (!fs || c.scorers.includes(fs)) && (fst === "gone" ? removed.has(keyOf(c)) : (!fst || (fst === "dis" ? c.disagree === "yes" : (statusOf(c) === fst && !removed.has(keyOf(c)))))); });
    $("items").innerHTML = shown.map(i => { const c = clips[i], r = review[keyOf(c)] || {};
      return `<div class="item ${i === cur ? "sel" : ""} ${removed.has(keyOf(c)) ? "gone" : ""}" data-i="${i}"><span class="dot ${r.label || (r.note ? "note" : "")}"></span><div><div class="t">${esc(c.id)} · frame ${c.es}</div><div class="s">${esc(c.behaviors.map(pretty).join(" + "))} · ${esc(c.scorers.join(", "))}</div></div></div>`; }).join("")
      || '<div class="item hint">No moments match.</div>';
    $("counts").textContent = `${clips.filter(c => statusOf(c) === "done" && !removed.has(keyOf(c))).length} of ${clips.length - removed.size} reviewed` + (removed.size ? ` · ${removed.size} removed` : "");
    $("items").querySelectorAll(".item[data-i]").forEach(el => el.onclick = () => select(+el.dataset.i));
  }
  $("fScorer").onchange = $("fStatus").onchange = renderList;

  async function select(i) {
    if (i < 0 || i >= clips.length) return; cur = i; const c = clips[i];
    const path = mode === "video" ? video.path : c.path; $("stageMsg").textContent = "";
    if (P.path !== path) { clearBuffers(); P.path = path; try { P.info = await api("/api/open?path=" + encodeURIComponent(path)); } catch (e) { P.path = null; $("stageMsg").textContent = "Could not open " + basename(path) + ": " + e.message; } }
    if (P.path) {
      P.base = mode === "video" ? 0 : c.cs; P.lo = mode === "video" ? c.cs : 0; P.hi = mode === "video" ? c.ce : c.ce - c.cs;
      $("scrub").min = P.lo; $("scrub").max = P.hi; seek(P.lo); setPlaying(true); fetchFrames(P.lo, 30); setSpeed(P.speed);
    }
    drawTimeline(c); drawOverview(c);
    $("dl").innerHTML = `
      <dt>Moment</dt><dd>${esc(c.id)} (${i + 1} of ${clips.length})</dd><dt>Rule</dt><dd>${esc(c.rule)}</dd>
      <dt>Event frames</dt><dd>${c.es}–${c.ee} · ${(c.es / c.fps).toFixed(2)} s – ${((c.ee + 1) / c.fps).toFixed(2)} s</dd>
      <dt>Found from</dt><dd>${esc(c.scorers.join(", "))}</dd>
      <dt>Scorers</dt><dd>${c.disagree === "yes" ? '<span class="badge dis">Disagree on these frames</span>' : c.disagree === "no" ? '<span class="badge agr">Agree on these frames</span>' : ""}</dd>
      ${c.merged > 1 ? `<dt>Joined</dt><dd>${c.merged} close events in one moment</dd>` : ""}`;
    const r = review[keyOf(c)] || {};
    $("labels").innerHTML = LABELS.map(([k, t, d], j) => `<button data-k="${k}" title="${esc(d)}" class="${k} ${r.label === k ? "on" : ""}">${j + 1} · ${t}</button>`).join("");
    $("labels").querySelectorAll("button").forEach(b => b.onclick = () => mark(b.dataset.k));
    $("rm").style.display = mode === "video" ? "" : "none"; $("rm").textContent = removed.has(keyOf(c)) ? "Put back" : "Remove from export"; $("note").value = r.note || ""; renderList(); const el = $("items").querySelector(".item.sel"); if (el) el.scrollIntoView({ block: "nearest" });
  }

  function drawTimeline(c) {
    const cs = c.cs, ce = c.ce, es = c.es, ee = c.ee, n = ce - cs + 1, tr = c.tracks;
    if (!tr) { $("tl").innerHTML = `<div class="hint" style="padding:6px">${esc(c.sb).replace(/; /g, "<br>")}</div>`; $("legend").innerHTML = ""; return; }
    const ruleB = new Set(c.behaviors), order = [...c.scorers, ...Object.keys(tr).filter(s => !c.scorers.includes(s)).sort()], used = new Set(); let html = "";
    for (const sc of order) {
      const iv = (tr[sc] || []).filter(x => showAll || ruleB.has(x[0])).map(x => [x[0], Math.max(x[1], cs), Math.min(x[2], ce), x[1], x[2]]).sort((a, b) => a[1] - b[1] || b[2] - a[2]);
      const lanes = [], lane = iv.map(x => { let k = lanes.findIndex(e => e < x[1]); if (k < 0) { k = lanes.length; lanes.push(0); } lanes[k] = x[2]; return k; });
      const h = Math.max(1, lanes.length) * 16 + 4, trig = c.scorers.includes(sc);
      const segs = iv.map((x, j) => { const col = colorOf(x[0]); used.add(x[0]);
        return `<div class="seg" style="left:${(x[1] - cs) / n * 100}%;width:${Math.max((x[2] - x[1] + 1) / n * 100, 0.4)}%;top:${2 + lane[j] * 16}px;height:14px;background:${col};color:${ink(col)}" title="${esc(pretty(x[0]))}: frames ${x[3]}–${x[4]}">${esc(pretty(x[0]))}</div>`; }).join("");
      html += `<div class="tlrow"><div class="tlname ${trig ? "trig" : ""}" title="${esc(sc)}${trig ? " (found the event)" : ""}">${esc(sc)}</div><div class="tltrack" style="height:${h}px">${segs || `<span class="hint" style="padding:0 8px;line-height:${h}px">nothing marked</span>`}</div></div>`;
    }
    html += `<div class="tllayer"><div class="evband" style="left:${(es - cs) / n * 100}%;width:${Math.max((ee - es + 1) / n * 100, 0.4)}%"><span class="evtag">event ${es}–${ee}</span></div><div class="playhead" id="ph"></div></div><div class="tlaxis"><span>frame ${cs}</span><span>frame ${ce}</span></div>`;
    $("tl").innerHTML = html; $("legend").innerHTML = [...used].sort().map(b => `<span><i style="background:${colorOf(b)}"></i>${esc(pretty(b))}</span>`).join("");
  }
  function drawOverview(c) {
    if (mode !== "video") return; const T = Math.max(video.frames, ...clips.map(x => x.ce + 1)), W = x => x / T * 100 + "%";
    $("overview").innerHTML = clips.map(x => `<div class="tick ${x === c ? "cur" : ""}" style="left:${W(x.es)};width:${(x.ee - x.es + 1) / T * 100}%"></div>`).join("")
      + `<div class="win" style="left:${W(c.cs)};width:${(c.ce - c.cs + 1) / T * 100}%"></div><div class="head" id="ovHead"></div>`; $("overview").dataset.t = T;
  }
  $("overview").addEventListener("click", e => { const r = $("overview").getBoundingClientRect(), f = (e.clientX - r.left) / r.width * +$("overview").dataset.t;
    let best = 0, d = 1e18; clips.forEach((c, i) => { const m = Math.abs((c.es + c.ee) / 2 - f); if (m < d) { d = m; best = i; } }); select(best); });
  $("tl").addEventListener("click", e => { const layer = $("tl").querySelector(".tllayer"); if (!layer || cur < 0) return; const c = clips[cur];
    const r = layer.getBoundingClientRect(), x = (e.clientX - r.left) / r.width; if (x < 0 || x > 1) return; setPlaying(false); seek(P.lo + Math.floor(x * (c.ce - c.cs + 1))); });
  $("tlAll").onclick = () => { showAll = true; $("tlAll").classList.add("on"); $("tlRule").classList.remove("on"); if (cur >= 0) drawTimeline(clips[cur]); };
  $("tlRule").onclick = () => { showAll = false; $("tlRule").classList.add("on"); $("tlAll").classList.remove("on"); if (cur >= 0) drawTimeline(clips[cur]); };
  function toggleRemove() { if (mode !== "video" || cur < 0) return; const k = keyOf(clips[cur]); const gone = !removed.has(k); gone ? removed.add(k) : removed.delete(k); renderList(); if (gone && cur + 1 < clips.length) select(cur + 1); else select(cur); }
  $("rm").onclick = toggleRemove;
  $("prev").onclick = () => select(cur - 1); $("next").onclick = () => select(cur + 1);

  document.addEventListener("keydown", e => {
    if (!clips.length || boxState || /^(TEXTAREA|INPUT|SELECT)$/.test(e.target.tagName) && e.target.type !== "range" || e.metaKey || e.ctrlKey) return;
    const k = e.key; if (cur < 0) return;
    if (k === "j" || k === "J") select(cur + 1); else if (k === "k" || k === "K") select(cur - 1);
    else if (k === "x" || k === "X") toggleRemove();
    else if (k === " ") { e.preventDefault(); setPlaying(!P.playing); }
    else if (k >= "1" && k <= "4") mark(LABELS[+k - 1][0]);
    else if (k === "[" || k === "]") { const sp = [1, 0.5, 0.25], i = sp.indexOf(P.speed); setSpeed(sp[Math.max(0, Math.min(2, i + (k === "]" ? 1 : -1)))]); }
    else if (k === "," || k === "." || k === "ArrowLeft" || k === "ArrowRight") { e.preventDefault(); setPlaying(false); seek(P.cur + (k === "." || k === "ArrowRight" ? 1 : -1)); }
  });

  // ------------------------------------------------------------ review marks
  const ensure = () => review[keyOf(clips[cur])] = review[keyOf(clips[cur])] || {};
  function mark(k) { const r = ensure(); r.label = r.label === k ? "" : k; r.date = new Date().toISOString(); dirty = true; renderList(); select(cur); }
  $("note").oninput = () => { const r = ensure(); r.note = $("note").value; r.date = new Date().toISOString(); dirty = true; };
  $("note").onchange = renderList;
  const csvCell = v => { v = String(v ?? ""); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  function parseCSV(text) { const rows = []; let row = [], f = "", q = false; text = text.replace(/^﻿/, "");
    for (let i = 0; i < text.length; i++) { const c = text[i];
      if (q) { if (c === '"') { if (text[i+1] === '"') { f += '"'; i++; } else q = false; } else f += c; } else if (c === '"') q = true; else if (c === ",") { row.push(f); f = ""; }
      else if (c === "\n") { row.push(f); rows.push(row); row = []; f = ""; } else if (c !== "\r") f += c; }
    if (f.length || row.length) { row.push(f); rows.push(row); } const head = rows.shift() || []; return rows.filter(r => r.length > 1).map(r => Object.fromEntries(head.map((h, i) => [h, r[i] ?? ""]))); }
  const safe = s => String(s).replace(/[^\w\-]+/g, "_");
  $("export").onclick = async () => {
    const who = $("reviewer").value.trim() || "reviewer", name = Object.fromEntries(LABELS.map(([k, t]) => [k, t]));
    const head = ["clip_id", "rule_name", "event_start_frame", "event_end_frame", "scorers", "review_label", "review_notes", "reviewer", "review_date"];
    const lines = [head.join(",")].concat(clips.filter(c => !removed.has(keyOf(c))).map(c => { const r = review[keyOf(c)] || {}; return [c.id, c.rule, c.es, c.ee, c.scorers.join(";"), name[r.label] || "", r.note || "", who, r.date || ""].map(csvCell).join(","); }));
    const target = mode === "folder" ? folderDir : outPath();
    if (!target) { note("Choose a folder under 'Create a review folder' first.", "err"); return; }
    try { const r = await api("/api/savereview", { folder: target, file: "review_" + safe(who) + ".csv", text: "﻿" + lines.join("\r\n") + "\r\n" }); dirty = false; note(`Saved ${r.file} in ${r.folder}`, "ok"); }
    catch (e) { note(e.message, "err"); }
  };
  function applyReviews(files) {
    if (!files || !files.length) return; const who = $("reviewer").value.trim(); let f = files[files.length - 1];
    if (who) { const m = files.find(x => x.file === "review_" + safe(who) + ".csv"); if (m) f = m; }
    const back = Object.assign(Object.fromEntries(LABELS.map(([k, t]) => [t, k])), OLD_LABELS); let n = 0;
    for (const r of parseCSV(f.text)) { const c = clips.find(x => (r.rule_name && x.rule === r.rule_name && x.es === +r.event_start_frame && x.ee === +r.event_end_frame) || (!r.rule_name && x.id === r.clip_id));
      if (c) { review[keyOf(c)] = { label: back[r.review_label] || "", note: r.review_notes || "", date: r.review_date || "" }; n++; } if (r.reviewer && !$("reviewer").value) $("reviewer").value = r.reviewer; }
    $("loadNote").textContent = `Loaded your earlier review (${f.file}, ${n} moments).` + (files.length > 1 ? ` ${files.length - 1} other review file(s) in this folder were not loaded. To load another, type that reviewer's name and open the folder again.` : "");
  }

  // ------------------------------------------------------------ save a clips folder to share
  $("saveClips").onclick = async () => {
    const keep = clips.filter(c => !removed.has(keyOf(c))); if (mode !== "video" || !keep.length) { $("saveMsg").className = "err"; $("saveMsg").textContent = "No clips left to save."; return; } $("saveMsg").className = "hint"; $("saveMsg").textContent = "Saving…"; $("saveWarn").textContent = "";
    try {
      const j = await api("/api/export", { video: video.path, out: outPath(), clips: keep, settings: { rule: lastRule.rule, params: lastRule, scorers: Object.fromEntries(scorers.map(s => [s.name, s.file])) } });
      for (;;) { await new Promise(r => setTimeout(r, 500)); const s = await api("/api/job?id=" + j.id);
        if (s.error) throw new Error(s.error); $("saveMsg").textContent = `Saving… ${s.done} of ${s.total}`;
        if (s.finished) { $("saveMsg").className = "ok"; $("saveMsg").textContent = `Review folder created: ${s.total} clips in ${s.out}`; $("saveWarn").textContent = (s.warnings || []).join(" "); break; } }
    } catch (e) { $("saveMsg").className = "err"; $("saveMsg").textContent = e.message; }
  };
  window.addEventListener("beforeunload", e => { if (dirty) { e.preventDefault(); e.returnValue = ""; } });
  updateSentence(); updateReady();
  window.__t = { P, get clips() { return clips; } };
})();
</script>
</body>
</html>
'''

if __name__ == "__main__":
    main()
