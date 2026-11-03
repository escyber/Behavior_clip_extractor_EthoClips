# EthoClips

One Python file for finding moments in scored videos and reviewing them. It works from your BORIS TSV files and the video (AVI or MP4, no converting). The screen is a page in your browser, but everything runs on your own computer and nothing is uploaded.

## What it does
- You choose the rule yourself: one behavior (from any scorer), two behaviors at the same time, or any frames with two or more behaviors at once.
- "Two behaviors at the same time" can mean one scorer marked both (for example rearing + resting), or two different scorers each marked one.
- Moments that several scorers marked are joined into one, and every scorer who found it is listed.
- A timeline shows what every scorer marked on those same frames, and flags when scorers disagree.
- It plays the video frame-exact, straight from the AVI.
- You mark each moment, add notes, remove moments you don't want, and save a folder of short clips for someone else to review.

## Install (once)
Needs Python 3 and two libraries:
```
pip install opencv-python numpy
```
Optional: ffmpeg (gives smaller clips with a frame-number overlay). Without it, clips are made with OpenCV.

## Start
```
python3 ethoclips.py
```
A link starting with `http://127.0.0.1:` is printed and your browser opens. Keep the Terminal window open while you work. Press Ctrl+C there to stop.

## Use it
1. **Choose video…** and **Add scoring files (TSV)…**, one TSV per scorer. In both windows you can type part of a name (for example `m1917`) in the search box to find files without knowing the folder. Each scorer gets a name (taken from the file name; you can edit it) and a ✓ or ⚠ showing whether the TSV matches the video.
2. Under **What to find**, pick the rule. "Shortest overlap" is in frames (default 5). "Context before/after" adds seconds around each moment.
3. Click **Find moments**.
4. For each moment: watch it, look at the scorers' timeline, then mark it and add a note if needed.
5. **Save review**.

### Marks
- **1 Valid event:** the behaviors really happen together in the video.
- **2 Contradictory labels:** the labels can't both be true (for example rearing and locomotion at once).
- **3 Mislabeled:** what a scorer marked does not match what the video shows.
- **4 Unsure.**

### Keys
J next · K previous · Space play/pause · ← → one frame back/forward · 1–4 mark · X remove from export · [ ] speed. Click the timeline to jump.

### What the red EVENT label means
It lights up while the video is inside the moment the rule found. Before and after it you see the extra seconds of context (set under **Context before/after**).

### Removing moments
Click **Remove from export** (or press X) to leave a moment out of the review folder. It stays in the list, dimmed, and can be put back. Removed moments are not saved in the review folder, `clips.csv` or the review CSV.

## Share a review folder (Box)
1. Open "Create a review folder to share", choose the folder to save into with **Browse…** (search works there too), and click **Create review folder**. A new folder is created inside it, named after the rule you used (for example `rearing_resting_same_scorer`; you can change the name). Run several analyses into the same folder and each gets its own subfolder. If a folder with that name already exists, nothing is overwritten. The new folder contains:
   - `clips/` with one MP4 per moment, named like `CLIP001_<video name>_frames_5327-5425.mp4` (the frames are the first and last frame of the clip)
   - `clips.csv` (one row per clip: rule, event and clip frames, scorers, what each scorer marked, whether they disagree)
   - `labels.json` (the scorers' labels for the timeline)
2. Upload the whole folder to Box.
3. The reviewer downloads it, runs this same script, clicks **Open review folder…**, marks and notes each clip, and clicks **Save review**. The review is saved inside that same folder as `review_<name>.csv`. They send the file back (or the whole folder).
4. The reviewer does not need the video or the TSV files.

Opening a review folder also loads the review saved in it, so you can stop and continue later. If a folder has several review files, the latest is loaded, or the one that matches the Reviewer name you typed before opening it.

## Review CSV
Columns: clip_id, rule_name, event_start_frame, event_end_frame, scorers, review_label, review_notes, reviewer, review_date. Reviewer is written on every row.

## Sherlock (keep big videos off your Mac)
On Sherlock:
```
python3 ethoclips.py --no-browser --port 8765
```
On your Mac, in a new Terminal window:
```
ssh -L 8765:localhost:8765 stanflab@login.sherlock.stanford.edu
```
Then open the link Sherlock printed in your Mac's browser, using `127.0.0.1:8765`. The video stays on Sherlock and only small frames travel to your screen. Run long jobs on a compute node, not the login node.

## Without the page (batch)
```
python3 ethoclips.py cut overlap --video v.avi --tsv a.tsv b.tsv --names Elena Johan --a rearing --b resting --out clips_out
```
Rules: `behavior --behavior face_swipe`, `overlap --a X --b Y`, `multi`. Options: `--scope same|different`, `--min-frames`, `--pad`, `--merge-gap`, `--frame-offset`.

## Good to know
- Behavior names are matched in a simple way: lowercase, spaces and hyphens become underscores, "locomotor" becomes "locomotive".
- In the TSV, STOP frames count as part of the behavior.
- Frame numbers are exact: frames are read by index, tested on AVI and MP4.
- The link has a private code and only works on the computer running the tool.
- "Frame offset" (under More options) shifts all scoring frame numbers, in case a TSV is off by a fixed number of frames.
