# SceneTrace gaze-centered relational-search experiment

A dependency-free, local browser experiment for one participant. Each session presents 18 randomized scenes and balances six unannounced relational question types across 1, 3, and 5 second question-onset intervals.

## Run

From this folder:

```bash
python3 -m http.server 8080
```

Then open <http://localhost:8080>. A modern desktop browser is recommended. The app makes no network requests and writes no data until the participant downloads JSON or CSV from the final results screen.

## Protocol

- 900 ms ready period
- Static scene appears with a central fixation mark
- After a balanced 1, 3, or 5 second viewing interval, the fixation mark is replaced by a compact question and answer grid
- The scene remains visible until the participant responds
- Response accuracy and time from question onset are recorded, with no correctness feedback during the run

The six question families are relative size, closest pair, furthest pair, highest position, leftmost position, and proximity to fixation. Scene generation is conditioned on the upcoming question and requires a minimum geometric or size margin between the correct answer and its nearest competitor. The question remains hidden until its scheduled onset, then appears near the center so the participant does not need to make a large eye movement.

## Data

The full JSON export contains the random seed, trial schedule, configuration, timestamps, every object, the enforced salience margin, questions and alternatives, response accuracy and latency, viewport details, and focus interruptions. The CSV provides one row per completed trial for analysis.

This is a research prototype, not a validated clinical or diagnostic instrument. For formal human-subjects research, obtain the required ethics review and add your approved consent/debrief language.
