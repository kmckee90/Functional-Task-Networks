# Monitor navigation

The current monitor on localhost:8016 serves a chronological navigation chain:
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → reduced actions →
corrected-input continuations → corrected engine → continuing lives / discounts.

`experiment_navigation.py` owns the order and shared-storage locations.
The HTTP handler injects Previous/Next links into each report at request time;
archived report HTML and training data are not rewritten. Phase 0 has no previous
link; the active experiment has no next link. Existing `/previous/` links work.
Canonical archived URLs are `/experiments/<key>/`; the active page remains `/`.

Phase 0 and Phase 1 monitor assets were copied from the stopped globo-dreamer
volume's `globo_dreamer_state/reviews/config24/{phase0,continuous}` to shared
`globo_archives/phase{0,1}/monitor`. Phase 1 is the existing archived-exploration
report, preserved as found. No checkpoints or replay were copied.
The input-correction report is under shared `globo_input_v2/monitor`, linked to
its existing shared run records. Other reports use their original shared paths.
All archive assets stay on cluster storage. The temporary CPU archive reader
is not needed after the copy and is stopped.

Validation: every report and its Previous/Next destinations returned HTTP 200;
the first video on every report returned HTTP 200. Legacy previous/phase aliases
also render the navigation. Agents and publishers are unchanged.
