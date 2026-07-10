"""Episode segmentation + insight/Q&A synthesis (design.md §5).

Submodules are imported explicitly by callers (segment is stdlib-only;
synthesize needs torch) so importing synth.segment off-pod doesn't drag in torch.
"""
