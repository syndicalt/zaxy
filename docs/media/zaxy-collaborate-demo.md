# Zaxy Coordinate/Collaborate Demo Media

This directory contains the scripted launch demo for Zaxy 1.0.0.

Generated assets:

- [zaxy-collaborate-demo.mp4](zaxy-collaborate-demo.mp4)
- [zaxy-collaborate-demo.gif](zaxy-collaborate-demo.gif)

The demo is a product-level walkthrough, not a live account recording. It shows
the flow Zaxy Coordinate is built for:

1. initialize local memory;
2. start a parent mission;
3. collect cited worker findings from isolated sessions;
4. review conflicts and promote accepted state;
5. assemble Memory Checkout for the next agent turn.

Regenerate the assets with:

```bash
python scripts/generate-release-media.py
```

The script writes deterministic frames under `docs/media/zaxy-collaborate-demo-frames/`
and encodes the MP4/GIF with `ffmpeg`.
