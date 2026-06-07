# AdRemix

Turn any competitor ad into a remixed version for your brand. Upload a video ad → auto-transcribe → rewrite the script → generate Seeddance prompts for each 15-second segment → export SRT captions.

## Quick Start

```bash
# 1. Clone / copy this folder, then:
cd ad-remix

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API keys
cp .env.example .env
# Edit .env and paste your OpenAI + Anthropic keys

# 4. Run
python -m uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## How It Works

1. **Upload** an ad video (mp4/mov/webm/mp3, under 25 MB)
2. **Transcribe** — uses OpenAI Whisper API (~$0.006/min)
3. **Rewrite** — Claude rewrites the script for your brand (~$0.01)
4. **Segment** — Claude breaks it into 15-sec chunks with Seeddance prompts and a style-lock for visual consistency (~$0.02)
5. **Export** — download prompts, SRT captions, or a full export file

## What You Still Do Manually

- Generate each segment in Seeddance/Higgsfield (paste the prompt from AdRemix)
- Stitch clips together in CapCut
- Add music (the SRT file handles captions)

## Costs

| Step | Service | Cost |
|------|---------|------|
| Transcribe | OpenAI Whisper | ~$0.006/min |
| Rewrite | Claude Sonnet | ~$0.01 |
| Segment | Claude Sonnet | ~$0.02 |
| **Total** | | **~$0.03-0.05/ad** |

## Tips

- Fill out the Brand Profile sidebar and click **Save** — it persists in your browser.
- You can edit the transcript and rewritten script before moving to the next step.
- Each segment prompt is editable — tweak before copying to Seeddance.
- The **Style Lock** section shows the visual constants to keep your segments consistent.
- Use "viral captions" SRT for the 2-3-word-at-a-time style popular on TikTok/Reels.

## Deployment

For a VPS (e.g. a $5/mo DigitalOcean droplet):

```bash
pip install -r requirements.txt
# Set .env with your keys
uvicorn app:app --host 0.0.0.0 --port 8000
```

Use a reverse proxy (nginx/caddy) + HTTPS if exposing to the internet.
