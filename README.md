# HeatherBot

**A fully local AI companion chatbot ecosystem for Telegram — with multi-platform funnel automation.**

HeatherBot is a Telegram bot (Bot API via python-telegram-bot) that runs entirely on your own hardware. No cloud APIs, no OpenAI, no subscriptions. Just a local LLM, a persona YAML file, and a stack of services that make the bot feel like a real person texting.

The ecosystem includes the core Telegram companion bot, a multi-platform outreach dashboard (Reddit + FetLife), and an autonomous management agent — all running locally.

## Architecture

```
  +-------------------+     +-------------------+     +-------------------+
  |   llama-server    |     |      Ollama       |     |     ComfyUI       |
  |  (Text AI, 12B)   |     |  (Image Analysis) |     | (Image Generation)|
  |    port 1234      |     |    port 11434     |     |    port 8188      |
  +--------+----------+     +--------+----------+     +--------+----------+
           |                         |                         |
           +------------+------------+------------+------------+
                        |                         |
               +--------+----------+     +--------+----------+
               |  HeatherBot Core  |     |   Chatterbox TTS  |
               |   (Bot API)       |     |    port 5001      |
               |  + Flask monitor  |     +-------------------+
               |    port 8888      |
               +--------+----------+
                        |
           +-----+------+------+------------+
           |            |                    |
  +--------+-------+  +-+-------------+  +--+--------------+
  | Frank Dashboard |  | Discord Bot   |  | OpenClaw MGMT   |
  | (Reddit/FetLife)|  | (discord.py)  |  | (Autonomous Agt)|
  |   port 8080     |  +---------------+  +-----------------+
  +--------+-------+
           |
  +--------+-------+
  | Twitter/X Bot   |
  | (Tweepy API)    |
  +----------------+
```

| Service | Port | Purpose |
|---------|------|---------|
| llama-server | 1234 | Text generation (llama.cpp with any GGUF model) |
| Ollama | 11434 | Image analysis (LLaVA or similar vision model) |
| ComfyUI | 8188 | Image generation (FLUX.1 dev) with face-swap workflows |
| Chatterbox TTS | 5001 | Voice synthesis (voice cloning) |
| Bot Monitor | 8888 | Web dashboard for analytics and admin |
| Frank Dashboard | 8080 | Multi-platform outreach (Reddit + FetLife auto-funnel) |
| Discord Bot | — | Community server with scheduled posting and auto-welcome |
| Twitter/X | — | Automated tweeting via Tweepy |

## Features

### Core Bot (Telegram)
- **Persona YAML system** — Define your character's identity, backstory, personality, physical description, and communication style in a single YAML file. Swap personas by pointing to a different file.
- **Bot API** — Standard Telegram bot using a BotFather token. Receives messages via long-polling. No phone number or session file required.
- **Conversation persistence** — Chat history is saved to `/app/data/conversations/<chat_id>.json` and reloaded on restart. History survives container rebuilds.
- **Gender detection** — Detects user gender from early messages and injects pronoun context into the LLM system prompt for accurate responses.
- **Image generation** — FLUX.1 dev via ComfyUI with:
  - Context-aware prompt expansion (LLM reads the last 10 chat messages to infer setting and pose)
  - Persona profile injected into prompt expansion (occupation, locations, typical outfits)
  - Runtime LoRA injection (NSFW Master, anatomy detail) with ComfyUI availability check — missing LoRAs are logged and skipped gracefully
  - Face image auto-upload to ComfyUI's input directory on first generation
  - Two-pass NSFW detection (unambiguous terms + anatomy/qualifier regex) with flag locked from original request before context expansion
  - Negative prompt preserved from workflow JSON (not overwritten by the bot)
  - OOM retry — detects CUDA out-of-memory errors, waits 5 seconds, retries once
  - Photo cap configurable via `PHOTO_CAP` env var; reset on `/reset`
- **Image analysis** — Receives and analyzes user photos via Ollama vision models.
- **Voice messages** — Chatterbox TTS voice cloning for sending voice notes.
- **Story system** — Pre-written story bank with per-user rotation.
- **Monitoring dashboard** — Real-time Flask dashboard at port 8888. Access controlled by IP allowlist (`MONITOR_ALLOWED_IPS`) with CIDR subnet support.
- **Breadcrumb logging** — Detailed pipeline tracing (incoming messages, AI replies, ComfyUI prompts) gated behind `BREADCRUMB_LOGGING=1`.
- **Content safety** — CSAM flag-and-review system, blocked user management, admin alerts.
- **AI disclosure** — Automatic first-message disclosure that includes the persona name. Reality-check responses own the AI status without breaking character.
- **Re-engagement** — Automatic outreach to inactive users. Configurable timing.
- **Adaptive kink personas** — 17 kink-specific personality overlays that detect and shift based on user behavior.
- **Arousal tracking** — Detects climax/heated/afterglow states and adjusts token budget and temperature accordingly.
- **Post-processing pipeline** — Strips thinking tags, asterisk actions, AI denial claims, and unprompted self-identification.

### Image Generation Pipeline

```
User request
  → _clean_photo_request()         strip conversational framing
  → NSFW flag locked in            from original request (before expansion)
  → build_image_prompt_from_context()  LLM expands with chat history + persona profile
  → build_heather_prompt()         prepend character description from persona YAML
  → ComfyUI /prompt                workflow submitted with face image, LoRAs, seeds
```

The character description prefix (hair, eyes, body type, breast description) is read from the `physical:` section of the persona YAML. Explicit SFW/NSFW prefix overrides and quality suffixes can be set in `physical.image_prompt`.

### Persona YAML — `physical.image_prompt` section

```yaml
physical:
  hair: "Dark brown, wavy, shoulder-length"
  eyes: "Brown"
  body_type: "Athletic, curvy"
  chest_description: "medium natural breasts"       # SFW image prompt
  breast_description: "medium natural breasts, small brown nipples"  # NSFW image prompt

  image_prompt:
    sfw_prefix: ""          # Full override (blank = auto-build from fields above)
    nsfw_prefix: ""
    quality_suffix: ", natural lighting, authentic amateur photo..."
    nsfw_quality_suffix: ", natural lighting, realistic skin with pores..."
    setting_context: ""     # Optional hint for setting inference (e.g. "at a bar or spa")
```

### Monitoring Dashboard

Access is controlled by IP allowlist, not a shared token. Set `MONITOR_ALLOWED_IPS` in `.env`:

```env
MONITOR_ALLOWED_IPS=192.168.1.0/24,10.0.0.1
```

Supports individual IPs and CIDR ranges. Defaults to `127.0.0.1,::1` (localhost only).

## Hardware Requirements

**Minimum:**
- 1× GPU with 24GB VRAM (RTX 3090, RTX 4090, etc.)
- 32GB system RAM
- Python 3.10+

**Recommended:**
- 2× GPUs (one for text, one for image generation)
- 64GB+ system RAM
- SSD for model storage

## Prerequisites

1. **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — Download a GGUF model and `llama-server`
2. **[Ollama](https://ollama.ai)** — Pull a vision model: `ollama pull llava`
3. **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — For image generation (optional)
4. **[Chatterbox TTS](https://github.com/resemble-ai/chatterbox)** — For voice messages (optional)
5. **Python 3.10+**
6. **Telegram bot token** — Create a bot via [@BotFather](https://t.me/BotFather)

## Docker Deployment

The recommended way to run HeatherBot is via Docker Compose. The image clones the latest code from GitHub on every build, so there is no stale code in the container.

```bash
# Build and start
docker compose up -d heatherbot

# Force rebuild to pull latest code from GitHub
docker compose build --no-cache heatherbot && docker compose up -d heatherbot

# View logs
docker compose logs -f heatherbot
```

### Volumes

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./heatherbot/data` | `/app/data` | Conversation history, user state |
| `./heatherbot/config` | `/app/config` | Persona YAML, ComfyUI workflow JSON, face image |

Place your persona YAML, `workflow_flux.json`, and `heather_face.png` in `./heatherbot/config/` before starting.

### docker-compose.yml

```yaml
services:
  heatherbot:
    build:
      context: ./heatherbot
      dockerfile: Dockerfile
    container_name: heatherbot
    restart: unless-stopped
    networks:
      - ollama-net
    env_file:
      - ./heatherbot/.env
    ports:
      - 8888:8888
    volumes:
      - ./heatherbot/data:/app/data
      - ./heatherbot/config:/app/config
    depends_on:
      - ollama

networks:
  ollama-net:
    external: true
```

## Manual Installation

```bash
git clone https://github.com/gdave44/heatherbot.git
cd heatherbot
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
```

## Configuration

### Environment Variables

See `.env.example` for all options. Key variables:

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_botfather_token
ADMIN_USER_ID=your_telegram_numeric_id

# Persona
HEATHER_PERSONA=my_persona.yaml

# LLM / AI services
MODEL=dolphin-llama3:8b
LLM_URL=http://ollama:11434
TTS_URL=http://kokoro:8000
COMFYUI_URL=http://comfyui:8188

# ComfyUI assets (place files in config volume)
COMFYUI_WORKFLOW_FILE=/app/config/workflow_flux.json
COMFYUI_FACE_IMAGE=/app/config/heather_face.png

# Photo cap
PHOTO_CAP=5                   # Max photos per rolling window (default: 5)
PHOTO_CAP_WINDOW_HOURS=2      # Rolling window in hours (default: 2)

# Monitoring dashboard
MONITOR_ALLOWED_IPS=127.0.0.1,::1   # IPs or CIDR ranges allowed access

# Debugging
BREADCRUMB_LOGGING=0          # Set to 1 for full pipeline trace logs
```

### Persona YAML

The bot ships with `persona_example.yaml` — a complete template. To create your own:

1. Copy `persona_example.yaml` to `config/my_persona.yaml`
2. Edit all sections
3. Set `HEATHER_PERSONA=my_persona.yaml` in `.env`

Key sections:
- **identity** — Name, age, location, occupation
- **physical** — Hair, eyes, body type, `image_prompt` overrides for ComfyUI
- **personality** — Traits, humor
- **communication** — Voice, phrases, flirting patterns
- **ai_behavior** — Guardrails, never-say list, mode behaviors
- **prompts** — The system prompts sent to the LLM (`base_personality` is the most important)

### ComfyUI Setup

1. Place your FLUX workflow JSON at `config/workflow_flux.json`
2. Place a face source image at `config/heather_face.png`
3. The bot uploads the face image to ComfyUI's input directory automatically on first generation
4. Edit negative prompts directly in the workflow JSON — the bot does not overwrite node 4
5. LoRAs (`NSFW_master.safetensors`, `flux-female-anatomy.safetensors`) are optional — the bot checks ComfyUI's available LoRA list at runtime and skips any that are not installed

## Running the Bot

```bash
# Basic
python heather_telegram_bot.py

# With monitoring dashboard
python heather_telegram_bot.py --monitoring

# Custom persona + monitoring
python heather_telegram_bot.py --personality my_persona.yaml --monitoring
```

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--personality` | `persona_example.yaml` | Persona YAML filename (looked up in config dir) |
| `--monitoring` | off | Enable web dashboard on port 8888 |
| `--small-model` | off | Condensed prompts for smaller models |
| `--text-port` | 1234 | llama-server port (overridden by `LLM_URL`) |
| `--log-dir` | `logs/` | Log directory |
| `--unfiltered` | off | Disable content filters |

### User Commands (in Telegram)

| Command | Description |
|---------|-------------|
| `/start` | Begin conversation |
| `/help` | Show available commands |
| `/menu` | Show command menu |
| `/selfie` | Request a photo (optionally with description) |
| `/reset` | Clear conversation history and photo cap |
| `/voice_on` / `/voice_off` | Toggle voice message mode |

### Admin Commands (send from your `ADMIN_USER_ID` account)

| Command | Description |
|---------|-------------|
| `/admin_stats` | User statistics and engagement metrics |
| `/admin_flags` | Review CSAM flags |
| `/admin_block <user_id>` | Block a user |
| `/admin_unblock <user_id>` | Unblock a user |
| `/admin_reset <user_id>` | Reset a specific user's conversation |
| `/takeover <user_id>` | Pause bot and reply manually |
| `/botreturn <user_id>` | Resume bot after takeover |
| `/say <user_id> <message>` | Send a message as the character |
| `/admin_reengage_scan` | Show re-engagement candidates |
| `/admin_reengage_send <user_id>` | Send a re-engagement message |
| `/admin_warmth <user_id>` | Show warmth tier for a user |
| `/admin_help` | Full list of admin commands |

## White-Labeling

HeatherBot is character-agnostic. All personality comes from the YAML files and media assets.

1. **Create a persona YAML** — Copy `persona_example.yaml`, fill in your character
2. **Place assets** — Face source image and workflow JSON in `config/`
3. **Configure LLM** — Set `MODEL` to whichever model you're running in Ollama/llama-server
4. **Create a bot** — Register with [@BotFather](https://t.me/BotFather), set `TELEGRAM_BOT_TOKEN`
5. **Optionally clone a voice** — Chatterbox TTS voice cloning

## Migrating from v3.x (Telethon/MTProto)

v4.0 replaces the Telethon MTProto userbot with the standard Telegram Bot API:

| v3.x (Telethon) | v4.0 (Bot API) |
|-----------------|----------------|
| `TELEGRAM_API_ID` + `TELEGRAM_API_HASH` | `TELEGRAM_BOT_TOKEN` (from BotFather) |
| Appears as a real Telegram user | Appears as a bot account |
| Session file (`heather_session.session`) | No session file — token-based auth |
| `iter_dialogs` for re-engagement | In-memory `conversation_activity` dict |
| File references expire | Bot API `file_id`s are permanent |
| `telethon>=1.36` | `python-telegram-bot>=21.0` |

Update your `.env`: remove `TELEGRAM_API_ID` and `TELEGRAM_API_HASH`, add `TELEGRAM_BOT_TOKEN`.

## Project Structure

```
heatherbot/
  heather_telegram_bot.py     # Main bot (Bot API v4.0)
  persona_example.yaml        # Character template
  requirements.txt
  Dockerfile
  docker-compose.yml
  .env.example

  config/                     # Mounted volume — place your files here
    my_persona.yaml
    workflow_flux.json
    heather_face.png

  data/                       # Mounted volume — auto-generated at runtime
    conversations/            # Per-user chat history (JSON)
    logs/
```

## Known Limitations

- **Bot API constraints** — Bots only receive messages sent directly to them.
- **LLM hallucinations** — Small models occasionally invent backstory details not in the persona YAML.
- **Image generation speed** — ComfyUI FLUX.1 takes 60–120 seconds per image depending on GPU warmth.
- **FLUX negative prompts** — FLUX.1 dev has weak response to negative prompts vs SDXL. Set them in the workflow JSON directly.
- **LoRA dependency** — NSFW LoRAs must be installed in ComfyUI separately. The bot checks availability and skips missing ones.

## Disclaimer

This software is for creating AI companion chatbots and can be configured for adult content.

- **You are responsible** for complying with applicable laws regarding adult content and automated messaging.
- **AI disclosure is built in** — the bot discloses its AI nature to new users. Do not disable this.
- **Content safety systems** (CSAM flagging, blocked user management) are included and should remain active.
- **This is a local tool** — no data leaves your machine unless you configure it to.

Use responsibly.

## License

MIT License. See [LICENSE](LICENSE) for details.
