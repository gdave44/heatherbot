# HeatherBot

**A fully local AI companion chatbot ecosystem for Telegram — with multi-platform funnel automation.**

HeatherBot is a Telegram bot (Bot API via python-telegram-bot) that runs entirely on your own hardware. No cloud APIs, no OpenAI, no subscriptions. Just a local LLM, a persona YAML file, and a stack of services that make the bot feel like a real person texting.

## Architecture

```
  +-------------------+     +-------------------+     +-------------------+
  |      Ollama       |     |      Ollama        |     |     ComfyUI       |
  | (Text: dolphin-   |     | (Vision: joycaption|     | (Image Generation)|
  |  llama3:8b)       |     |  /llava)           |     |    port 8188      |
  |    port 11434     |     |    port 11434      |     +--------+----------+
  +--------+----------+     +--------+----------+              |
           |                         |                         |
           +------------+------------+                         |
                        |                                      |
               +--------+----------+                          |
               |  HeatherBot Core  +-------------------------+
               |   (Bot API)       |
               |  + Flask monitor  |     +-------------------+
               |    port 8888      |     |    Kokoro TTS     |
               +--------+----------+     |    port 8000      |
                        |               +-------------------+
           +------------+------------+
           |                         |
  +--------+-------+       +---------+--------+
  | Frank Dashboard|       | OpenClaw MGMT    |
  | (Reddit/FetLife)|      | (Autonomous Agt) |
  |   port 8080    |       +------------------+
  +----------------+
```

| Service | Port | Purpose |
|---------|------|---------|
| Ollama (text) | 11434 | Text generation — `dolphin-llama3:8b` or any Ollama model |
| Ollama (vision) | 11434 | Image analysis — JoyCaption / LLaVA vision model |
| Ollama (prompt gen) | 11434 | FLUX prompt expansion — `brxce/stable-diffusion-prompt-generator` |
| ComfyUI | 8188 | Image generation (Pony / FLUX) with ReActor face-swap |
| Kokoro TTS | 8000 | Voice synthesis |
| Bot Monitor | 8888 | Web dashboard for analytics and admin |
| Frank Dashboard | 8080 | Multi-platform outreach (Reddit + FetLife auto-funnel) |

## Features

### Core Bot (Telegram)

- **Persona YAML system** — Define your character's identity, backstory, personality, physical description, and communication style in a single YAML file. Swap personas by pointing to a different file.
- **Persona self-learning** — Background LLM extraction of details the character invents during conversation. New facts are appended to a `learned_details` section in the persona YAML and injected into future sessions as `[DETAILS YOU'VE MENTIONED IN PAST CONVERSATIONS]`.
- **Bot API** — Standard Telegram bot using a BotFather token. Receives messages via long-polling. No phone number or session file required.
- **Conversation persistence** — Chat history saved to `/app/data/conversations/<chat_id>.json` and reloaded on restart.
- **Gender detection** — Detects user gender from early messages and injects pronoun context into the LLM system prompt.
- **Family awareness** — The character's family context (`family_awareness` in the persona YAML) is injected into the chat system prompt. Referenced naturally in conversation without breaking character.
- **Lust system** — 0–100 score per user. Tiers: `cold / warm / interested / hot`. HOT (≥70) unlocks more receptive sexual tone, higher token budgets, and expanded image reveal. Score decays over time and rises with engagement. Proactive photo sends trigger at elevated lust.
- **Story bank** — Pre-written explicit stories in `config/heather_stories.yaml`. Served on demand (60% banked / 40% LLM-generated) with per-user rotation and a 25-message cooldown. LLM-generated stories are automatically saved back to the YAML for future reuse.
- **Character violation guard** — Extended list of phrases caught and redirected: AI existence admissions ("I don't have a body", "I'm just an AI"), bodiless claims, and reality breaks. Responses are replaced with in-character deflections.
- **Photo context confusion** — If the user requests a photo without any sexual lead-up (≥4 turns, no escalation), the bot voices confusion rather than immediately complying.
- **AI disclosure** — Automatic first-message disclosure that includes the persona name. Reality-check responses own the AI status without breaking character.
- **Re-engagement** — Automatic outreach to inactive users. Configurable timing.
- **Voice messages** — Kokoro TTS for sending voice notes.
- **Monitoring dashboard** — Real-time Flask dashboard at port 8888. Access controlled by IP allowlist (`MONITOR_ALLOWED_IPS`) with CIDR subnet support.
- **Breadcrumb logging** — Detailed pipeline tracing gated behind `BREADCRUMB_LOGGING=1`.
- **Content safety** — CSAM flag-and-review, blocked user management, admin alerts.

### Image Generation Pipeline

```
User request
  → Family member detection      resolve name/role → age + physical details from persona
  → Minor safety guard           force SFW if any person under 18 is detected
  → CALL 1: dolphin-llama3       classify SFW/NSFW + write plain scene description
      - Character physical details injected
      - Family context injected (lust ≥ 75 required for permissive home settings)
      - Two-person face rule enforced (man's face always hidden/cropped)
      - Sexual act composition guide (_SEXUAL_ACT_GLOSSARY) injected when matched
      - Reveal limit from lust tier applied
  → CALL 2: brxce/stable-diffusion-prompt-generator
      - Expands plain scene description into comma-separated FLUX phrases
      - Specialized model trained on SD prompts — correct format natively
  → _enforce_family_ages()       splice age terms into final prompt (guaranteed)
  → build_heather_prompt()       prepend character prefix + quality suffix from persona
  → ComfyUI /prompt              workflow submitted with face image, LoRAs, seeds
  → ReActor face swap            gender-filtered (female only, both source and target)
  → SeedVR2 upscaler             optional upscale pass
```

**Two-model approach:** `dolphin-llama3` handles all context, rules, and classification. `brxce/stable-diffusion-prompt-generator` handles format — it was fine-tuned on SD prompts and naturally outputs comma-separated diffusion phrases without needing format instructions.

**Family member detection:** When a photo request mentions a family member by name ("Sofia") or role ("daughter", "kid", "husband"), the bot resolves them from the persona YAML and injects their age and physical details into the scene description. If any person under 18 is detected, the scene is hard-forced SFW at every return point.

**Sexual act glossary:** ~30-entry dictionary maps user phrases ("blowjob", "riding", "from behind") to precise FLUX composition guidance injected into the scene prompt.

**Face compositing rules:**
- Man's face is always obscured (turned away, cropped, in shadow) in two-person scenes
- No face-to-genitals proximity in any prompt
- ReActor face-swap targets female faces only (`detect_gender_input/source: female`)

### Vision / Image Analysis

- Incoming user photos are analyzed by the configured vision model (JoyCaption / LLaVA)
- Description is cached per photo and used as context for the LLM response
- NSFW image classifier (Falconsai ViT via `transformers`) with Ollama vision fallback
- Image rating responses are written in first person, directed at the user

### Story Bank (`config/heather_stories.yaml`)

```yaml
stories:
  story_key:
    kinks:
      - blowjob
      - stranger
    content: |
      Full first-person story text. 200+ words recommended.
      End with a question turning it back to the user.
```

- Hot-reload with `/stories reload` (no restart needed)
- LLM-generated stories are auto-saved with key `llm_YYYYMMDD_HHMMSS` and kinks auto-detected
- Stories enter rotation immediately after being saved

### Persona YAML — Key Sections

```yaml
identity:
  name: "Rebecca"
  age: 33
  occupation: "Licensed Massage Therapist"
  husband:
    name: "Dan"
    dynamic: "Mostly committed, but with a naughty streak"

family:
  children:
    - name: "Sofia"
      age: 12
      status: "Middle schooler"
      details: "Smart kid, into robotics and soccer."
  family_awareness: |
    Rebecca is a working mom. She keeps her online life separate.
    She never discusses Sofia in sexual contexts.

physical:
  hair: "brunette with natural silver streaks, long"
  eyes: "brown"
  body_type: "5'9\", fit and curvy, athletic build"
  chest_description: "full figure, strong"        # SFW prompts
  breast_description: "DD cup natural breasts"    # NSFW prompts

  image_prompt:
    sfw_prefix: ""          # blank = auto-build from fields above
    nsfw_prefix: ""
    quality_suffix: ", natural lighting, authentic amateur photo..."
    nsfw_quality_suffix: ", natural lighting, realistic skin with pores..."
    setting_context: ""     # hint for setting inference
```

### Monitoring Dashboard

Access controlled by IP allowlist. Set `MONITOR_ALLOWED_IPS` in `.env`:

```env
MONITOR_ALLOWED_IPS=192.168.1.0/24,10.0.0.1
```

Supports individual IPs and CIDR ranges. Defaults to `127.0.0.1,::1`.

## Hardware Requirements

**Minimum:**
- 1× GPU with 24GB VRAM (RTX 3090, RTX 4090, etc.)
- 32GB system RAM
- Python 3.11+

**Recommended:**
- 2× GPUs (one for LLM, one for image generation)
- 64GB+ system RAM
- SSD for model storage

## Prerequisites

1. **[Ollama](https://ollama.ai)** — Pull the required models:
   ```bash
   ollama pull dolphin-llama3:8b
   ollama pull brxce/stable-diffusion-prompt-generator
   ollama pull llava   # or your preferred vision model
   ```
2. **[ComfyUI](https://github.com/comfyanonymous/ComfyUI)** — For image generation (optional)
3. **[Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI)** — For voice messages (optional)
4. **Python 3.11+**
5. **Telegram bot token** — Create a bot via [@BotFather](https://t.me/BotFather)

## Docker Deployment

The recommended deployment. The image clones the latest code from GitHub on every build.

```bash
# Build and start
docker compose up -d heatherbot

# Force rebuild (pull latest code)
docker compose build --no-cache heatherbot && docker compose up -d heatherbot

# View logs
docker compose logs -f heatherbot
```

### Volumes

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./heatherbot/data` | `/app/data` | Conversation history, user state |
| `./heatherbot/config` | `/app/config` | Persona YAML, workflow JSON, face image, story bank |

Place your persona YAML, `workflow_pony.json` (or `workflow_flux.json`), `heather_face.png`, and optionally `heather_stories.yaml` in `./heatherbot/config/` before starting.

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

## Configuration

### Environment Variables

```env
# Telegram
TELEGRAM_BOT_TOKEN=your_botfather_token
ADMIN_USER_ID=your_telegram_numeric_id

# Persona
HEATHER_PERSONA=my_persona.yaml

# LLM / AI services
MODEL=dolphin-llama3:8b
PROMPT_GEN_MODEL=brxce/stable-diffusion-prompt-generator
VISION_MODEL=llama-joycaption-beta-one-hf-llava.IQ4_XS
LLM_URL=http://ollama:11434
TTS_URL=http://kokoro:8000
COMFYUI_URL=http://comfyui:8188

# ComfyUI assets (place files in config volume)
COMFYUI_WORKFLOW_FILE=/app/config/workflow_pony.json
COMFYUI_FACE_IMAGE=/app/config/heather_face.png

# Photo cap
PHOTO_CAP=100                 # Max photos per rolling window
PHOTO_CAP_WINDOW_HOURS=2      # Rolling window in hours

# Monitoring dashboard
MONITOR_ALLOWED_IPS=127.0.0.1,::1

# Debugging
BREADCRUMB_LOGGING=0          # Set to 1 for full pipeline trace logs
```

### ComfyUI Setup

1. Place your workflow JSON at `config/workflow_pony.json` (Pony) or `config/workflow_flux.json` (FLUX)
2. Place a face source image at `config/heather_face.png`
3. The bot uploads the face image to ComfyUI's input directory automatically on first generation
4. Negative prompts are preserved from the workflow JSON — the bot does not overwrite them
5. LoRAs are checked at runtime against ComfyUI's available list; missing ones are logged and skipped
6. ReActor face-swap nodes should have `detect_gender_input` and `detect_gender_source` set to `"female"`

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
| `--log-dir` | `logs/` | Log directory |

### User Commands (in Telegram)

| Command | Description |
|---------|-------------|
| `/start` | Begin conversation |
| `/help` | Show available commands |
| `/selfie` | Request a photo (optionally with description) |
| `/reset` | Clear conversation history and photo cap |
| `/voice_on` / `/voice_off` | Toggle voice message mode |
| `/stories reload` | Hot-reload story bank from YAML (no restart) |

### Admin Commands

| Command | Description |
|---------|-------------|
| `/hubaloo tip <stars> [chat_id]` | Simulate a tip to test warmth changes |
| `/hubaloo lust <0-100> [chat_id]` | Directly set lust score for a chat |
| `/hubaloo story` | Force a story (random banked or LLM, 60/40 split) |
| `/hubaloo story llm` | Force an LLM-generated story |
| `/hubaloo story banked` | Force a banked story (respects per-user rotation) |
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

## Project Structure

```
heatherbot/
  heather_telegram_bot.py     # Main bot
  persona_example.yaml        # Character template
  requirements.txt
  Dockerfile
  docker-compose.yml
  .env.example

  config/                     # Mounted volume — place your files here
    my_persona.yaml           # Your character definition
    workflow_pony.json        # ComfyUI workflow (Pony or FLUX)
    heather_face.png          # Face source for ReActor
    heather_stories.yaml      # Story bank (auto-created, hot-reloadable)
    lora_info.yaml            # Optional LoRA metadata overrides

  data/                       # Mounted volume — auto-generated at runtime
    conversations/            # Per-user chat history (JSON)
    logs/
```

## Known Limitations

- **Bot API constraints** — Bots only receive messages sent directly to them.
- **LLM hallucinations** — Small models occasionally invent backstory details not in the persona YAML. The self-learning system captures and persists these so they stay consistent.
- **Image generation speed** — ComfyUI takes 10–30 seconds per image on modern GPUs; longer with the SeedVR2 upscaler pass.
- **Pony negative prompts** — Set directly in the workflow JSON. The bot preserves them from the workflow file.
- **LoRA dependency** — NSFW LoRAs must be installed in ComfyUI separately. The bot checks availability at runtime and skips missing ones.
- **Prompt generator model** — `brxce/stable-diffusion-prompt-generator` must be pulled in Ollama before image generation works. The bot will fall back to the raw cleaned request if the model is unavailable.

## Disclaimer

This software is for creating AI companion chatbots and can be configured for adult content.

- **You are responsible** for complying with applicable laws regarding adult content and automated messaging.
- **AI disclosure is built in** — the bot discloses its AI nature to new users. Do not disable this.
- **Content safety systems** (CSAM flagging, blocked user management, minor detection in image prompts) are included and should remain active.
- **This is a local tool** — no data leaves your machine unless you configure it to.

Use responsibly.

## License

MIT License. See [LICENSE](LICENSE) for details.
