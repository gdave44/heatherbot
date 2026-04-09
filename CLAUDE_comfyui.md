# HeatherBot → ComfyUI Integration — Context for Claude

This document captures everything about how HeatherBot uses ComfyUI, intended as
starting context for a companion project that streamlines ComfyUI specifically for
HeatherBot's image generation needs.

---

## Overview

HeatherBot generates character images on demand via ComfyUI's REST API.  The bot
does **not** use the ComfyUI web UI at runtime — it constructs or modifies a
workflow JSON in memory, posts it to `/prompt`, polls `/history/{id}`, and fetches
the finished image from `/view`.

The pipeline is **FLUX.1 dev** with ReActor face-swap.  Negative prompts are
effectively ignored by FLUX (kept in the workflow for structural compatibility).

---

## Environment Variables (heatherbot side)

```
COMFYUI_URL=http://comfyui:8188          # Base URL — no trailing slash
COMFYUI_WORKFLOW_FILE=/app/config/workflow_flux.json   # Base workflow JSON
COMFYUI_FACE_IMAGE=/app/config/heather_face.png        # Source face for ReActor
```

Default port: **8188**.

---

## REST API Calls Made by HeatherBot

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload/image` | POST (multipart) | Upload face image to ComfyUI input dir on first use |
| `/prompt` | POST (JSON) | Queue a workflow for generation |
| `/history/{prompt_id}` | GET | Poll until job completes or errors |
| `/view?filename=…&subfolder=…&type=output` | GET | Fetch finished image bytes |
| `/object_info/LoraLoader` | GET | Enumerate available LoRA files at startup |
| `/system_stats` or `/queue` | GET | Health check (ComfyUI alive?) |

### Face Image Upload

On first generation, heatherbot calls `upload_image_to_comfyui()` which POSTs
`COMFYUI_FACE_IMAGE` to `/upload/image` with `multipart/form-data`.  The returned
filename is cached globally and written into `workflow["10"]["inputs"]["image"]`
on every subsequent call.  If the upload fails, it falls back to
`os.path.basename(HEATHER_FACE_IMAGE)` (assumes the file is already in ComfyUI's
`input/` directory).

### LoRA Availability Check

At first LoRA use, heatherbot calls `GET /object_info/LoraLoader` and parses the
`LoraLoader.input.required.lora_name[0]` list into a set.  Any LoRA not in that
set is silently skipped with a warning log.  The set is cached for the process
lifetime (no hot-reload).

---

## Workflow JSON Structure (workflow_flux.json)

HeatherBot expects a specific set of node IDs.  It reads the workflow file once at
startup and deep-copies it in memory for each generation, mutating only the fields
listed below.  **Everything else in the JSON is left exactly as loaded** — including
the negative prompt, sampler settings, dimensions, etc.

### Fixed Node ID Map

| Node ID | Class / Role | Fields heatherbot writes |
|---|---|---|
| `"1"` | CheckpointLoaderSimple | Checkpoint source — LoRA chain starts here (`model[0]`, `clip[1]`) |
| `"3"` | CLIPTextEncode (positive) | `inputs.text` ← full positive prompt |
| `"4"` | CLIPTextEncode (negative) | `inputs.clip` ← rewired when LoRAs active; text left from JSON |
| `"5"` | FluxGuidance | `inputs.guidance` ← `FLUX_GUIDANCE` (default 5.0) |
| `"6"` | EmptyLatentImage | `inputs.width` / `inputs.height` — overridden for landscape poses |
| `"7"` | KSampler | `inputs.seed`, `inputs.model`, `inputs.positive`, `inputs.negative` |
| `"8"` | VAEDecode | Intermediate output (pre-face-swap) |
| `"9"` | SaveImage (final) | **Primary output node** — heatherbot reads from this |
| `"10"` | LoadImage (face) | `inputs.image` ← uploaded face filename |
| `"11"` | ReActorFaceSwap | Face swap node |
| `"12"` | SaveImage (fallback) | Fallback output node — checked if node 9 has no images |
| `"14"` | KSampler (face blend) | `inputs.seed` ← randomized alongside node 7 |
| `"15"` | (blend/composite) | Part of face-swap pipeline |

**Dynamic nodes injected at runtime** (not in base JSON):

| Node ID | Injected when | Role |
|---|---|---|
| `"20"` | NSFW, `NSFW_master.safetensors` available | NSFW Master LoRA |
| `"21"` | NSFW + vulva keywords + `flux-female-anatomy.safetensors` available | Anatomy detail LoRA |
| `"22"` | NSFW + toy keywords + `flux1-dildo1.safetensors` available | Dildo/toy LoRA |
| `"50"` | Pose with `use_controlnet: true` | LoadImage (pose skeleton PNG) |
| `"51"` | Pose with ControlNet | ControlNetLoader |
| `"52"` | Pose with ControlNet | ControlNetApplySD3 |

### Seed Randomization

Nodes `"7"` and `"14"` both have their `seed` field set to `random.randint(0, 2**53 - 1)`
on every call.  This is the only way heatherbot varies the output — all other
sampler parameters come from the JSON.

---

## Prompt Construction Pipeline

The full positive prompt sent to node 3 is built in this order:

```
[scene description], [character physical prefix][quality suffix]
```

### 1. User Request → Raw Description

`extract_image_description(user_message)` strips the conversational framing
("send me a picture of you…") to extract the scene description.

- If the extracted description is **≥ 30 characters**: used verbatim (user gave real detail)
- If **< 30 characters** and NSFW: replaced with a hardcoded pose-specific or generic NSFW description
- If **not NSFW**: returned as-is for LLM expansion

### 2. NSFW Lock-In

`_is_nsfw_context(raw_description)` is called **before** any LLM expansion.  The
boolean result (`original_is_nsfw`) is passed through the entire pipeline so the
clothed/nude decision cannot be flipped by later expansion.

NSFW detection uses two passes:
- Word list: `nude`, `naked`, `topless`, `pussy`, `cock`, `penis`, etc.
- Regex: anatomy word (`breast|nipple|ass|…`) near qualifier (`exposed|bare|visible|…`)

### 3. Sex Toy Context Propagation

After LLM expansion, heatherbot scans the last 10 conversation messages for
sex toy keywords (`dildo`, `vibrator`, `sex toy`, `butt plug`, `strap-on`).
If found and the expanded description doesn't already contain them, it appends
`", using a <prop>"` so the dildo LoRA keyword check fires.

### 4. LLM Context Expansion

`build_image_prompt_from_context(chat_id, anchor)` calls the text LLM with:
- The cleaned anchor (pose/scene from the user's request)
- Last 10 conversation messages as context
- Character persona details (occupation, location, typical dress)
- Instruction: **add setting and pose only if absent; never change the anchor;
  never infer nudity or clothing from history**

Output is capped at 80 tokens, ~15–25 words added.

### 5. Character Prefix (from persona YAML)

`personality.get_image_prompt_prefix(nsfw=is_nsfw)` builds the character description:
- Reads `physical.hair`, `physical.eyes`, `physical.body_type`, `identity.age`
- SFW: uses `physical.chest_description`
- NSFW: uses `physical.breast_description`
- Override: set `physical.image_prompt.sfw_prefix` / `nsfw_prefix` in the YAML to hardcode it

### 6. Quality Suffix (from persona YAML)

`personality.get_image_quality_suffix(nsfw=is_nsfw)` reads:
- `physical.image_prompt.quality_suffix` (SFW)
- `physical.image_prompt.nsfw_quality_suffix` (NSFW)

Both suffixes emphasize authentic amateur phone-camera aesthetics, realistic skin,
correct anatomy, and five-fingered hands.

---

## LoRA Injection Details

All LoRA nodes are injected dynamically — they are **not** in the base workflow JSON.
They chain off the checkpoint (node 1) and the KSampler (node 7) and both CLIP
encoders (nodes 3 and 4) are rewired to follow the last LoRA in the chain.

### Chain topology

```
Node 1 (checkpoint)
  └─► Node 20 (NSFW_master.safetensors, strength 0.8)      [if NSFW]
        └─► Node 21 (flux-female-anatomy.safetensors, 0.5)  [if vulva keywords]
              └─► Node 22 (flux1-dildo1.safetensors, 0.75)  [if toy keywords]
                    └─► Node 7 (KSampler) ← model
                    └─► Nodes 3, 4 (CLIP encoders) ← clip
```

### Trigger keywords for each LoRA

| LoRA | Trigger keywords in prompt |
|---|---|
| `NSFW_master.safetensors` | Any NSFW context (`is_nsfw=True`) |
| `flux-female-anatomy.safetensors` | `pussy`, `vulva`, `labia`, `spread`, `laying`, `legs apart`, `exposed`, `closeup` |
| `flux1-dildo1.safetensors` | `dildo`, `vibrator`, `sex toy`, `butt plug`, `using a toy`, `strap-on`, `strapon` |

Any LoRA not found in ComfyUI's available list is skipped with a `WARNING` log entry.

---

## Pose System

Poses are detected from the prompt text via `POSE_KEYWORDS` (ordered list, most
specific first).  Each pose has a `POSE_MAP` entry with:

- `prompt_boost`: prepended to the positive prompt
- `landscape`: if true, swaps canvas to 1344×768
- `use_controlnet`: if true, injects nodes 50/51/52 (skeleton PNG + ControlNet)
- `skip_face_swap`: if true, removes ReActor nodes 10/11/13/14/15 and wires
  the VAEDecode output directly to the final save node

**ControlNet model:** `FLUX-controlnet-union-pro-2.0.safetensors`  
**Strength:** 0.65, end percent: 0.65

Poses that are back-facing (`from_behind`, `ass_up`) skip face swap because
ReActor would paste the face on the back of the head.

Most poses use prompt-only (no ControlNet) because ControlNet causes hand/face
artifacts on some poses — only `from_behind`, `side_view`, and `ass_up` use it.

---

## OOM Handling

ComfyUI frequently hits CUDA OOM on the first attempt when VRAM is fragmented.
HeatherBot handles this with a retry:

1. Queue prompt → poll for result
2. If the status message contains `out of memory` / `cuda out of memory`:
   - Log warning
   - Sleep **5 seconds** (lets CUDA GC run)
   - Re-queue the **same workflow** (same seeds — already randomized)
   - Poll again
3. If retry also fails → exception propagates to the caller

Timeout per poll cycle: **300 seconds** (FLUX.1 dev takes ~60s warm, ~120s cold
load).  Polling interval: 2 seconds.

---

## Health Check / Circuit Breaker

`check_comfyui_status()` makes a lightweight GET to `/system_stats` or `/queue`.
A `ServiceHealth` circuit breaker tracks failures (threshold: 3 failures → open,
recovery: 120 seconds).  When the circuit is open, generation attempts are
rejected immediately without hitting ComfyUI.

---

## Face Swap Pipeline (ReActor)

- Source face: uploaded from `COMFYUI_FACE_IMAGE` on first use
- ReActor node: `"11"`
- The base workflow wires `VAEDecode("8") → ReActor("11") → SaveImage("9")`
- A second KSampler (`"14"`) handles face-blend post-processing
- For back-facing poses, the entire face-swap subgraph (nodes 10, 11, 13, 14, 15)
  is removed from the workflow dict and node 8's output is written directly to
  the final save node

---

## Negative Prompt

FLUX.1 dev largely ignores negative prompts.  The negative prompt text is defined
in `workflow_flux.json` (node 4) and is **never overwritten by heatherbot** — it
is left exactly as authored in the JSON.  The CLIP input for node 4 is rewired
when LoRAs are active (to follow the last LoRA in the chain) but the text stays.

---

## Key Files on Disk

```
/app/config/workflow_flux.json    # Base ComfyUI workflow — source of truth for node wiring
/app/config/heather_face.png      # Face image uploaded to ComfyUI for ReActor
/app/config/poses/                # Skeleton PNGs for ControlNet poses
    from_behind.png
    side_view.png
    ass_up.png
    (others used prompt-only, no PNG needed)
```

---

## What a Streamlined ComfyUI Setup Needs

Based on heatherbot's usage, a purpose-built ComfyUI instance requires:

**Models:**
- FLUX.1 dev checkpoint (in `models/checkpoints/`)
- ReActor face-swap extension + insightface models
- FLUX ControlNet Union Pro 2.0 (in `models/controlnet/`)

**LoRAs** (in `models/loras/`):
- `NSFW_master.safetensors`
- `flux-female-anatomy.safetensors`
- `flux1-dildo1.safetensors`

**Custom nodes required:**
- ComfyUI-ReActor (face swap)
- ComfyUI ControlNet nodes (for `ControlNetApplySD3` class)
- Standard FLUX nodes (usually bundled)

**Not needed:**
- The web UI (ComfyUI can run `--headless` or with `--disable-auto-launch`)
- Any non-FLUX checkpoints
- Any samplers other than what's in the workflow JSON
- SDXL, SD1.5, or any other model families

**Performance notes:**
- FLUX.1 dev is VRAM-hungry (~12–16 GB for full precision, 8–10 GB quantized)
- OOM on first attempt is common when VRAM is fragmented — the 5s retry handles it
- `--lowvram` or `--medvram` flags reduce quality and increase generation time
- Offloading to CPU ram helps if you have enough system RAM (32+ GB recommended)
