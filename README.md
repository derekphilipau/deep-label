# deep-label

**deep-label** is an intelligent, agentic computer vision pipeline designed to analyze complex artwork with “obnoxiously exhaustive” detail.

Unlike standard object detection models that only find the most obvious elements (e.g., "person," "dog"), this system uses a **recursive feedback loop** to force Large Language Models (LLMs) to look deeper, identifying specific background details, individual crowd members, and subtle narrative elements.

It then uses this exhaustive data as "Ground Truth" to generate high-quality, hallucination-free accessibility descriptions (Alt Text and Long Descriptions) for museum contexts.

---

## 📖 Non-Technical Overview

Imagine asking a curator to list everything they see in a painting.
1.  **First pass:** They list the main subjects—the castle, the king, the river.
2.  **Second pass:** You hand them their own list and say, *"These are already labeled. What else do you see?"* They look closer and list the hunters, the boat, and the dogs.
3.  **Third pass:** You repeat the process. Now they are squinting at the background, listing the tiny village, the birds in the trees, and the specific clothing of the servants.

**deep-label** automates this process. It refuses to stop until it has "squeezed" every drop of visual information from the image.

Once it has a massive list of verified facts, it passes that list to a "Writer" agent. Because the Writer has a verified list of facts, it doesn't make things up. It knows exactly how many dogs are in the pack and exactly where they are standing, resulting in a highly accurate visual description.

---

## 🔄 System Flow

```mermaid
graph TD
    %% Nodes
    Input(Input Image)
    subgraph Phase 1: The Detective
    Agent[Labeling Agent]
    Verify{No-new threshold met?}
    Update[Update Context: Include labeled instances (box+label)]
    JSON[Master JSON Payload]
    end
    
    subgraph Phase 2: The Writer
    Prompt[Description Prompt]
    Describer[Description Agent]
    end

    subgraph Phase 3: Visualization
    Annotate[Annotate Image (sharp + SVG)]
    end
    
    Output(Final Output: JSON + Descriptions + Annotated Image)

    %% Flow
    Input --> Agent
    Agent --> Verify
    Verify -- Yes --> Update
    Update --> Agent
    Verify -- No --> JSON
    
    JSON --> Prompt
    Input --> Prompt
    Prompt --> Describer
    Describer --> Annotate
    Input --> Annotate
    Annotate --> Output

    %% Styling
    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Output fill:#9f9,stroke:#333,stroke-width:2px
    style JSON fill:#ff9,stroke:#333,stroke-width:2px
```

---

## ⚙️ How It Works

### Phase 1: Recursive Detection (The Agent)
Written in TypeScript using the **Vercel AI SDK**, the agent queries a multimodal LLM (Google Gemini).
1.  **Normalization:** The image space is normalized to a `[0-1000]` integer coordinate system.
2.  **Schema Enforcement:** The model is forced to output structured JSON, preventing conversational fluff.
3.  **Context Injection:** In every loop, the script injects a list of already-labeled instances (label + bounding box) and instructs the model to find additional instances (including more people/dogs) without duplicating existing boxes.
4.  **Stopping Rule:** The run stops after a configurable number of consecutive “no new objects” iterations, then optionally runs extra verification passes to catch remaining misses.

### Phase 2: Grounded Description (The Describer)
Once the exhaustive list is complete, the system generates accessibility text.
*   **Fact Checking:** The detected object list acts as a guardrail against hallucinations.
*   **Spatial Accuracy:** The `0-1000` coordinates allow the system to accurately describe "foreground," "background," "left," and "right" without guessing.
*   **Accessibility Standards:** The output follows strict museum standards for Alt Text (10-18 words) and Long Descriptions (150-200 words).

---

## 🛠️ Tech Stack

*   **Runtime:** Node.js / TypeScript
*   **AI Framework:** [Vercel AI SDK](https://sdk.vercel.ai/docs) (`generateObject`)
*   **Model Provider:** Google Generative AI (Gemini)
*   **Validation:** Zod (Schema validation)
*   **Visualization:** `sharp` + SVG overlay (annotated output image)

---

## 🚀 Getting Started

### 1. Installation
Clone the repo and install dependencies:
```bash
npm install
```

### 2. Environment Setup
Create a `.env` file in the root directory:
```env
GOOGLE_GENERATIVE_AI_API_KEY=your_api_key_here
```

### 3. Usage
Place your target image in the root folder (e.g., `artwork.jpg`).

Run the agent:
```bash
npx tsx agent.ts --image artwork.jpg --output detected_objects.json
```

For more exhaustive detection on dense scenes, increase passes and require multiple “no new objects” loops before stopping:
```bash
npx tsx agent.ts --image artwork.jpg --output detected_objects.json --max-iterations 16 --no-new-threshold 3 --verify-passes 2
```

### 4. Output
The system produces:
*   `detected_objects.json`: full output payload including the exhaustive object list and final accessibility descriptions.
*   `annotated_painting.png` (or your chosen `--annotated-output`): an image with bounding boxes drawn.

Annotated image generation is done in TypeScript using `sharp` + an SVG overlay. No Python is required.

---

## 📊 Data Format

The output file is a single JSON object. The primary fields you’ll use are `objects` and `descriptions`:

```json
{
  "image_path": "painting.jpg",
  "model_name": "gemini-3-pro-preview",
  "description_model_name": "gemini-3-pro-preview",
  "no_new_threshold": 2,
  "verify_passes": 1,
  "objects": [
    {
      "label": "Crossbowman in Blue Doublet",
      "type": "person",
      "box_2d": [80, 640, 155, 780]
    }
  ],
  "descriptions": {
    "alt_text": "…",
    "long_description": "…"
  }
}
```

Each object’s `box_2d` is normalized to `[0, 1000]` as `[xmin, ymin, xmax, ymax]`.

---

## Notes (Before You Rename / Next Session)

*   **Model IDs change:** If Gemini model names change again, override via `--model`, `--description-model`, or env vars `MODEL_NAME` / `DESCRIPTION_MODEL_NAME`.
*   **Density vs. duplicates:** High density is expected and useful; expect some near-duplicates due to label variation. Post-processing (clustering by IoU, grouping by type) is a good next step before UI work.
*   **Prompt context limit:** Only a subset of labeled instances is injected per iteration to keep prompts bounded. If you want even deeper recall, raise `CONTEXT_OBJECT_LIMIT` via env var.
*   **Cost/latency:** Dense runs (many iterations + verification passes) can be expensive; treat `--max-iterations`, `--no-new-threshold`, and `--verify-passes` as “quality knobs”.
*   **Web app direction:** The `objects` list + normalized coords are already a good substrate for a zoomable viewer; a common next step is tiling the image and running targeted passes per tile for small/distant details.
*   **Renaming the folder:** After you rename the directory/repo, consider also updating the `package.json` `"name"` field if you plan to publish or install it as a package.

## 🔮 Future Roadmap
*   **Visualization Improvements:** Improve label layout, collision handling, and interactive web previews.
*   **Model Agnosticism:** Add support for GPT-4o and Claude 3.5 Sonnet via Vercel AI SDK.
*   **Batch Processing:** Run the agent across an entire folder of museum assets.
