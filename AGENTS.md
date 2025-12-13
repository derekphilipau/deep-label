# Repository Guidelines

## Model Configuration
- Default model: `gemini-3-pro-preview` (do not change without testing)
- Override via `--model <name>` flag or `MODEL_NAME` env var

## Project Structure

```
scripts/           # CLI tools
  agent.ts         # detection agent
  annotator.ts     # image annotation helper
src/               # Next.js web app
public/artworks/   # artwork data (per-slug directories)
```

### Detection Agent (`scripts/agent.ts`)
- Hybrid detect+verify agent using Vercel AI SDK with Google Gemini
- Phase 1: Discover object kinds with estimated counts
- Phase 2: For each kind, detect instances then verify with visual feedback
- Phase 3: Global deduplication and importance scoring
- Supports tiled detection for high-count objects
- Generates cutout images for each detected object

### Web Application
- Next.js app in `src/` for browsing detection results
- Artworks auto-discovered from `public/artworks/{slug}/`:
  - `artwork.json` - metadata (title, artist, date, medium)
  - `image.jpg` - the artwork image
  - `detected_objects.json` - detection payload
  - `cutouts/` - cutout images (full/ and thumb/)
  - `cutouts.json` - cutout index

## Commands
- `npm install` - install dependencies (Node 18+)
- `npm run dev` - start Next.js dev server
- `npm run agent <slug>` - run detection agent on an artwork

## Artwork Ingestion Workflow

1. Create a directory: `public/artworks/<slug>/`
2. Add an image file (named `image.jpg`, `image.png`, or any `.jpg`/`.png`)
3. (Optional) Add metadata: `artwork.json` with `{"title": "...", "artist": "...", "date": "..."}`
4. Run the agent:
   ```
   npx tsx scripts/agent.ts <slug>
   ```

This generates all outputs in the artwork directory:
- `detected_objects.json` - detection payload
- `cutouts/` - cutout images (full/ and thumb/)
- `cutouts.json` - cutout index
- `annotated.png` - annotated image with bounding boxes

If outputs already exist, use `--force` to overwrite.

### Agent CLI Options
```
npx tsx scripts/agent.ts <artwork-slug> [options]
npx tsx scripts/agent.ts -i <image> -o <output> [options]

Options:
      --force                  Overwrite existing output files
      --max-kinds <n>          Max kinds to discover (default: 50)
      --verify-rounds <n>      Verification rounds per kind (default: 2)
      --tile-threshold <n>     Tiled detection threshold (default: 12, 0=disable)
      --kind-concurrency <n>   Parallel kind processing (default: 3)
      --no-cutouts             Disable cutout generation
      --cutouts-format <fmt>   Cutout format: webp|png (default: webp)
      --cutouts-padding <pct>  Padding around cutouts (default: 0.10 = 10%)
      --model <name>           Detection model
      --no-annotate            Skip annotated image
      --only-kinds <list>      Only process specific kinds (e.g., "Hound,Stag")
      --mock                   Dry run without API calls
  -h, --help                   Show help
```

## Coding Style
- TypeScript: 2-space indent, prefer `const`/`let`, async/await
- Uppercase constants (`IMAGE_PATH`), camelCase functions (`runAgent`)
- Bounding boxes normalized to `[0, 1000]` as `[xmin, ymin, xmax, ymax]`

## Environment Variables
- `GOOGLE_GENERATIVE_AI_API_KEY` - required for agent
- `MODEL_NAME` - override default model
- `KIND_CONCURRENCY` - parallel kind processing (default: 3)
- `TILE_THRESHOLD` - tiled detection threshold (default: 12)
