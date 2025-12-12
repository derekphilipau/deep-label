# Repository Guidelines

## Project Structure & Module Organization
- `agent.ts`: TypeScript agent that calls the Vercel AI SDK with Google Gemini to iteratively label an input image, writing results to `detected_objects.json`.
- `annotator.ts`: TypeScript visualizer that overlays normalized bounding boxes via `sharp` + SVG and saves an annotated PNG.
- Assets: `artwork.jpg` (sample input), `annotated_painting.png` (generated output).
- Tooling: `package.json` with `tsx`/TypeScript runtime deps and `sharp` for image overlays.

## Setup, Build, Test, and Development Commands
- Install Node deps: `npm install` (Node 18+ recommended).
- Run the agent (requires `GOOGLE_GENERATIVE_AI_API_KEY` in `.env`): `npx tsx agent.ts`. Output: `detected_objects.json` with bounding boxes plus alt-text/long description in stdout.
- Annotated image is generated automatically at the end of the run (`annotated_painting.png` by default).
- Tests: no automated suite yet; `npm test` is a placeholder that currently exits with an error.

## Coding Style & Naming Conventions
- TypeScript/JavaScript: 2-space indent, prefer `const`/`let`, async/await for promises. Uppercase for constants (e.g., `IMAGE_PATH`, `MODEL_NAME`), camelCase for functions/vars (`runAgent`, `generateDescription`).
- JSON outputs: indent with two spaces; bounding boxes normalized to `[0, 1000]` as `[xmin, ymin, xmax, ymax]`.

## Testing & Verification Guidelines
- After `npx tsx agent.ts`, confirm `detected_objects.json` exists and includes new objects per iteration plus `finished` logic respected in logs.
- For visualization, inspect `annotated_painting.png` to ensure labels align with objects and color categories match the type mapping; adjust coordinates before committing.
- When changing prompts or schemas, sanity-check with a short run (limit iterations) to avoid excessive token use.

## Commit & Pull Request Guidelines
- Use clear, imperative commits (e.g., `add sharp overlay for bounding boxes`, `tune gemini prompt for alt text`). If a formal standard is adopted later, follow that.
- PRs should state intent, summarize changes, list how to run/verify (`npx tsx agent.ts`), and attach updated artifact snippets if relevant (`detected_objects.json`, `annotated_painting.png` crops).
- Keep secrets out of commits (`.env`, API keys); prefer `.env` and document required variables.
