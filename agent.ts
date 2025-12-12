import { generateObject } from 'ai';
import { google } from '@ai-sdk/google';
import { z } from 'zod';
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { annotateImage } from './annotator';

dotenv.config();

// ==========================================
// CONFIGURATION / CLI
// ==========================================
const DEFAULT_IMAGE_PATH = process.env.IMAGE_PATH || 'painting.jpg';
const DEFAULT_OUTPUT_FILE = process.env.OUTPUT_FILE || 'detected_objects.json';
const DEFAULT_ANNOTATED_OUTPUT = process.env.ANNOTATED_OUTPUT || 'annotated_painting.png';
const DEFAULT_MAX_ITERATIONS = Number(process.env.MAX_ITERATIONS || 8);
const DEFAULT_NO_NEW_THRESHOLD = Number(process.env.NO_NEW_THRESHOLD || 2);
const DEFAULT_VERIFY_PASSES = Number(process.env.VERIFY_PASSES || 1);
const CONTEXT_OBJECT_LIMIT = Number(process.env.CONTEXT_OBJECT_LIMIT || 120);
const DEFAULT_MODEL_NAME = process.env.MODEL_NAME || 'gemini-3-pro-preview';
const DEFAULT_DESCRIPTION_MODEL_NAME =
  process.env.DESCRIPTION_MODEL_NAME || 'gemini-3-pro-preview';

type AgentConfig = {
  imagePath: string;
  outputFile: string;
  annotatedOutput: string;
  maxIterations: number;
  noNewThreshold: number;
  verifyPasses: number;
  modelName: string;
  descriptionModelName: string;
  annotate: boolean;
  resume: boolean;
  mock: boolean;
};

function printHelp() {
  console.log(`
Usage: npx tsx agent.ts [options]

Options:
  -i, --image <path>              Input image path (default: ${DEFAULT_IMAGE_PATH})
  -o, --output <path>             Output JSON path (default: ${DEFAULT_OUTPUT_FILE})
      --annotated-output <path>   Annotated image path (default: ${DEFAULT_ANNOTATED_OUTPUT})
      --max-iterations <n>        Max labeling loops (default: ${DEFAULT_MAX_ITERATIONS})
      --no-new-threshold <n>      Stop after N consecutive no-new loops (default: ${DEFAULT_NO_NEW_THRESHOLD})
      --verify-passes <n>         Review passes after looping (default: ${DEFAULT_VERIFY_PASSES})
      --model <name>              Detection model (default: ${DEFAULT_MODEL_NAME})
      --description-model <name>  Description model (default: ${DEFAULT_DESCRIPTION_MODEL_NAME})
      --resume                    Resume from existing output file
      --no-annotate               Skip annotated image step
      --mock                      Run without API calls (for local sanity checks)
  -h, --help                      Show help
`);
}

function requireValue(args: string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (!value || value.startsWith('-')) {
    console.error(`❌ Missing value for ${flag}`);
    process.exit(1);
  }
  return value;
}

function getConfig(): AgentConfig {
  const args = process.argv.slice(2);
  const config: AgentConfig = {
    imagePath: DEFAULT_IMAGE_PATH,
    outputFile: DEFAULT_OUTPUT_FILE,
    annotatedOutput: DEFAULT_ANNOTATED_OUTPUT,
    maxIterations: DEFAULT_MAX_ITERATIONS,
    noNewThreshold: DEFAULT_NO_NEW_THRESHOLD,
    verifyPasses: DEFAULT_VERIFY_PASSES,
    modelName: DEFAULT_MODEL_NAME,
    descriptionModelName: DEFAULT_DESCRIPTION_MODEL_NAME,
    annotate: true,
    resume: false,
    mock: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '-i':
      case '--image':
        config.imagePath = requireValue(args, i, arg);
        i++;
        break;
      case '-o':
      case '--output':
        config.outputFile = requireValue(args, i, arg);
        i++;
        break;
      case '--annotated-output':
        config.annotatedOutput = requireValue(args, i, arg);
        i++;
        break;
      case '--max-iterations': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value <= 0) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]}`);
          process.exit(1);
        }
        config.maxIterations = value;
        i++;
        break;
      }
      case '--no-new-threshold': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value <= 0) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]}`);
          process.exit(1);
        }
        config.noNewThreshold = value;
        i++;
        break;
      }
      case '--verify-passes': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 0) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]}`);
          process.exit(1);
        }
        config.verifyPasses = value;
        i++;
        break;
      }
      case '--model':
        config.modelName = requireValue(args, i, arg);
        i++;
        break;
      case '--description-model':
        config.descriptionModelName = requireValue(args, i, arg);
        i++;
        break;
      case '--resume':
        config.resume = true;
        break;
      case '--no-annotate':
        config.annotate = false;
        break;
      case '--annotate':
        config.annotate = true;
        break;
      case '--mock':
        config.mock = true;
        break;
      case '-h':
      case '--help':
        printHelp();
        process.exit(0);
      default:
        if (arg.startsWith('-')) {
          console.warn(`⚠️ Unknown argument ignored: ${arg}`);
        }
    }
  }

  return config;
}

// ==========================================
// SCHEMA DEFINITION
// ==========================================
const BoundingBoxSchema = z.object({
  label: z.string().min(1),
  type: z.string().min(1),
  box_2d: z
    .array(z.number().int().min(0).max(1000))
    .min(4)
    .max(4)
    .describe(
      'Bounding box [xmin, ymin, xmax, ymax] in normalized 0–1000 coordinates'
    ),
});

const ResponseSchema = z.object({
  objects: z.array(BoundingBoxSchema).describe('NEW objects found this iteration'),
  finished: z
    .boolean()
    .describe('True only if no further unlabeled objects remain'),
});

const ReviewSchema = z.object({
  objects: z
    .array(BoundingBoxSchema)
    .describe('Objects missing from the existing list'),
});

type DetectedObject = z.infer<typeof BoundingBoxSchema>;
type DescriptionPayload = {
  alt_text: string;
  long_description: string;
};

type OutputPayload = {
  image_path: string;
  model_name: string;
  description_model_name: string;
  max_iterations: number;
  no_new_threshold: number;
  verify_passes: number;
  iterations_run: number;
  finished: boolean;
  objects: DetectedObject[];
  descriptions: DescriptionPayload | null;
  generated_at: string;
};

// ==========================================
// HELPERS
// ==========================================
const DEBUG_ERRORS =
  process.env.DEBUG_ERRORS === '1' ||
  process.env.DEBUG_ERRORS === 'true' ||
  process.env.DEBUG_ERRORS === 'yes';

function safeStringify(value: unknown) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    const anyError = error as any;
    const parts: string[] = [];
    const name = anyError.name || 'Error';
    const message = anyError.message || String(error);
    parts.push(`${name}: ${message}`);
    if (anyError.code) parts.push(`code=${anyError.code}`);
    if (anyError.statusCode || anyError.status) {
      parts.push(`status=${anyError.statusCode ?? anyError.status}`);
    }
    if (anyError.type) parts.push(`type=${anyError.type}`);
    if (anyError.response) {
      const response = anyError.response as any;
      const status = response.status ?? response.statusCode;
      const statusText = response.statusText;
      const data = response.data ?? response.body;
      const summary = safeStringify({ status, statusText, data });
      parts.push(`response=${summary}`);
    }
    if (anyError.cause) {
      parts.push(`cause=${formatError(anyError.cause)}`);
    }
    return parts.join(' | ');
  }

  return safeStringify(error);
}

function logErrorDetails(prefix: string, error: unknown) {
  console.warn(prefix + formatError(error));
  if (DEBUG_ERRORS && error instanceof Error && error.stack) {
    console.warn(error.stack);
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

type Box2D = [number, number, number, number];

function normalizeBox(box: number[]): Box2D {
  let x1 = box[0] ?? 0;
  let y1 = box[1] ?? 0;
  let x2 = box[2] ?? 0;
  let y2 = box[3] ?? 0;
  x1 = clamp(Math.round(x1), 0, 1000);
  y1 = clamp(Math.round(y1), 0, 1000);
  x2 = clamp(Math.round(x2), 0, 1000);
  y2 = clamp(Math.round(y2), 0, 1000);

  const xmin = Math.min(x1, x2);
  const xmax = Math.max(x1, x2);
  const ymin = Math.min(y1, y2);
  const ymax = Math.max(y1, y2);

  return [xmin, ymin, xmax, ymax];
}

function normalizeObject(object: DetectedObject): DetectedObject {
  return {
    label: object.label.trim(),
    type: object.type.trim(),
    box_2d: normalizeBox(object.box_2d),
  };
}

function iou(a: Box2D, b: Box2D) {
  const [ax1, ay1, ax2, ay2] = a;
  const [bx1, by1, bx2, by2] = b;

  const interX1 = Math.max(ax1, bx1);
  const interY1 = Math.max(ay1, by1);
  const interX2 = Math.min(ax2, bx2);
  const interY2 = Math.min(ay2, by2);

  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const union = areaA + areaB - interArea;

  return union > 0 ? interArea / union : 0;
}

function isDuplicate(candidate: DetectedObject, existing: DetectedObject) {
  const sameLabel =
    candidate.label.toLowerCase() === existing.label.toLowerCase();
  return (
    sameLabel &&
    iou(
      normalizeBox(candidate.box_2d),
      normalizeBox(existing.box_2d)
    ) >= 0.7
  );
}

function mergeObjects(existing: DetectedObject[], incoming: DetectedObject[]) {
  const normalizedIncoming = incoming.map(normalizeObject);
  const filtered = normalizedIncoming.filter(
    (object) => !existing.some((e) => isDuplicate(object, e))
  );
  return [...existing, ...filtered];
}

function buildTypeSummary(objects: DetectedObject[]) {
  const counts = new Map<string, number>();
  for (const o of objects) {
    const key = (o.type || 'other').toLowerCase();
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([type, count]) => `${type}:${count}`)
    .join(', ');
}

function buildExistingContext(objects: DetectedObject[], limit = CONTEXT_OBJECT_LIMIT) {
  if (objects.length === 0) return '(none)';
  const slice = objects.length > limit ? objects.slice(-limit) : objects;
  return slice
    .map(
      (o) =>
        `- ${o.label} (${o.type}) box=[${o.box_2d
          .map((v) => Math.round(v))
          .join(', ')}]`
    )
    .join('\n');
}

function readExistingObjects(outputFile: string): DetectedObject[] {
  if (!fs.existsSync(outputFile)) return [];

  try {
    const raw = fs.readFileSync(outputFile, 'utf-8');
    const parsed = JSON.parse(raw);
    const objects = Array.isArray(parsed)
      ? parsed
      : Array.isArray(parsed?.objects)
        ? parsed.objects
        : [];
    const normalized: DetectedObject[] = [];
    for (const o of objects) {
      const result = BoundingBoxSchema.safeParse(o);
      if (result.success) {
        normalized.push(normalizeObject(result.data));
      }
    }
    return normalized;
  } catch {
    return [];
  }
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function withRetries<T>(
  fn: () => Promise<T>,
  attempts = 3,
  baseDelayMs = 1000
): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i < attempts; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      const delay = baseDelayMs * Math.pow(2, i);
      logErrorDetails(`⚠️ Attempt ${i + 1} failed. `, error);
      console.warn(`↪ retrying in ${delay}ms...`);
      await sleep(delay);
    }
  }
  throw lastError;
}

function writeOutput(outputFile: string, payload: OutputPayload) {
  fs.writeFileSync(outputFile, JSON.stringify(payload, null, 2));
}

async function tryGenerateAnnotatedImage(
  config: AgentConfig,
  objects: DetectedObject[]
) {
  if (!config.annotate) return;

  try {
    await annotateImage({
      imagePath: config.imagePath,
      objects: objects.map((o) => ({
        ...o,
        box_2d: normalizeBox(o.box_2d),
      })),
      outputPath: config.annotatedOutput,
    });
    console.log(`🖼️ Annotated image written to ${config.annotatedOutput}`);
  } catch (error) {
    logErrorDetails('⚠️ Annotation step failed. ', error);
  }
}

// ==========================================
// DESCRIPTION AGENT
// ==========================================
async function generateDescription(
  imageBuffer: Buffer,
  objects: DetectedObject[],
  modelName: string
): Promise<DescriptionPayload> {
  const contextData = objects
    .map((o) => {
      const xCenter = (o.box_2d[0] + o.box_2d[2]) / 2;
      const yCenter = (o.box_2d[1] + o.box_2d[3]) / 2;
      const hPos = xCenter < 333 ? 'Left' : xCenter > 666 ? 'Right' : 'Center';
      const vPos =
        yCenter < 333
          ? 'Background/Top'
          : yCenter > 666
            ? 'Foreground/Bottom'
            : 'Midground';
      return `- ${o.label} (${o.type}) at ${hPos}, ${vPos}`;
    })
    .join('\n');

  const prompt = `
You are an accessibility-focused describer for museum artworks.
You will receive an image and a list of verified objects with approximate locations.

VERIFIED OBJECTS (GROUND TRUTH):
${contextData || '(no objects provided)'}

RULES:
- Treat the verified list as factual; do not invent new objects.
- Use the locations to describe the scene in a stable left→right, foreground→background order.
- Aggregate repeats naturally (e.g., "a pack of hounds" instead of listing each).
- Describe only visible content; no symbolism, titles, dates, or artist intent.

OUTPUT:
Return only valid JSON with exactly these keys:
{"alt_text":"","long_description":""}

Alt text:
- 10–18 words, single sentence fragment, no final period.

Long description:
- One paragraph, ~150–220 words, present tense, plain language.
- Start with a brief overview, then a single consistent spatial pass.
`;

  const { object } = await generateObject({
    model: google(modelName),
    schema: z.object({
      alt_text: z.string(),
      long_description: z.string(),
    }),
    temperature: 0.2,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          { type: 'image', image: imageBuffer },
        ],
      },
    ],
  });

  return object;
}

async function reviewForMissingObjects(
  imageBuffer: Buffer,
  objects: DetectedObject[],
  modelName: string
): Promise<DetectedObject[]> {
  const existingContext = buildExistingContext(objects, CONTEXT_OBJECT_LIMIT);
  const typeSummary = buildTypeSummary(objects);

  const promptText = `
You are a meticulous computer vision reviewer.
You will receive an image and a list of ALREADY LABELED object instances (label + box in normalized 0–1000 coordinates).

Your task: find any MISSING visible object instances not already in the list.
It is OK to return additional instances of the same kind (e.g., more dogs or people) as long as they are different boxes.

ALREADY LABELED (${objects.length} total; by type: ${typeSummary || 'none'}):
${existingContext}

Instructions:
1. Return ONLY missing objects not already labeled.
2. If a missing object is similar to an existing one, give it a distinct label like "Hound #9" or "Distant hunter (new)".
3. Use conservative, tight boxes. Do not hallucinate hidden elements.
4. If nothing is missing, return an empty objects array.
`;

  const { object } = await generateObject({
    model: google(modelName),
    schema: ReviewSchema,
    temperature: 0.1,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'text', text: promptText },
          { type: 'image', image: imageBuffer },
        ],
      },
    ],
  });

  return object.objects;
}

// ==========================================
// LABELING AGENT
// ==========================================
async function runAgent() {
  const config = getConfig();
  console.log(`🚀 Starting Exhaustive Labeling Agent using ${config.modelName}...`);

  if (!config.mock && !process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
    console.error(
      '❌ GOOGLE_GENERATIVE_AI_API_KEY is missing. Add it to .env or set it in your environment.'
    );
    process.exit(1);
  }

  const resolvedImagePath = path.resolve(config.imagePath);
  if (!fs.existsSync(resolvedImagePath)) {
    console.error(`❌ Image not found at ${resolvedImagePath}`);
    process.exit(1);
  }

  const imageBuffer = fs.readFileSync(resolvedImagePath);

  let allObjects: DetectedObject[] = config.resume
    ? readExistingObjects(config.outputFile)
    : [];

  if (allObjects.length > 0) {
    console.log(`📌 Resuming with ${allObjects.length} existing objects.`);
  }

  let iteration = 1;
  let iterationsRun = 0;
  let isFinished = false;
  let noNewStreak = 0;

  while (!isFinished && iteration <= config.maxIterations) {
    console.log(`\n--- Iteration ${iteration} ---`);
    const existingContext = buildExistingContext(allObjects, CONTEXT_OBJECT_LIMIT);
    const typeSummary = buildTypeSummary(allObjects);

    let promptText = `
You are an expert computer vision annotator known for being "obnoxiously exhaustive."
Identify distinct visible objects in the image with bounding boxes normalized to [0, 1000].

Instructions:
1. Label people, animals, buildings, landscape features, and notable objects.
2. Use specific, curator-like labels (e.g., "Crossbowman in blue doublet", not just "person").
3. Return boxes as [xmin, ymin, xmax, ymax] with xmin < xmax and ymin < ymax.
4. Treat each visible instance as separate. If you see multiple dogs or people, return EACH with its own box.
   If you find another of the same kind, give it a distinct label like "Hound #7" or "Distant hunter (new)".
5. Do not guess hidden elements.

Systematic scan checklist:
- Foreground left → right, then midground left → right, then background left → right.
- Look for small/distant figures, animals, and background details after main subjects.
`;

    if (iteration === 1) {
      promptText += '\nStart by identifying the most prominent objects.';
    } else {
      promptText += `

Already labeled instances (${allObjects.length} total; by type: ${typeSummary || 'none'}):
${existingContext}

CRITICAL:
- Do NOT return duplicate boxes for the same instances above.
- It IS OK to return more instances of the same kind if they are different boxes.
- Look deeper for background details, small animals, distant figures, or distinct garments you missed.
- If you cannot find any NEW objects this pass, set "finished" to true.
`;
    }

    if (config.mock) {
      console.log('🧪 Mock mode: skipping API call.');
      isFinished = true;
    } else {
      try {
        const { object } = await withRetries(() =>
          generateObject({
            model: google(config.modelName),
            schema: ResponseSchema,
            temperature: 0.2,
            messages: [
              {
                role: 'user',
                content: [
                  { type: 'text', text: promptText },
                  { type: 'image', image: imageBuffer },
                ],
              },
            ],
          })
        );

        const merged = mergeObjects(allObjects, object.objects);
        const addedCount = merged.length - allObjects.length;

        if (addedCount > 0) {
          console.log(`✅ Added ${addedCount} new objects.`);
          allObjects = merged;
          noNewStreak = 0;
        } else {
          console.log('⚠️ No new objects after de-duplication.');
          noNewStreak += 1;
        }

        if (object.finished && addedCount === 0) {
          console.log(
            '🏁 Model signaled completion; confirming with another scan.'
          );
        }

        if (noNewStreak >= config.noNewThreshold) {
          console.log(
            `🏁 Stopping after ${noNewStreak} consecutive no-new iterations.`
          );
          isFinished = true;
        }
      } catch (error) {
        logErrorDetails('❌ API Error. ', error);
        break;
      }
    }

    iterationsRun = iteration;

    writeOutput(config.outputFile, {
      image_path: config.imagePath,
      model_name: config.modelName,
      description_model_name: config.descriptionModelName,
      max_iterations: config.maxIterations,
      no_new_threshold: config.noNewThreshold,
      verify_passes: config.verifyPasses,
      iterations_run: iterationsRun,
      finished: isFinished,
      objects: allObjects,
      descriptions: null,
      generated_at: new Date().toISOString(),
    });

    iteration++;
  }

  if (!config.mock && config.verifyPasses > 0) {
    console.log('\n🔎 Running verification pass(es) for missing objects...');
    for (let pass = 1; pass <= config.verifyPasses; pass++) {
      console.log(`\n--- Verification Pass ${pass} ---`);
      try {
        const missingObjects = await withRetries(() =>
          reviewForMissingObjects(imageBuffer, allObjects, config.modelName)
        );
        const before = allObjects.length;
        allObjects = mergeObjects(allObjects, missingObjects);
        const added = allObjects.length - before;

        if (added > 0) {
          console.log(`✅ Verification added ${added} objects.`);
        } else {
          console.log('✅ Verification found no new objects.');
          break;
        }
      } catch (error) {
        logErrorDetails('❌ Verification pass failed. ', error);
        break;
      }
    }
  }

  console.log(
    `\n✨ Process Complete. Total unique objects found: ${allObjects.length}`
  );

  let descriptions: DescriptionPayload | null = null;
  if (!config.mock) {
    console.log('\n📝 Generating accessibility descriptions...');
    try {
      descriptions = await withRetries(() =>
        generateDescription(imageBuffer, allObjects, config.descriptionModelName)
      );
      console.log('\nAlt text:\n' + descriptions.alt_text);
      console.log('\nLong description:\n' + descriptions.long_description);
    } catch (error) {
      logErrorDetails('❌ Description generation failed. ', error);
    }
  }

  const finalPayload: OutputPayload = {
    image_path: config.imagePath,
    model_name: config.modelName,
    description_model_name: config.descriptionModelName,
    max_iterations: config.maxIterations,
    no_new_threshold: config.noNewThreshold,
    verify_passes: config.verifyPasses,
    iterations_run: iterationsRun,
    finished: isFinished,
    objects: allObjects,
    descriptions,
    generated_at: new Date().toISOString(),
  };

  writeOutput(config.outputFile, finalPayload);
  console.log(`📂 Full output written to ${config.outputFile}`);
  await tryGenerateAnnotatedImage(config, allObjects);
}

runAgent();
