import { z } from 'zod';
import sharp from 'sharp';
import { AIPool } from './pool';
import fs from 'fs';
import path from 'path';

// Import shared utilities
import {
  Box2D,
  StoredObject,
  DetectedObject,
  clamp,
  normalizeBox,
  dedupeObjectsByGeometry,
  computeImportanceGeom,
  mapTileBoxToFullImage,
  logErrorDetails,
  buildVerificationImage,
  sanitizeFilePart,
  runWithConcurrency,
} from './detection-utils';

// Import annotator
import { annotateImage } from './annotator';

// ==========================================
// RECURSIVE SCHEMAS
// ==========================================

const RecursiveKindSchema = z.object({
  kinds: z.array(
    z.object({
      label: z.string().describe("Singular noun (e.g. 'hound', 'face', 'tree')"),
      type: z.string().describe("Category (person, animal, object, etc)"),

      // Heuristic Metrics
      count: z.enum(['none', '1', 'few', 'many', 'crowd']).describe("Estimate in THIS view"),
      size: z.enum(['tiny', 'small', 'medium', 'large', 'giant']).describe("Size relative to THIS view"),
      importance: z.enum(['primary', 'secondary', 'background']).describe("Artistic importance"),

      // The Decision - expanded with segmentation strategies
      action: z.enum([
        'detect_individual',   // Box every instance (for distinct, countable objects)
        'detect_representative', // Box 3-8 diverse examples (many similar objects, secondary importance)
        'detect_region',       // Box as area/mass (textures, crowds, backgrounds)
        'zoom_in',             // Too small/dense, needs subdivision
        'ignore'               // Artifact/noise, skip entirely
      ]).describe(`
        detect_individual: Clear/large enough to box every instance now. Use for primary subjects.
        detect_representative: Many similar objects - detect a few examples (3-8) spatially distributed.
        detect_region: Detect as a bounding area/mass, not individual instances. Use for textures, crowds.
        zoom_in: Too small/dense/cluttered to box here, needs further subdivision.
        ignore: Artifact, noise, or not worth detecting. Skip.
      `),

      reason: z.string().describe("Brief reason for the action (e.g. 'Too many to count', 'Clearly visible')"),
    })
  ),

  // Global scene context for children
  scene_description: z.string().describe("Brief description of what is visible in this specific crop, to pass to child nodes."),
});

const DetectionSchema = z.object({
  objects: z.array(z.object({
    label: z.string(),
    type: z.string(),
    box_2d: z.array(z.number()).min(4).max(4),
    confidence: z.number().optional(),
  }))
});

const VerifyResponseSchema = z.object({
  wrong_indices: z
    .array(z.number().int().min(0))
    .describe('Indices of boxes to REMOVE (not this kind, or severely wrong)'),
  corrections: z
    .array(
      z.object({
        index: z.number().int().min(0),
        box_2d: z.array(z.number().int().min(0).max(1000)).min(4).max(4),
      })
    )
    .describe('Boxes that need position adjustment'),
  missing: z
    .array(
      z.object({
        box_2d: z.array(z.number().int().min(0).max(1000)).min(4).max(4),
      })
    )
    .describe('Instances visible but not yet boxed'),
  complete: z
    .boolean()
    .describe('True if ALL visible instances are now correctly boxed'),
});

// Schema for post-recursion reconciliation
const ReconciliationResultSchema = z.object({
  confirmed_indices: z
    .array(z.number().int().min(0))
    .describe('Indices of objects that ARE real and visible in the full image'),
  rejected_indices: z
    .array(z.number().int().min(0))
    .describe('Indices of objects that are artifacts, hallucinations, or misidentifications'),
  notes: z
    .string()
    .optional()
    .describe('Brief notes on what was filtered and why'),
});

// Schema for description generation
const DescriptionSchema = z.object({
  alt_text: z
    .string()
    .describe('Concise alt text (10-20 words) describing the image for screen readers'),
  long_description: z
    .string()
    .describe('Detailed description (150-250 words) of the artwork for accessibility'),
});

// ==========================================
// TYPES
// ==========================================

export type RecursiveConfig = {
  imagePath: string;
  outputFile: string;
  maxDepth: number;
  concurrency: number;
  modelName: string;
  minTileSize: number; // Don't split if tile is smaller than this (px)
  verifyRounds: number; // Number of verification rounds per detection (default: 2)
  // Output options
  annotate: boolean; // Generate annotated image
  annotatedOutput: string; // Path for annotated output
  generateDescriptions: boolean; // Generate alt text and long description
  descriptionModelName: string; // Model for descriptions (can be different)
  // Cutouts (optional)
  cutouts: boolean;
  cutoutsFormat: 'webp' | 'png';
  cutoutsThumbSize: number;
  cutoutsMax: number;
  cutoutsConcurrency: number;
  cutoutsPadding: number;
};

type DetectionResult = StoredObject & {
  depth: number;
  parent_context: string;
};

type NodeContext = {
  depth: number;
  rect: { left: number; top: number; width: number; height: number }; // Pixel coords in full image
  globalImageSize: { width: number; height: number };
  parentDescription: string;
  path: string; // e.g. "root.0.3"
};

// ==========================================
// CLASS
// ==========================================

export class RecursiveDetector {
  private pool: AIPool;
  private config: RecursiveConfig;
  private fullImageBuffer: Buffer | null = null;
  private results: DetectionResult[] = [];

  constructor(config: RecursiveConfig) {
    this.config = config;
    this.pool = {
      // Mock pool interface, will be replaced by real one in run()
      generateObject: async () => ({} as any),
    } as any;
  }

  // Initialize pool and load image
  async init() {
    // We'll use the createAIPool from the other file, but we need to import it dynamically or pass it in.
    // For now, we assume the caller sets up the pool or we duplicate the logic.
    // Let's just use the imported AIPool type and assume we get a real one.
    const { createAIPool } = await import('./pool');
    this.pool = createAIPool({
      maxConcurrency: this.config.concurrency,
      model: this.config.modelName,
      debug: true,
    });

    this.fullImageBuffer = fs.readFileSync(this.config.imagePath);
  }

  // Main recursive function
  async processNode(ctx: NodeContext): Promise<void> {
    const { depth, rect, globalImageSize, parentDescription, path: nodePath } = ctx;
    const logPrefix = `[${nodePath}] (d=${depth})`;

    console.log(`${logPrefix} Processing ${rect.width}x${rect.height} region...`);

    // 1. Crop Image
    const rawImageBuffer = await sharp(this.fullImageBuffer!)
      .extract({ left: rect.left, top: rect.top, width: rect.width, height: rect.height })
      .jpeg()
      .toBuffer();

    // Optimize for AI (resize if too large to prevent socket errors)
    const aiImageBuffer = await this.optimizeImageForAI(rawImageBuffer);

    // 2. Discovery & Heuristic
    const discovery = await this.discover(aiImageBuffer, parentDescription);

    console.log(`${logPrefix} Scene: "${discovery.scene_description}"`);

    const toDetectIndividual: typeof discovery.kinds = [];
    const toDetectRepresentative: typeof discovery.kinds = [];
    const toDetectRegion: typeof discovery.kinds = [];
    let needsZoom = false;

    for (const k of discovery.kinds) {
      console.log(`${logPrefix}   - ${k.label}: ${k.size}, ${k.count} -> ${k.action} (${k.reason})`);

      if (k.action === 'detect_individual') {
        toDetectIndividual.push(k);
      } else if (k.action === 'detect_representative') {
        toDetectRepresentative.push(k);
      } else if (k.action === 'detect_region') {
        toDetectRegion.push(k);
      } else if (k.action === 'zoom_in') {
        needsZoom = true;
      }
      // 'ignore' actions are simply skipped
    }

    // 3a. Detect Individual Objects (with full verification)
    if (toDetectIndividual.length > 0) {
      console.log(`${logPrefix}   📍 Detecting ${toDetectIndividual.length} kinds (individual)...`);

      const detected = await this.detectWithVerification(
        aiImageBuffer,
        toDetectIndividual,
        `${logPrefix}      `
      );

      for (const obj of detected) {
        const globalBox = this.localToGlobalBox(obj.box_2d, rect, globalImageSize);
        this.results.push({
          label: obj.label,
          type: obj.type,
          box_2d: globalBox,
          depth,
          parent_context: parentDescription
        });
      }
      console.log(`${logPrefix}   ✅ Found ${detected.length} individual objects`);
    }

    // 3b. Detect Representative Objects (limited to 3-8 spatially diverse examples)
    if (toDetectRepresentative.length > 0) {
      console.log(`${logPrefix}   📍 Detecting ${toDetectRepresentative.length} kinds (representative)...`);

      // Use simpler detection without full verification for representative
      const detected = await this.detectRepresentative(
        aiImageBuffer,
        toDetectRepresentative,
        `${logPrefix}      `
      );

      for (const obj of detected) {
        const globalBox = this.localToGlobalBox(obj.box_2d, rect, globalImageSize);
        this.results.push({
          label: obj.label,
          type: obj.type,
          box_2d: globalBox,
          depth,
          parent_context: parentDescription
        });
      }
      console.log(`${logPrefix}   ✅ Found ${detected.length} representative objects`);
    }

    // 3c. Detect Regions (bounding areas, not individual instances)
    if (toDetectRegion.length > 0) {
      console.log(`${logPrefix}   📍 Detecting ${toDetectRegion.length} kinds (region)...`);

      const detected = await this.detectRegions(
        aiImageBuffer,
        toDetectRegion,
        `${logPrefix}      `
      );

      for (const obj of detected) {
        const globalBox = this.localToGlobalBox(obj.box_2d, rect, globalImageSize);
        this.results.push({
          label: obj.label,
          type: obj.type,
          box_2d: globalBox,
          depth,
          parent_context: parentDescription
        });
      }
      console.log(`${logPrefix}   ✅ Found ${detected.length} region boxes`);
    }

    // 4. Recurse (Split) if needed
    if (needsZoom) {
      if (depth >= this.config.maxDepth) {
        console.warn(`${logPrefix}   ⚠️ Max depth reached, cannot zoom further.`);
      } else if (rect.width < this.config.minTileSize || rect.height < this.config.minTileSize) {
        console.warn(`${logPrefix}   ⚠️ Tile too small (${rect.width}px), cannot zoom further.`);
      } else {
        console.log(`${logPrefix}   🔲 Splitting into quadrants...`);
        await this.splitAndRecurse(ctx, discovery.scene_description);
      }
    }
  }

  async splitAndRecurse(ctx: NodeContext, currentDescription: string) {
    const { rect, depth, globalImageSize, path: nodePath } = ctx;
    const halfW = Math.floor(rect.width / 2);
    const halfH = Math.floor(rect.height / 2);
    
    // Overlap for boundary objects (10%)
    const overlapX = Math.floor(halfW * 0.1);
    const overlapY = Math.floor(halfH * 0.1);

    const quadrants = [
      { r: 0, c: 0, l: 0, t: 0 },
      { r: 0, c: 1, l: halfW - overlapX, t: 0 },
      { r: 1, c: 0, l: 0, t: halfH - overlapY },
      { r: 1, c: 1, l: halfW - overlapX, t: halfH - overlapY }
    ];

    const promises = quadrants.map(async (q, i) => {
      // Calculate child rect in global pixels
      // Be careful with clamping to parent rect
      const childL = rect.left + q.l;
      const childT = rect.top + q.t;
      
      // Width is half + overlap, but clamped to parent bounds
      let childW = halfW + overlapX;
      let childH = halfH + overlapY;
      
      // Fix right/bottom edge overflow
      if (childL + childW > rect.left + rect.width) childW = (rect.left + rect.width) - childL;
      if (childT + childH > rect.top + rect.height) childH = (rect.top + rect.height) - childT;

      const childCtx: NodeContext = {
        depth: depth + 1,
        rect: { left: childL, top: childT, width: childW, height: childH },
        globalImageSize,
        parentDescription: currentDescription, // Pass down the description of THIS node
        path: `${nodePath}.${i}`
      };

      return this.processNode(childCtx);
    });

    await Promise.all(promises);
  }

  // AI: Discover Kinds
  async discover(imageBuffer: Buffer, parentContext: string) {
    const prompt = `
You are analyzing a specific crop of a larger artwork.
PARENT CONTEXT (What contains this crop): "${parentContext}"

Task: Analyze THIS CROP. Identify ALL distinct object types and decide the appropriate action for each.

ACTION SELECTION GUIDE:

1. detect_individual - Use when you can ACCURATELY draw boxes around EVERY instance:
   - Objects are large/medium size AND clearly separated
   - Count is 'few' (2-5) or '1'
   - You are CONFIDENT you can box ALL of them accurately at this resolution

2. detect_representative - Use for many similar objects where exhaustive detection isn't needed:
   - Background elements (trees in a forest, leaves, clouds)
   - Secondary importance objects where 3-8 examples suffice
   - NOT for narratively important objects like people, animals, figures

3. detect_region - Use for masses/areas, not individual items:
   - Bodies of water, sky areas, forest masses
   - Crowds too dense to separate even when zoomed
   - Atmospheric effects (fog, clouds as a mass)

4. zoom_in - USE THIS LIBERALLY for important small objects:
   - ANY time count is 'many' or 'crowd' AND objects are small/tiny but significant
   - People, figures, animals that are too small to box accurately
   - Dense narrative scenes with multiple subjects
   - Weapons, tools, or other important small details
   - When you're UNCERTAIN if you can accurately box everything
   - PREFER zoom_in over detect_representative for narratively important objects

5. ignore - Only for:
   - Pure texture (brushstrokes, canvas texture)
   - Empty space, sky gradients
   - Artifacts or noise

CRITICAL RULES:
- For DENSE paintings with many figures/animals: PREFER zoom_in over detect_representative
- If objects are 'tiny' or 'small' AND count is 'many' or 'crowd': USE zoom_in
- People, animals, figures are ALWAYS narratively important - use zoom_in if small/numerous
- It's better to zoom_in and detect accurately than to miss objects with detect_representative
- Don't be conservative with zoom_in - thoroughness is more important than speed
`;

    return this.pool.generateObject({
      schema: RecursiveKindSchema,
      messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
    });
  }

  // AI: Detect Instances
  async detect(imageBuffer: Buffer, kinds: z.infer<typeof RecursiveKindSchema>['kinds']): Promise<DetectedObject[]> {
    const labels = kinds.map(k => k.label).join(', ');
    const prompt = `
Task: Detect ALL instances of these specific kinds: ${labels}.
Return bounding boxes [xmin, ymin, xmax, ymax] (0-1000).
`;

    const result = await this.pool.generateObject({
      schema: DetectionSchema,
      messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
    });

    // Map back to specific types from the request
    return result.objects.map(obj => {
      const match = kinds.find(k => k.label.toLowerCase() === obj.label.toLowerCase()) || kinds[0];
      return {
        label: obj.label,
        type: match.type,
        box_2d: normalizeBox(obj.box_2d),
      };
    });
  }

  // AI: Verify Instances
  async verifyInstances(
    visualContextBuffer: Buffer,
    kind: { label: string; type: string },
    instanceCount: number
  ): Promise<z.infer<typeof VerifyResponseSchema>> {
    const prompt = `
You are verifying bounding box annotations for: "${kind.label}" (${kind.type})

The image shows ${instanceCount} numbered box(es), labeled 0 to ${instanceCount - 1}.

CRITICAL: Be skeptical. Some boxes may be on objects that are NOT "${kind.label}" at all.

Your tasks:

1. WRONG BOXES (wrong_indices) — CHECK THIS FIRST:
   Remove boxes where:
   - There is NO "${kind.label}" in or near the box (hallucinated detection)
   - The box is on a DIFFERENT object type (misidentification)
   - The shape/texture was mistaken for "${kind.label}" but isn't one
   Be honest: if it's not clearly a "${kind.label}", remove it.

2. CORRECTIONS (corrections):
   For boxes that ARE on a real "${kind.label}" but are misaligned:
   - Provide the index and corrected [xmin, ymin, xmax, ymax] in 0-1000 coords
   - Only correct boxes that truly contain a "${kind.label}"

3. MISSING (missing):
   Find any CLEARLY VISIBLE "${kind.label}" instances that have NO box yet:
   - Provide box coordinates [xmin, ymin, xmax, ymax] for each
   - Be conservative — only add instances you're certain about
   - Do NOT hallucinate from textures, patterns, shadows, or similar shapes

4. COMPLETE (complete):
   Set true only if ALL visible "${kind.label}" instances are now correctly boxed.

Output JSON only with these exact fields.
`;

    return this.pool.generateObject({
      schema: VerifyResponseSchema,
      temperature: 0.1,
      messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: visualContextBuffer }] }],
    });
  }

  // Detect with verification loop
  async detectWithVerification(
    imageBuffer: Buffer,
    kinds: z.infer<typeof RecursiveKindSchema>['kinds'],
    logPrefix: string
  ): Promise<DetectedObject[]> {
    // Get actual image dimensions (may have been resized)
    const imgMetadata = await sharp(imageBuffer).metadata();
    const imageWidth = imgMetadata.width!;
    const imageHeight = imgMetadata.height!;

    // Initial detection
    let instances: DetectedObject[];
    try {
      instances = await this.detect(imageBuffer, kinds);
      console.log(`${logPrefix}Found ${instances.length} instance(s)`);
    } catch (error) {
      logErrorDetails(`${logPrefix}⚠️ Detection failed. `, error);
      return [];
    }

    if (this.config.verifyRounds <= 0 || instances.length === 0) {
      return instances;
    }

    // Process each kind separately for verification
    const allVerified: DetectedObject[] = [];
    const kindGroups = new Map<string, DetectedObject[]>();

    for (const inst of instances) {
      const key = `${inst.label}|${inst.type}`;
      if (!kindGroups.has(key)) {
        kindGroups.set(key, []);
      }
      kindGroups.get(key)!.push(inst);
    }

    for (const [key, kindInstances] of kindGroups) {
      const [label, type] = key.split('|');
      let verified = kindInstances;

      for (let round = 0; round < this.config.verifyRounds; round++) {
        if (verified.length === 0) break;

        console.log(`${logPrefix}🔍 Verify ${label} (${round + 1}/${this.config.verifyRounds})...`);

        const visualCtx = await buildVerificationImage({
          imageBuffer,
          imageWidth,
          imageHeight,
          instances: verified,
          context: { kind: label, type },
        });

        let verification: z.infer<typeof VerifyResponseSchema>;
        try {
          verification = await this.verifyInstances(visualCtx, { label, type }, verified.length);
        } catch (error) {
          logErrorDetails(`${logPrefix}⚠️ Verification failed. `, error);
          break;
        }

        let changed = false;

        // Remove wrong boxes
        const wrongSet = new Set(verification.wrong_indices.filter((i) => i >= 0 && i < verified.length));
        if (wrongSet.size > 0) {
          console.log(`${logPrefix}  ❌ Remove ${wrongSet.size} box(es)`);
          verified = verified.filter((_, i) => !wrongSet.has(i));
          changed = true;
        }

        // Apply corrections
        if (verification.corrections.length > 0) {
          let correctionCount = 0;
          for (const corr of verification.corrections) {
            if (wrongSet.has(corr.index)) continue;
            const removedBelow = [...wrongSet].filter((w) => w < corr.index).length;
            const adjustedIndex = corr.index - removedBelow;
            if (adjustedIndex >= 0 && adjustedIndex < verified.length) {
              verified[adjustedIndex] = { ...verified[adjustedIndex], box_2d: normalizeBox(corr.box_2d) };
              correctionCount++;
            }
          }
          if (correctionCount > 0) {
            console.log(`${logPrefix}  ✏️  Correct ${correctionCount} box(es)`);
            changed = true;
          }
        }

        // Add missing
        if (verification.missing.length > 0) {
          console.log(`${logPrefix}  ➕ Add ${verification.missing.length} missing`);
          for (const m of verification.missing) {
            verified.push({ label, type, box_2d: normalizeBox(m.box_2d) });
          }
          changed = true;
        }

        if (!changed) {
          console.log(`${logPrefix}  ✅ Stable`);
          break;
        }

        if (verification.complete) {
          console.log(`${logPrefix}  ✅ Complete`);
          break;
        }
      }

      allVerified.push(...verified);
    }

    return allVerified;
  }

  // Detect representative examples (3-8 spatially diverse instances)
  async detectRepresentative(
    imageBuffer: Buffer,
    kinds: z.infer<typeof RecursiveKindSchema>['kinds'],
    logPrefix: string
  ): Promise<DetectedObject[]> {
    const labels = kinds.map(k => k.label).join(', ');
    const prompt = `
Task: Detect 3-8 REPRESENTATIVE examples of these kinds: ${labels}.
Choose examples that are:
- Spatially diverse (spread across different parts of the image)
- Clearly visible and well-defined
- Representative of the variety present

Do NOT try to detect every instance. Just pick good examples spread across the image.
Return bounding boxes [xmin, ymin, xmax, ymax] (0-1000).
`;

    try {
      const result = await this.pool.generateObject({
        schema: DetectionSchema,
        messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
      });

      let objects = result.objects.map(obj => {
        const match = kinds.find(k => k.label.toLowerCase() === obj.label.toLowerCase()) || kinds[0];
        return {
          label: obj.label,
          type: match.type,
          box_2d: normalizeBox(obj.box_2d),
        };
      });

      // Cap at 8 per kind
      const byKind = new Map<string, DetectedObject[]>();
      for (const obj of objects) {
        const key = obj.label.toLowerCase();
        if (!byKind.has(key)) byKind.set(key, []);
        byKind.get(key)!.push(obj);
      }

      const capped: DetectedObject[] = [];
      for (const [, kindObjects] of byKind) {
        capped.push(...kindObjects.slice(0, 8));
      }

      console.log(`${logPrefix}Found ${capped.length} representative instance(s)`);
      return capped;
    } catch (error) {
      logErrorDetails(`${logPrefix}⚠️ Representative detection failed. `, error);
      return [];
    }
  }

  // Detect regions/areas rather than individual instances
  async detectRegions(
    imageBuffer: Buffer,
    kinds: z.infer<typeof RecursiveKindSchema>['kinds'],
    logPrefix: string
  ): Promise<DetectedObject[]> {
    const labels = kinds.map(k => `"${k.label}" (${k.type})`).join(', ');
    const prompt = `
Task: Detect REGIONS/AREAS where these kinds appear: ${labels}.

Draw bounding boxes around AREAS/MASSES, not individual items:
- For a forest: one box around the forested area, not each tree
- For clouds: boxes around cloud masses, not each cloud
- For a crowd: one box around the crowd area, not each person
- For grass/fields: boxes around grassy areas

Typically 1-5 region boxes per kind maximum.
Boxes can be large and encompassing.

Return bounding boxes [xmin, ymin, xmax, ymax] (0-1000).
`;

    try {
      const result = await this.pool.generateObject({
        schema: DetectionSchema,
        messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
      });

      const objects = result.objects.map(obj => {
        const match = kinds.find(k => k.label.toLowerCase() === obj.label.toLowerCase()) || kinds[0];
        return {
          label: obj.label,
          type: match.type,
          box_2d: normalizeBox(obj.box_2d),
        };
      });

      console.log(`${logPrefix}Found ${objects.length} region(s)`);
      return objects;
    } catch (error) {
      logErrorDetails(`${logPrefix}⚠️ Region detection failed. `, error);
      return [];
    }
  }

  // Post-recursion reconciliation: filter hallucinations using full-image context
  // NOTE: This is DISABLED by default because:
  // 1. Objects found at deep levels (depth 2-3) are too small to verify against full image
  // 2. The verification loop already catches hallucinations at the tile level
  // 3. Full-image reconciliation is overly aggressive and removes valid detections
  async reconcileResults(objects: DetectionResult[]): Promise<DetectionResult[]> {
    if (objects.length === 0) return objects;

    // Skip reconciliation - verification during detection is sufficient
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`POST-PROCESSING: Reconciliation (Skipped)`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`   Skipping full-image reconciliation (${objects.length} objects)`);
    console.log(`   Reason: Per-tile verification is sufficient; full-image reconciliation`);
    console.log(`   cannot accurately validate tiny objects found at deeper recursion levels.`);
    return objects;

    // Original reconciliation code below (kept for reference/future use)
    /*
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`POST-PROCESSING: Reconciliation (Hallucination Filter)`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);

    // Build list of all detected objects for the prompt
    const objectList = objects.map((obj, i) => {
      const [xmin, ymin, xmax, ymax] = obj.box_2d;
      const cx = (xmin + xmax) / 2;
      const cy = (ymin + ymax) / 2;
      const hPos = cx < 333 ? 'left' : cx > 666 ? 'right' : 'center';
      const vPos = cy < 333 ? 'top' : cy > 666 ? 'bottom' : 'middle';
      return `[${i}] ${obj.label} (${obj.type}) at ${hPos}-${vPos}, depth=${obj.depth}`;
    }).join('\n');

    const prompt = `
You are reviewing object detections from a multi-scale analysis of an artwork.
The detections were made at different zoom levels (depth). Now you see the FULL image.

DETECTED OBJECTS (from recursive analysis):
${objectList}

YOUR TASK:
Look at the FULL IMAGE and determine which detections are REAL vs ARTIFACTS.

REJECT objects that:
- Don't actually exist in the image (hallucinations)
- Were mistaken from textures, patterns, or brush strokes
- Are duplicates of the same object under different names
- Were misidentified (e.g., a shadow called "figure")

CONFIRM objects that:
- Are clearly visible in the full image
- Match their labeled description
- Represent distinct, real entities in the artwork

Output confirmed_indices and rejected_indices as arrays of the object index numbers.
`;

    try {
      const result = await this.pool.generateObject({
        schema: ReconciliationResultSchema,
        messages: [
          { role: 'user', content: [
            { type: 'text', text: prompt },
            { type: 'image', image: this.fullImageBuffer! }
          ]}
        ],
      });

      const confirmedSet = new Set(result.confirmed_indices);
      const rejectedSet = new Set(result.rejected_indices);

      // If an index is in neither, default to confirmed
      const filtered = objects.filter((_, i) => {
        if (rejectedSet.has(i)) return false;
        return true; // Keep if confirmed or not mentioned
      });

      const rejectedCount = objects.length - filtered.length;
      console.log(`   Reconciliation: ${objects.length} → ${filtered.length} (rejected ${rejectedCount} artifacts)`);
      if (result.notes) {
        console.log(`   Notes: ${result.notes}`);
      }

      return filtered;
    } catch (error) {
      logErrorDetails('   ⚠️ Reconciliation failed, keeping all objects. ', error);
      return objects;
    }
    */
  }

  // Helper: Resize image for AI if too large
  async optimizeImageForAI(buffer: Buffer): Promise<Buffer> {
    const metadata = await sharp(buffer).metadata();
    // Resize if larger than 2048px on any side
    if ((metadata.width && metadata.width > 2048) || (metadata.height && metadata.height > 2048)) {
      return sharp(buffer)
        .resize({ width: 2048, height: 2048, fit: 'inside' })
        .jpeg({ quality: 80 })
        .toBuffer();
    }
    return buffer;
  }

  // Helper: Local (0-1000) -> Global (0-1000) using shared utility
  localToGlobalBox(localBox: number[], tileRect: NodeContext['rect'], globalSize: { width: number, height: number }): Box2D {
    return mapTileBoxToFullImage(
      normalizeBox(localBox),
      tileRect,
      globalSize.width,
      globalSize.height
    );
  }

  // Generate accessibility descriptions using detected objects as ground truth
  async generateDescriptions(objects: DetectionResult[]): Promise<{ alt_text: string; long_description: string }> {
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`OUTPUT: Generating Descriptions`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);

    // Build object summary for the prompt
    const objectSummary = objects
      .sort((a, b) => (a.importance_rank ?? 999) - (b.importance_rank ?? 999))
      .slice(0, 30)
      .map(o => `${o.label} (${o.type})`)
      .join(', ');

    const prompt = `
You are generating accessibility descriptions for an artwork.

VERIFIED OBJECTS IN THE IMAGE (sorted by importance):
${objectSummary}

RULES:
1. ONLY describe objects from the verified list above. Do NOT add objects not in the list.
2. Alt text: 10-20 words, describes what a user needs to know at a glance
3. Long description: 150-250 words, comprehensive description for screen readers
4. Focus on spatial relationships, composition, and narrative
5. Do NOT hallucinate or add objects not verified in the list
`;

    try {
      const result = await this.pool.generateObject({
        schema: DescriptionSchema,
        messages: [
          { role: 'user', content: [
            { type: 'text', text: prompt },
            { type: 'image', image: this.fullImageBuffer! }
          ]}
        ],
      });

      console.log(`   ✅ Generated alt text (${result.alt_text.split(' ').length} words)`);
      console.log(`   ✅ Generated long description (${result.long_description.split(' ').length} words)`);

      return result;
    } catch (error) {
      logErrorDetails('   ⚠️ Description generation failed. ', error);
      return { alt_text: '', long_description: '' };
    }
  }

  // Generate cutouts for detected objects
  async generateCutouts(
    objects: DetectionResult[],
    imageWidth: number,
    imageHeight: number
  ): Promise<{ count: number; entries: any[] }> {
    if (!this.config.cutouts || objects.length === 0) {
      return { count: 0, entries: [] };
    }

    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`OUTPUT: Generating Cutouts`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);

    const outputDir = path.dirname(this.config.outputFile);
    const cutoutsDir = path.join(outputDir, 'cutouts');
    if (!fs.existsSync(cutoutsDir)) {
      fs.mkdirSync(cutoutsDir, { recursive: true });
    }

    const padding = this.config.cutoutsPadding;
    const format = this.config.cutoutsFormat;
    const thumbSize = this.config.cutoutsThumbSize;

    // Limit to top N objects
    const toProcess = objects
      .sort((a, b) => (a.importance_rank ?? 999) - (b.importance_rank ?? 999))
      .slice(0, this.config.cutoutsMax);

    const entries: any[] = [];

    await runWithConcurrency(toProcess, this.config.cutoutsConcurrency, async (obj, i) => {
      const [xmin, ymin, xmax, ymax] = obj.box_2d;

      // Convert normalized coords to pixels
      const pxXmin = (xmin / 1000) * imageWidth;
      const pxYmin = (ymin / 1000) * imageHeight;
      const pxXmax = (xmax / 1000) * imageWidth;
      const pxYmax = (ymax / 1000) * imageHeight;

      const boxW = pxXmax - pxXmin;
      const boxH = pxYmax - pxYmin;
      const padX = boxW * padding;
      const padY = boxH * padding;

      const left = Math.max(0, Math.round(pxXmin - padX));
      const top = Math.max(0, Math.round(pxYmin - padY));
      const right = Math.min(imageWidth, Math.round(pxXmax + padX));
      const bottom = Math.min(imageHeight, Math.round(pxYmax + padY));
      const width = right - left;
      const height = bottom - top;

      if (width < 10 || height < 10) return;

      const labelPart = sanitizeFilePart(obj.label);
      const filename = `${String(i).padStart(3, '0')}-${labelPart}.${format}`;
      const cutoutPath = path.join(cutoutsDir, filename);

      try {
        const cutoutPipeline = sharp(this.fullImageBuffer!)
          .extract({ left, top, width, height });

        if (format === 'webp') {
          await cutoutPipeline.webp({ quality: 85 }).toFile(cutoutPath);
        } else {
          await cutoutPipeline.png().toFile(cutoutPath);
        }

        // Generate thumbnail
        let thumbPath: string | null = null;
        if (thumbSize > 0) {
          const thumbFilename = `${String(i).padStart(3, '0')}-${labelPart}.thumb.${format}`;
          thumbPath = path.join(cutoutsDir, thumbFilename);
          const thumbPipeline = sharp(cutoutPath)
            .resize({ height: thumbSize, fit: 'inside' });

          if (format === 'webp') {
            await thumbPipeline.webp({ quality: 75 }).toFile(thumbPath);
          } else {
            await thumbPipeline.png().toFile(thumbPath);
          }
        }

        entries.push({
          index: i,
          label: obj.label,
          type: obj.type,
          cutout_path: `cutouts/${filename}`,
          thumb_path: thumbPath ? `cutouts/${path.basename(thumbPath)}` : null,
        });
      } catch (error) {
        logErrorDetails(`   ⚠️ Cutout ${i} failed. `, error);
      }
    });

    console.log(`   ✅ Generated ${entries.length} cutouts`);

    // Write cutouts index
    const cutoutsIndexPath = path.join(outputDir, 'cutouts.json');
    const cutoutsIndex = { count: entries.length, entries };
    fs.writeFileSync(cutoutsIndexPath, JSON.stringify(cutoutsIndex, null, 2));

    return cutoutsIndex;
  }

  async run() {
    await this.init();
    if (!this.fullImageBuffer) throw new Error("No image loaded");

    const metadata = await sharp(this.fullImageBuffer).metadata();
    const width = metadata.width!;
    const height = metadata.height!;

    const rootCtx: NodeContext = {
      depth: 0,
      rect: { left: 0, top: 0, width, height },
      globalImageSize: { width, height },
      parentDescription: "The full artwork.",
      path: "root"
    };

    console.log(`🚀 Starting Recursive Detection (Max Depth: ${this.config.maxDepth}, Verify Rounds: ${this.config.verifyRounds})`);
    await this.processNode(rootCtx);

    // Phase: Reconciliation (hallucination filter)
    this.results = await this.reconcileResults(this.results);

    // Phase: Deduplication
    console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
    console.log(`POST-PROCESSING: Deduplication & Scoring`);
    console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);

    const beforeDedupe = this.results.length;
    this.results = dedupeObjectsByGeometry(this.results) as DetectionResult[];
    console.log(`   Deduplication: ${beforeDedupe} → ${this.results.length} (removed ${beforeDedupe - this.results.length} duplicates)`);

    // Phase: Importance scoring
    this.results = computeImportanceGeom(this.results) as DetectionResult[];
    console.log(`   Importance scores computed for ${this.results.length} objects`);

    console.log(`\n✅ Detection complete. Found ${this.results.length} objects.`);

    // Phase: Output Generation
    const outputMetadata = await sharp(this.fullImageBuffer!).metadata();
    const imageWidth = outputMetadata.width!;
    const imageHeight = outputMetadata.height!;

    // Generate descriptions if enabled
    let descriptions: { alt_text: string; long_description: string } | undefined;
    if (this.config.generateDescriptions) {
      descriptions = await this.generateDescriptions(this.results);
    }

    // Generate cutouts if enabled
    let cutoutsInfo: { count: number; entries: any[] } | undefined;
    if (this.config.cutouts) {
      cutoutsInfo = await this.generateCutouts(this.results, imageWidth, imageHeight);
    }

    // Generate annotated image if enabled
    if (this.config.annotate && this.config.annotatedOutput) {
      console.log(`\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);
      console.log(`OUTPUT: Generating Annotated Image`);
      console.log(`━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━`);

      try {
        await annotateImage({
          imagePath: this.config.imagePath,
          objects: this.results,
          outputPath: this.config.annotatedOutput,
        });
        console.log(`   ✅ Annotated image saved to ${this.config.annotatedOutput}`);
      } catch (error) {
        logErrorDetails('   ⚠️ Annotation failed. ', error);
      }
    }

    // Print Usage Stats
    if (this.pool.getUsageStats) {
      const stats = this.pool.getUsageStats();
      console.log(`\n📊 Usage Stats:`);
      console.log(`   - API Calls: ${stats.totalCalls}`);
      console.log(`   - Prompt Tokens: ${stats.totalPromptTokens.toLocaleString()}`);
      console.log(`   - Completion Tokens: ${stats.totalCompletionTokens.toLocaleString()}`);
      console.log(`   - Estimated Cost: $${stats.estimatedCost.toFixed(4)}`);
    }

    // Build output payload
    const outputPayload: any = {
      strategy: 'recursive-v2',
      image_path: this.config.imagePath,
      model_name: this.config.modelName,
      max_depth: this.config.maxDepth,
      min_tile_size: this.config.minTileSize,
      verify_rounds: this.config.verifyRounds,
      objects: this.results.map(r => ({
        label: r.label,
        type: r.type,
        box_2d: r.box_2d,
        importance: r.importance,
        importance_geom: r.importance_geom,
        importance_rank: r.importance_rank,
        aliases: r.aliases,
        depth: r.depth,
      })),
      generated_at: new Date().toISOString(),
    };

    // Add optional fields
    if (descriptions && (descriptions.alt_text || descriptions.long_description)) {
      outputPayload.descriptions = descriptions;
    }

    if (cutoutsInfo) {
      outputPayload.cutouts = {
        enabled: true,
        count: cutoutsInfo.count,
        format: this.config.cutoutsFormat,
      };
    }

    // Save results
    fs.writeFileSync(this.config.outputFile, JSON.stringify(outputPayload, null, 2));
    console.log(`\n📂 Output written to ${this.config.outputFile}`);

    return outputPayload;
  }
}
