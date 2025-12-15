/**
 * Core detection logic for artwork object detection
 */
import { z } from 'zod';
import fs from 'fs';
import path from 'path';
import sharp from 'sharp';
import { annotateImage } from './annotator';
import { createAIPool, AIPool } from './pool';

// ==========================================
// SCHEMAS
// ==========================================
const BoundingBoxSchema = z.object({
  label: z.string().min(1),
  type: z.string().min(1),
  box_2d: z.array(z.number().int().min(0).max(1000)).min(4).max(4),
});

const CountEstimate = z.enum(['few', 'moderate', 'many', 'very_many']);
type CountEstimateType = z.infer<typeof CountEstimate>;

const SizeEstimate = z.enum(['tiny', 'small', 'medium', 'large', 'giant']);
type SizeEstimateType = z.infer<typeof SizeEstimate>;

const SegmentationStrategy = z.enum(['individual', 'representative', 'region']);
type SegmentationStrategyType = z.infer<typeof SegmentationStrategy>;

const ArtisticImportance = z.enum(['primary', 'secondary', 'background']);
type ArtisticImportanceType = z.infer<typeof ArtisticImportance>;

const KindSchema = z.object({
  kinds: z.array(
    z.object({
      kind: z.string().min(1),
      type: z.string().min(1).describe('Category for this kind'),
      estimated_count: CountEstimate,
      estimated_size: SizeEstimate.describe('Typical size of each instance relative to image: tiny (<2%), small (2-10%), medium (10-30%), large (30-60%), giant (>60%)'),
      segmentation: SegmentationStrategy.describe('How to detect: individual (each instance), representative (a few examples), or region (bounding area)'),
      importance: ArtisticImportance.describe('Artistic importance: primary (main subject), secondary (supporting), background (contextual)'),
    })
  ),
});

const InstancesSchema = z.object({
  objects: z.array(BoundingBoxSchema),
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

const RegionCountSchema = z.object({
  estimated_count: CountEstimate,
});

// Quadrant names for spatial tracking
const QuadrantName = z.enum(['top-left', 'top-right', 'bottom-left', 'bottom-right']);
type QuadrantNameType = z.infer<typeof QuadrantName>;

const ReconciliationSchema = z.object({
  kinds: z.array(
    z.object({
      kind: z.string().min(1),
      type: z.string().min(1),
      is_real: z.boolean().describe('True if this object type actually exists in the image, false if it was a misidentification'),
      quadrants: z.array(QuadrantName).describe('Which quadrants contain instances of this kind (empty if is_real=false)'),
      estimated_count: CountEstimate,
      estimated_size: SizeEstimate,
      segmentation: SegmentationStrategy,
      importance: ArtisticImportance,
      detection_scale: z.enum(['full', 'quadrant']).describe('Whether to detect on full image or per-quadrant'),
    })
  ),
});

type ReconciledKind = z.infer<typeof ReconciliationSchema>['kinds'][number];

// ==========================================
// TYPES
// ==========================================
export type DetectedObject = z.infer<typeof BoundingBoxSchema>;
export type Kind = z.infer<typeof KindSchema>['kinds'][number];

export type StoredObject = DetectedObject & {
  importance?: number;
  importance_geom?: number;
  importance_rank?: number | null;
  aliases?: string[];
};
export type DescriptionPayload = {
  alt_text: string;
  long_description: string;
};

export type DetectionConfig = {
  imagePath: string;
  outputFile: string;
  annotatedOutput: string;
  maxKinds: number;
  verifyRounds: number;
  tileThreshold: number;
  concurrency: number;
  multiScaleDiscovery: boolean;
  cutouts: boolean;
  cutoutsFormat: 'webp' | 'png';
  cutoutsThumbSize: number;
  cutoutsMax: number;
  cutoutsConcurrency: number;
  cutoutsPadding: number;
  modelName: string;
  descriptionModelName: string;
  annotate: boolean;
  mock: boolean;
  onlyKinds: string[] | null;
};

export type OutputPayload = {
  strategy: 'hybrid-detect-verify';
  image_path: string;
  model_name: string;
  description_model_name: string;
  max_kinds: number;
  verify_rounds: number;
  tile_threshold: number;
  cutouts?: {
    enabled: boolean;
    format: 'webp' | 'png';
    thumb_size: number;
    max: number;
    concurrency: number;
    index_path: string;
    directory_path: string;
    count: number;
  };
  kinds: Kind[];
  objects: StoredObject[];
  descriptions: DescriptionPayload | null;
  generated_at: string;
};

type Box2D = [number, number, number, number];
type TileDefinition = {
  row: number;
  col: number;
  left: number;
  top: number;
  width: number;
  height: number;
  label: string;
  depth?: number;
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

function normalizeBox(box: number[]): Box2D {
  const x1 = clamp(Math.round(box[0] ?? 0), 0, 1000);
  const y1 = clamp(Math.round(box[1] ?? 0), 0, 1000);
  const x2 = clamp(Math.round(box[2] ?? 0), 0, 1000);
  const y2 = clamp(Math.round(box[3] ?? 0), 0, 1000);
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

function normalizeType(value: string) {
  return (value || 'other').trim().toLowerCase();
}

function getBoxArea(box: Box2D) {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

function intersectionArea(a: Box2D, b: Box2D) {
  const interX1 = Math.max(a[0], b[0]);
  const interY1 = Math.max(a[1], b[1]);
  const interX2 = Math.min(a[2], b[2]);
  const interY2 = Math.min(a[3], b[3]);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  return interW * interH;
}

function boxSimilarity(a: Box2D, b: Box2D) {
  const areaA = getBoxArea(a);
  const areaB = getBoxArea(b);
  const inter = intersectionArea(a, b);
  const union = areaA + areaB - inter;
  const iouScore = union > 0 ? inter / union : 0;
  const minArea = Math.max(1, Math.min(areaA, areaB));
  const coverMin = inter / minArea;
  const areaRatio = Math.min(areaA, areaB) / Math.max(1, Math.max(areaA, areaB));
  return { iou: iouScore, coverMin, areaRatio };
}

function mergeAliases(target: StoredObject, incoming: StoredObject) {
  const set = new Set<string>();
  if (target.aliases) for (const a of target.aliases) set.add(a);
  if (incoming.aliases) for (const a of incoming.aliases) set.add(a);
  if (incoming.label && incoming.label !== target.label) set.add(incoming.label);

  const aliases = Array.from(set.values()).sort();
  return aliases.length ? { ...target, aliases } : target;
}

function dedupeObjectsByGeometry(objects: StoredObject[]): StoredObject[] {
  const kept: StoredObject[] = [];
  for (const obj of objects) {
    const boxA = normalizeBox(obj.box_2d);
    const typeA = normalizeType(obj.type);

    let matchIndex = -1;
    for (let i = 0; i < kept.length; i++) {
      const other = kept[i];
      const boxB = normalizeBox(other.box_2d);
      const typeB = normalizeType(other.type);
      const { iou: iouScore, coverMin, areaRatio } = boxSimilarity(boxA, boxB);

      const sameType = typeA === typeB;
      const strictTypeMismatch = !sameType && typeA !== 'other' && typeB !== 'other';

      const isSameInstance =
        (iouScore >= 0.88 && areaRatio >= 0.65) ||
        (coverMin >= 0.94 && areaRatio >= 0.72);

      if (strictTypeMismatch) {
        if (isSameInstance && iouScore >= 0.92 && areaRatio >= 0.8) {
          matchIndex = i;
          break;
        }
        continue;
      }

      if (isSameInstance) {
        matchIndex = i;
        break;
      }
    }

    if (matchIndex === -1) {
      kept.push(obj);
    } else {
      kept[matchIndex] = mergeAliases(kept[matchIndex], obj);
    }
  }
  return kept;
}

function getLabelFamily(label: string) {
  return label
    .trim()
    .toLowerCase()
    .replace(/\s+#\d+$/g, '')
    .replace(/\s+\d+$/g, '')
    .trim();
}

function computeImportanceGeom(objects: StoredObject[]): StoredObject[] {
  const total = objects.length || 1;
  const typeCounts = new Map<string, number>();
  const familyCounts = new Map<string, number>();

  for (const obj of objects) {
    const typeKey = normalizeType(obj.type || 'other');
    typeCounts.set(typeKey, (typeCounts.get(typeKey) || 0) + 1);
    const familyKey = getLabelFamily(obj.label || '');
    familyCounts.set(familyKey, (familyCounts.get(familyKey) || 0) + 1);
  }

  const maxLog = Math.log(total + 1);

  const scores = objects.map((obj) => {
    const box = normalizeBox(obj.box_2d);
    const w = Math.max(0, box[2] - box[0]);
    const h = Math.max(0, box[3] - box[1]);
    const area = (w * h) / (1000 * 1000);
    const areaScore = Math.sqrt(clamp(area, 0, 1));

    const cx = (box[0] + box[2]) / 2;
    const cy = (box[1] + box[3]) / 2;
    const dx = cx - 500;
    const dy = cy - 500;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const distMax = Math.sqrt(2 * 500 * 500);
    const centrality = clamp(1 - dist / distMax, 0, 1);
    const foreground = clamp(cy / 1000, 0, 1);

    const typeKey = normalizeType(obj.type || 'other');
    const familyKey = getLabelFamily(obj.label || '');
    const typeCount = typeCounts.get(typeKey) || total;
    const familyCount = familyCounts.get(familyKey) || total;
    const rarityType = clamp(Math.log((total + 1) / (typeCount + 1)) / maxLog, 0, 1);
    const rarityFamily = clamp(
      Math.log((total + 1) / (familyCount + 1)) / maxLog,
      0,
      1
    );

    const score =
      0.3 * areaScore +
      0.25 * centrality +
      0.15 * foreground +
      0.2 * rarityFamily +
      0.1 * rarityType;

    return clamp(score, 0, 1);
  });

  const ranking = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score)
    .map((x, rank) => ({ ...x, rank: rank + 1 }));

  const rankByIndex = new Map<number, number>();
  for (const r of ranking) rankByIndex.set(r.index, r.rank);

  return objects.map((obj, index) => ({
    ...obj,
    importance_geom: Number(scores[index].toFixed(4)),
    importance: Number(scores[index].toFixed(4)),
    importance_rank: rankByIndex.get(index) || null,
  }));
}

function ensureDirSync(dirPath: string) {
  fs.mkdirSync(dirPath, { recursive: true });
}

// Generate quadrant definitions with safe bounds (55% size, 10% overlap)
function getQuadrantDefs(imageWidth: number, imageHeight: number): Array<{ name: QuadrantNameType; left: number; top: number; width: number; height: number }> {
  const halfW = Math.floor(imageWidth * 0.55);
  const halfH = Math.floor(imageHeight * 0.55);
  const offsetX = Math.floor(imageWidth * 0.45);
  const offsetY = Math.floor(imageHeight * 0.45);

  return [
    { name: 'top-left', left: 0, top: 0, width: halfW, height: halfH },
    { name: 'top-right', left: offsetX, top: 0, width: imageWidth - offsetX, height: halfH },
    { name: 'bottom-left', left: 0, top: offsetY, width: halfW, height: imageHeight - offsetY },
    { name: 'bottom-right', left: offsetX, top: offsetY, width: imageWidth - offsetX, height: imageHeight - offsetY },
  ];
}

function sanitizeFilePart(value: string) {
  const trimmed = value.trim().toLowerCase();
  return trimmed
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 80);
}

async function runWithConcurrency<T, R>(
  items: T[],
  concurrency: number,
  fn: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let nextIndex = 0;

  async function worker() {
    while (true) {
      const i = nextIndex;
      nextIndex += 1;
      if (i >= items.length) return;
      results[i] = await fn(items[i], i);
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, items.length) }, () =>
    worker()
  );
  await Promise.all(workers);
  return results;
}

// ==========================================
// TILING
// ==========================================

function mapTileBoxToFullImage(
  tileBox: Box2D,
  tile: TileDefinition,
  fullImageWidth: number,
  fullImageHeight: number
): Box2D {
  const [txmin, tymin, txmax, tymax] = tileBox;
  const pxXmin = (txmin / 1000) * tile.width;
  const pxYmin = (tymin / 1000) * tile.height;
  const pxXmax = (txmax / 1000) * tile.width;
  const pxYmax = (tymax / 1000) * tile.height;

  const fullPxXmin = tile.left + pxXmin;
  const fullPxYmin = tile.top + pxYmin;
  const fullPxXmax = tile.left + pxXmax;
  const fullPxYmax = tile.top + pxYmax;

  const normXmin = (fullPxXmin / fullImageWidth) * 1000;
  const normYmin = (fullPxYmin / fullImageHeight) * 1000;
  const normXmax = (fullPxXmax / fullImageWidth) * 1000;
  const normYmax = (fullPxYmax / fullImageHeight) * 1000;

  return [
    Math.round(clamp(normXmin, 0, 1000)),
    Math.round(clamp(normYmin, 0, 1000)),
    Math.round(clamp(normXmax, 0, 1000)),
    Math.round(clamp(normYmax, 0, 1000)),
  ];
}

async function cropTileBuffer(imageBuffer: Buffer, tile: TileDefinition): Promise<Buffer> {
  return sharp(imageBuffer)
    .extract({ left: tile.left, top: tile.top, width: tile.width, height: tile.height })
    .jpeg({ quality: 92 })
    .toBuffer();
}

function isBoxClippedAtTileBoundary(
  tileBox: Box2D,
  tile: TileDefinition,
  fullImageWidth: number,
  fullImageHeight: number,
  edgeThreshold: number = 15
): boolean {
  const [xmin, ymin, xmax, ymax] = tileBox;
  const touchesLeft = xmin <= edgeThreshold && tile.left > 0;
  const touchesTop = ymin <= edgeThreshold && tile.top > 0;
  const touchesRight = xmax >= (1000 - edgeThreshold) && (tile.left + tile.width) < fullImageWidth;
  const touchesBottom = ymax >= (1000 - edgeThreshold) && (tile.top + tile.height) < fullImageHeight;
  return touchesLeft || touchesTop || touchesRight || touchesBottom;
}

// ==========================================
// ADAPTIVE TILING
// ==========================================
const COUNT_LEVELS: Record<CountEstimateType, number> = {
  few: 1,
  moderate: 2,
  many: 3,
  very_many: 4,
};

function isCountAboveThreshold(count: CountEstimateType, threshold: CountEstimateType): boolean {
  return COUNT_LEVELS[count] > COUNT_LEVELS[threshold];
}

const QUADRANT_OVERLAP_PCT = 0.15;

function subdivideIntoQuadrants(
  tile: TileDefinition,
  fullImageWidth: number,
  fullImageHeight: number
): TileDefinition[] {
  const { left, top, width, height, depth = 0 } = tile;
  const halfW = Math.floor(width / 2);
  const halfH = Math.floor(height / 2);
  const overlapX = Math.round(halfW * QUADRANT_OVERLAP_PCT);
  const overlapY = Math.round(halfH * QUADRANT_OVERLAP_PCT);

  const quadrants: Array<{ row: number; col: number; l: number; t: number; r: number; b: number }> = [
    { row: 0, col: 0, l: left, t: top, r: left + halfW + overlapX, b: top + halfH + overlapY },
    { row: 0, col: 1, l: left + halfW - overlapX, t: top, r: left + width, b: top + halfH + overlapY },
    { row: 1, col: 0, l: left, t: top + halfH - overlapY, r: left + halfW + overlapX, b: top + height },
    { row: 1, col: 1, l: left + halfW - overlapX, t: top + halfH - overlapY, r: left + width, b: top + height },
  ];

  return quadrants.map((q) => {
    const clampedLeft = Math.max(0, q.l);
    const clampedTop = Math.max(0, q.t);
    const clampedRight = Math.min(fullImageWidth, q.r);
    const clampedBottom = Math.min(fullImageHeight, q.b);

    return {
      row: q.row,
      col: q.col,
      left: clampedLeft,
      top: clampedTop,
      width: clampedRight - clampedLeft,
      height: clampedBottom - clampedTop,
      label: `${tile.label}.q${q.row}${q.col}`,
      depth: depth + 1,
    };
  });
}

// ==========================================
// CUTOUTS
// ==========================================
type CutoutIndexEntry = {
  index: number;
  label: string;
  type: string;
  box_2d: number[];
  importance_geom?: number;
  importance_rank?: number;
  cutout_path: string;
  thumb_path: string | null;
  crop_px: { left: number; top: number; width: number; height: number };
};

type CutoutIndexPayload = {
  version: 1;
  image_path: string;
  image_width: number;
  image_height: number;
  generated_at: string;
  count: number;
  entries: CutoutIndexEntry[];
};

function getCutoutsPaths(outputFile: string) {
  const parsed = path.parse(path.resolve(outputFile));
  const indexPath = path.join(parsed.dir, 'cutouts.json');
  const directoryPath = path.join(parsed.dir, 'cutouts');
  const fullDir = path.join(directoryPath, 'full');
  const thumbDir = path.join(directoryPath, 'thumb');
  return { indexPath, directoryPath, fullDir, thumbDir };
}

async function generateCutouts(options: {
  imageBuffer: Buffer;
  imagePath: string;
  imageWidth: number;
  imageHeight: number;
  objects: StoredObject[];
  outputFile: string;
  format: 'webp' | 'png';
  thumbSize: number;
  max: number;
  concurrency: number;
  padding: number;
}): Promise<{ indexPath: string; directoryPath: string; count: number }> {
  const { indexPath, directoryPath, fullDir, thumbDir } = getCutoutsPaths(options.outputFile);

  ensureDirSync(fullDir);
  if (options.thumbSize > 0) ensureDirSync(thumbDir);

  const base = sharp(options.imageBuffer);
  const count = Math.min(options.objects.length, options.max);

  console.log(`\n✂️  Writing ${count} cutout(s) to ${path.relative(process.cwd(), directoryPath)}/ ...`);

  const entries: Array<CutoutIndexEntry | null> = new Array(count).fill(null);
  const indices = Array.from({ length: count }, (_, i) => i);

  await runWithConcurrency(indices, options.concurrency, async (i) => {
    const obj = options.objects[i];
    const norm = normalizeObject(obj);
    const [xmin, ymin, xmax, ymax] = normalizeBox(norm.box_2d);

    let left = Math.floor((xmin / 1000) * options.imageWidth);
    let top = Math.floor((ymin / 1000) * options.imageHeight);
    let right = Math.ceil((xmax / 1000) * options.imageWidth);
    let bottom = Math.ceil((ymax / 1000) * options.imageHeight);

    if (options.padding > 0) {
      const baseWidth = right - left;
      const baseHeight = bottom - top;
      const padX = Math.round(baseWidth * options.padding);
      const padY = Math.round(baseHeight * options.padding);
      left -= padX;
      top -= padY;
      right += padX;
      bottom += padY;
    }

    left = clamp(left, 0, options.imageWidth - 2);
    top = clamp(top, 0, options.imageHeight - 2);
    right = clamp(right, left + 2, options.imageWidth);
    bottom = clamp(bottom, top + 2, options.imageHeight);

    const width = right - left;
    const height = bottom - top;
    const area = width * height;
    if (area < 9) return;

    const safeLabel = sanitizeFilePart(norm.label || `object-${i}`);
    const id = String(i).padStart(5, '0');
    const baseName = safeLabel ? `${id}-${safeLabel}` : id;

    const ext = options.format === 'png' ? 'png' : 'webp';
    const cutoutRel = `${path.basename(directoryPath)}/full/${baseName}.${ext}`;
    const thumbRel = options.thumbSize > 0 ? `${path.basename(directoryPath)}/thumb/${baseName}.${ext}` : null;

    const cutoutAbs = path.join(fullDir, `${baseName}.${ext}`);
    const thumbAbs = thumbRel != null ? path.join(thumbDir, `${baseName}.${ext}`) : null;

    const crop = base.clone().extract({ left, top, width, height });
    if (options.format === 'png') {
      await crop.png().toFile(cutoutAbs);
    } else {
      await crop.webp({ quality: 90 }).toFile(cutoutAbs);
    }

    if (thumbAbs && options.thumbSize > 0) {
      const thumb = base
        .clone()
        .extract({ left, top, width, height })
        .resize({ width: options.thumbSize, height: options.thumbSize, fit: 'inside', withoutEnlargement: true });
      if (options.format === 'png') {
        await thumb.png().toFile(thumbAbs);
      } else {
        await thumb.webp({ quality: 82 }).toFile(thumbAbs);
      }
    }

    entries[i] = {
      index: i,
      label: norm.label,
      type: norm.type,
      box_2d: norm.box_2d,
      importance_geom: obj.importance_geom,
      importance_rank: obj.importance_rank ?? undefined,
      cutout_path: cutoutRel,
      thumb_path: thumbRel,
      crop_px: { left, top, width, height },
    };
  });

  const compactEntries: CutoutIndexEntry[] = entries.filter((e): e is CutoutIndexEntry => e != null);
  const payload: CutoutIndexPayload = {
    version: 1,
    image_path: options.imagePath,
    image_width: options.imageWidth,
    image_height: options.imageHeight,
    generated_at: new Date().toISOString(),
    count: compactEntries.length,
    entries: compactEntries,
  };

  fs.writeFileSync(indexPath, JSON.stringify(payload, null, 2));
  console.log(`🧾 Cutout index written to ${path.relative(process.cwd(), indexPath)}`);

  return { indexPath, directoryPath, count: compactEntries.length };
}

// ==========================================
// VERIFICATION IMAGE
// ==========================================
function getTypeColorHex(type: string): string {
  const key = normalizeType(type);
  if (key === 'person') return '#e63946';
  if (key === 'animal') return '#2ec4b6';
  if (key === 'building') return '#457b9d';
  if (key === 'landscape') return '#f4a261';
  if (key === 'object') return '#9b5de5';
  return '#ffc300';
}

function escapeXml(text: string): string {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

async function buildVerificationImage(options: {
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
  instances: DetectedObject[];
  kind: Kind;
}): Promise<Buffer> {
  const { imageBuffer, imageWidth, imageHeight, instances, kind } = options;

  const scaleX = imageWidth / 1000;
  const scaleY = imageHeight / 1000;
  const color = getTypeColorHex(kind.type);
  const thickness = Math.max(2, Math.round(Math.min(imageWidth, imageHeight) / 400));
  const fontSize = Math.max(16, Math.round(Math.min(imageWidth, imageHeight) / 40));

  const boxes = instances
    .map((inst, i) => {
      const [xmin, ymin, xmax, ymax] = normalizeBox(inst.box_2d);
      const x = xmin * scaleX;
      const y = ymin * scaleY;
      const w = (xmax - xmin) * scaleX;
      const h = (ymax - ymin) * scaleY;

      const badgeSize = fontSize * 1.4;
      const badge = `
        <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}"
              width="${badgeSize.toFixed(1)}" height="${badgeSize.toFixed(1)}"
              fill="${color}" />
        <text x="${(x + badgeSize / 2).toFixed(1)}" y="${(y + badgeSize / 2 + fontSize * 0.35).toFixed(1)}"
              font-size="${fontSize}" font-weight="bold" fill="white"
              font-family="system-ui, -apple-system, sans-serif"
              text-anchor="middle">${i}</text>
      `;

      return `
        <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}"
              width="${w.toFixed(1)}" height="${h.toFixed(1)}"
              fill="none" stroke="${color}" stroke-width="${thickness}" />
        ${badge}
      `;
    })
    .join('\n');

  const headerText = escapeXml(`Verify: ${instances.length} "${kind.kind}" box(es) — check for errors & missing`);
  const headerWidth = Math.min(imageWidth, 700);
  const header = `
    <rect x="0" y="0" width="${headerWidth}" height="36" fill="rgba(0,0,0,0.75)" />
    <text x="10" y="25" font-size="18" fill="white"
          font-family="system-ui, -apple-system, sans-serif">${headerText}</text>
  `;

  const svg = `
<svg xmlns="http://www.w3.org/2000/svg" width="${imageWidth}" height="${imageHeight}" viewBox="0 0 ${imageWidth} ${imageHeight}">
${header}
${boxes}
</svg>
  `.trim();

  return sharp(imageBuffer)
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .jpeg({ quality: 90 })
    .toBuffer();
}

// ==========================================
// AI FUNCTIONS
// ==========================================

async function estimateCountInRegion(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  kind: Kind;
}): Promise<CountEstimateType> {
  const { pool, imageBuffer, kind } = options;

  const prompt = `
You are counting instances of a specific object type in an image region.

Task: Estimate how many "${kind.kind}" (${kind.type}) are clearly visible in this image.

Count categories:
- "few": 1-10 instances
- "moderate": 11-25 instances
- "many": 26-50 instances
- "very_many": 50+ instances

Rules:
- Only count clearly visible instances of "${kind.kind}"
- Do not count partial/obscured instances
- Do not hallucinate from patterns in textures
- If none visible, return "few"

Output JSON only: {"estimated_count": "few|moderate|many|very_many"}
`;

  const result = await pool.generateObject({
    schema: RegionCountSchema,
    temperature: 0.1,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
  });

  return result.estimated_count;
}

async function adaptiveSubdivide(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
  kind: Kind;
  threshold: CountEstimateType;
  maxDepth: number;
  logPrefix?: string;
}): Promise<TileDefinition[]> {
  const { pool, imageBuffer, imageWidth, imageHeight, kind, threshold, maxDepth, logPrefix = '' } = options;

  const rootTile: TileDefinition = {
    row: 0,
    col: 0,
    left: 0,
    top: 0,
    width: imageWidth,
    height: imageHeight,
    label: 'root',
    depth: 0,
  };

  const pendingTiles: TileDefinition[] = [rootTile];
  const finalTiles: TileDefinition[] = [];

  while (pendingTiles.length > 0) {
    const tile = pendingTiles.pop()!;
    const depth = tile.depth ?? 0;
    const tileLogPrefix = `${logPrefix}[${'  '.repeat(depth)}${tile.label}] `;

    // Crop tile region
    let tileBuffer: Buffer;
    if (tile.left === 0 && tile.top === 0 && tile.width === imageWidth && tile.height === imageHeight) {
      tileBuffer = imageBuffer;
    } else {
      tileBuffer = await cropTileBuffer(imageBuffer, tile);
    }

    // Estimate count in this region
    let count: CountEstimateType;
    try {
      count = await estimateCountInRegion({ pool, imageBuffer: tileBuffer, kind });
      console.log(`${tileLogPrefix}count: ${count}`);
    } catch (error) {
      logErrorDetails(`${tileLogPrefix}⚠️ Count estimation failed, using tile as-is. `, error);
      finalTiles.push(tile);
      continue;
    }

    // Decide: subdivide or keep
    if (isCountAboveThreshold(count, threshold) && depth < maxDepth) {
      const quadrants = subdivideIntoQuadrants(tile, imageWidth, imageHeight);
      console.log(`${tileLogPrefix}→ subdividing into 4 quadrants`);
      pendingTiles.push(...quadrants);
    } else {
      if (count === 'few' && depth === 0) {
        // Full image with few objects - no tiling needed
        console.log(`${tileLogPrefix}→ few objects, no tiling needed`);
      } else {
        console.log(`${tileLogPrefix}→ keeping tile (count=${count}, depth=${depth})`);
      }
      finalTiles.push({ ...tile, _estimatedCount: count } as TileDefinition & { _estimatedCount: CountEstimateType });
    }
  }

  return finalTiles;
}

async function generateDescription(
  pool: AIPool,
  imageBuffer: Buffer,
  objects: DetectedObject[]
): Promise<DescriptionPayload> {
  const contextData = objects
    .map((o) => {
      const xCenter = (o.box_2d[0] + o.box_2d[2]) / 2;
      const yCenter = (o.box_2d[1] + o.box_2d[3]) / 2;
      const hPos = xCenter < 333 ? 'Left' : xCenter > 666 ? 'Right' : 'Center';
      const vPos = yCenter < 333 ? 'Background/Top' : yCenter > 666 ? 'Foreground/Bottom' : 'Midground';
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

  return pool.generateObject({
    schema: z.object({ alt_text: z.string(), long_description: z.string() }),
    temperature: 0.2,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
  });
}

// Size levels for comparison (smaller = needs more zoom)
const SIZE_LEVELS: Record<SizeEstimateType, number> = {
  tiny: 1,
  small: 2,
  medium: 3,
  large: 4,
  giant: 5,
};

function dedupeKinds(kinds: Kind[]): Kind[] {
  const seen = new Map<string, Kind>();
  for (const k of kinds) {
    const normalizedKind = k.kind.trim().toLowerCase();
    const key = `${k.type}:${normalizedKind}`;
    const existing = seen.get(key);

    if (existing) {
      // Keep higher count estimate
      if (COUNT_LEVELS[k.estimated_count] > COUNT_LEVELS[existing.estimated_count]) {
        existing.estimated_count = k.estimated_count;
      }
      // Keep smaller size estimate (more conservative - if any view thinks it's small, use tiling)
      const existingSize = existing.estimated_size ?? 'medium';
      const incomingSize = k.estimated_size ?? 'medium';
      if (SIZE_LEVELS[incomingSize] < SIZE_LEVELS[existingSize]) {
        existing.estimated_size = incomingSize;
      }
      continue;
    }

    seen.set(key, {
      kind: normalizedKind,
      type: k.type,
      estimated_count: k.estimated_count,
      estimated_size: k.estimated_size ?? 'medium',
      segmentation: k.segmentation ?? 'individual',
      importance: k.importance ?? 'secondary',
    });
  }
  return Array.from(seen.values());
}

async function discoverKinds(pool: AIPool, imageBuffer: Buffer, maxKinds: number): Promise<Kind[]> {
  const prompt = `
You are labeling an artwork for accessibility and cataloging.

Task: Identify the unique OBJECT KINDS visible in this image, considering their artistic importance, size, and how they should be detected.

CRITICAL: Think about what makes this artwork meaningful. What are the SUBJECTS vs the CONTEXT?

== DEFINITIONS ==

A "kind" is a noun phrase category (e.g., "hound", "hunter", "demon", "mountain", "forest").
Each kind MUST correspond to at least one clearly visible element.

== LABEL GUIDELINES ==

- Use SINGULAR form (e.g., "demon" not "demons")
- Use lowercase (e.g., "hound" not "Hound")
- Be specific when visually distinct (e.g., "crossbowman" not just "person")
- Do NOT invent things from patterns in water/clouds/foliage

== TYPE (category) ==

Choose an appropriate category. Common types: person, animal, building, landscape, object, vehicle, plant, symbol, creature, figure
For specialized subjects: angel, demon, deity, hero, specimen, etc.

== ESTIMATED SIZE ==

How large is a TYPICAL INSTANCE of this kind, relative to the full image?

- "tiny": <2% of image area (distant birds, specks, tiny figures in vast landscape)
- "small": 2-10% of image (horses in landscape, people in crowd, background figures)
- "medium": 10-30% of image (group portrait subjects, main animals)
- "large": 30-60% of image (primary portrait subject, dominant figure)
- "giant": >60% of image (close-up face, single subject filling frame)

IMPORTANT: Size is about the OBJECT relative to the IMAGE, not real-world size.
A horse in a vast landscape might be "tiny" or "small", while a beetle in a specimen painting might be "large".

== IMPORTANCE (artistic role) ==

- "primary": Main subjects, focal points, narrative figures (people in a portrait, demons in a hellscape, the central action)
- "secondary": Supporting elements that add meaning (specific animals, important objects, architectural details)
- "background": Contextual/environmental elements (trees in a forest, clouds, rocks, grass, water)

== SEGMENTATION (detection strategy) ==

- "individual": Detect EVERY instance separately. Use for:
  - Primary subjects (each person, each demon, each main animal)
  - Elements where each instance is unique/interesting
  - Countable narrative elements

- "representative": Detect a FEW examples (3-8). Use for:
  - Secondary elements with many similar instances
  - Background elements worth noting but not exhaustively cataloging
  - E.g., "a few representative trees" in a forest landscape

- "region": Detect as area/mass, not instances. Use for:
  - Masses of similar elements (forest, crowd, field of grass)
  - Atmospheric elements (clouds, mist, sky)
  - Textures/surfaces (water, snow, ground)

== EXAMPLES ==

Landscape painting (Bierstadt):
- mountain: landscape, few, large, individual, primary
- eagle: animal, few, small, individual, primary
- horse: animal, few, tiny, individual, secondary (distant horses in vast landscape!)
- rider: person, few, tiny, individual, secondary (distant figures)
- forest: plant, few, large, region, background
- cloud: landscape, many, medium, region, background

Bosch-style hellscape:
- demon: creature, very_many, small, individual, primary (EACH demon is interesting!)
- sinner: person, many, small, individual, primary
- flame: element, very_many, small, region, background

Portrait:
- nobleman: person, few, giant, individual, primary (fills most of frame)
- ring: object, few, tiny, individual, secondary (small detail on hand)

== ESTIMATED COUNT ==

- "few": 1-10, "moderate": 11-25, "many": 26-50, "very_many": 50+

== OUTPUT ==

JSON only:
{"kinds":[{"kind":"","type":"","estimated_count":"","estimated_size":"","segmentation":"","importance":""}, ...]}

Keep the list <= ${maxKinds}. Prefer fewer, well-chosen kinds over exhaustive lists.
`;

  const result = await pool.generateObject({
    schema: KindSchema,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
  });

  return dedupeKinds(result.kinds).slice(0, maxKinds);
}

type QuadrantDiscovery = {
  name: QuadrantNameType;
  kinds: Kind[];
};

async function reconcileKinds(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  fullImageKinds: Kind[];
  quadrantDiscoveries: QuadrantDiscovery[];
  maxKinds: number;
}): Promise<ReconciledKind[]> {
  const { pool, imageBuffer, fullImageKinds, quadrantDiscoveries, maxKinds } = options;

  // Build the context of what was found where
  const fullKindsList = fullImageKinds.map(k => `- ${k.kind} (${k.type}): ${k.estimated_count}, ${k.estimated_size}, ${k.importance}`).join('\n');

  const quadrantSections = quadrantDiscoveries.map(q => {
    if (q.kinds.length === 0) return `${q.name.toUpperCase()}: (nothing found)`;
    const kindsList = q.kinds.map(k => `- ${k.kind} (${k.type}): ${k.estimated_count}, ${k.estimated_size}`).join('\n');
    return `${q.name.toUpperCase()}:\n${kindsList}`;
  }).join('\n\n');

  // Collect all unique kinds for reference
  const allKinds = new Map<string, Kind>();
  for (const k of fullImageKinds) {
    const key = `${k.type}:${k.kind.toLowerCase()}`;
    allKinds.set(key, k);
  }
  for (const q of quadrantDiscoveries) {
    for (const k of q.kinds) {
      const key = `${k.type}:${k.kind.toLowerCase()}`;
      if (!allKinds.has(key)) {
        allKinds.set(key, k);
      }
    }
  }

  const prompt = `
You are reconciling object detection results from multiple views of an artwork.

The image was analyzed at two scales:
1. FULL IMAGE - seeing the complete artwork
2. QUADRANTS - four overlapping 55% crops (top-left, top-right, bottom-left, bottom-right)

Here's what was discovered:

== FULL IMAGE ANALYSIS ==
${fullKindsList || '(nothing found)'}

== QUADRANT ANALYSIS ==
${quadrantSections}

== YOUR TASK ==

Looking at the FULL IMAGE with complete context, reconcile these findings:

1. **Filter Artifacts**: Some quadrant detections may be WRONG — textures misread as objects,
   cloud shapes mistaken for animals, mountain ridges seen as figures, etc.
   Mark these as is_real=false.

2. **Confirm Real Objects**: For each REAL object kind, specify:
   - Which quadrants actually contain instances of it
   - Whether to detect at full image scale or quadrant scale

3. **Spatial Attribution**: A kind found in "top-left" quadrant discovery should only have
   "top-left" in its quadrants array IF instances actually exist there.

== DETECTION SCALE GUIDANCE ==

- detection_scale="full": Object is large enough to detect reliably on full image
  (medium, large, giant size OR low count that doesn't need zoom)

- detection_scale="quadrant": Object is small/tiny OR there are many instances,
  requiring zoomed detection. Will only detect in specified quadrants.

== OUTPUT ==

Return ALL real object kinds with their spatial information.
Keep the list <= ${maxKinds} kinds, prioritizing primary/secondary importance.

JSON only:
{"kinds":[{"kind":"","type":"","is_real":true/false,"quadrants":["top-left",...],"estimated_count":"","estimated_size":"","segmentation":"","importance":"","detection_scale":"full|quadrant"}, ...]}
`;

  const result = await pool.generateObject({
    schema: ReconciliationSchema,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
  });

  // Filter to only real kinds and limit
  return result.kinds
    .filter(k => k.is_real)
    .slice(0, maxKinds);
}

async function detectInstancesForKind(options: { pool: AIPool; imageBuffer: Buffer; kind: Kind }): Promise<DetectedObject[]> {
  const prompt = `
You are an expert computer vision annotator.

Task: find ALL visible instances of: "${options.kind.kind}" (${options.kind.type})

Rules:
- Only return instances that are clearly visible.
- Boxes must be tight and accurate.
- Return an empty list if you see none.
- Use label exactly "${options.kind.kind}" for all objects.

Output JSON only:
{"objects":[{"label":"","type":"","box_2d":[xmin,ymin,xmax,ymax]}, ...]}
box_2d is normalized 0–1000.
`;

  const result = await options.pool.generateObject({
    schema: InstancesSchema,
    temperature: 0.15,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: options.imageBuffer }] }],
  });

  return result.objects.map((o) =>
    normalizeObject({ ...o, label: options.kind.kind, type: options.kind.type })
  );
}

async function verifyInstances(options: {
  pool: AIPool;
  visualContextBuffer: Buffer;
  kind: Kind;
  instanceCount: number;
}): Promise<z.infer<typeof VerifyResponseSchema>> {
  const { pool, visualContextBuffer, kind, instanceCount } = options;

  const prompt = `
You are verifying bounding box annotations for: "${kind.kind}" (${kind.type})

The image shows ${instanceCount} numbered box(es), labeled 0 to ${instanceCount - 1}.

CRITICAL: Be skeptical. Some boxes may be on objects that are NOT "${kind.kind}" at all.

Your tasks:

1. WRONG BOXES (wrong_indices) — CHECK THIS FIRST:
   Remove boxes where:
   - There is NO "${kind.kind}" in or near the box (hallucinated detection)
   - The box is on a DIFFERENT object type (misidentification)
   - The shape/texture was mistaken for "${kind.kind}" but isn't one
   Be honest: if it's not clearly a "${kind.kind}", remove it.

2. CORRECTIONS (corrections):
   For boxes that ARE on a real "${kind.kind}" but are misaligned:
   - Provide the index and corrected [xmin, ymin, xmax, ymax] in 0-1000 coords
   - Only correct boxes that truly contain a "${kind.kind}"

3. MISSING (missing):
   Find any CLEARLY VISIBLE "${kind.kind}" instances that have NO box yet:
   - Provide box coordinates [xmin, ymin, xmax, ymax] for each
   - Be conservative — only add instances you're certain about
   - Do NOT hallucinate from textures, patterns, shadows, or similar shapes

4. COMPLETE (complete):
   Set true only if ALL visible "${kind.kind}" instances are now correctly boxed.

Output JSON only with these exact fields.
`;

  return pool.generateObject({
    schema: VerifyResponseSchema,
    temperature: 0.1,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: visualContextBuffer }] }],
  });
}

// ==========================================
// DETECTION PIPELINE
// ==========================================
async function detectAndVerifyRegion(options: {
  pool: AIPool;
  regionBuffer: Buffer;
  regionWidth: number;
  regionHeight: number;
  kind: Kind;
  maxVerifyRounds: number;
  logPrefix: string;
}): Promise<DetectedObject[]> {
  const { pool, regionBuffer, regionWidth, regionHeight, kind, maxVerifyRounds, logPrefix } = options;

  let instances: DetectedObject[];
  try {
    instances = await detectInstancesForKind({ pool, imageBuffer: regionBuffer, kind });
    console.log(`${logPrefix}Found ${instances.length} instance(s)`);
  } catch (error) {
    logErrorDetails(`${logPrefix}⚠️ Detection failed. `, error);
    return [];
  }

  if (maxVerifyRounds <= 0 || instances.length === 0) {
    return instances;
  }

  for (let round = 0; round < maxVerifyRounds; round++) {
    console.log(`${logPrefix}🔍 Verify ${round + 1}/${maxVerifyRounds}...`);

    const visualCtx = await buildVerificationImage({
      imageBuffer: regionBuffer,
      imageWidth: regionWidth,
      imageHeight: regionHeight,
      instances,
      kind,
    });

    let verification: z.infer<typeof VerifyResponseSchema>;
    try {
      verification = await verifyInstances({
        pool,
        visualContextBuffer: visualCtx,
        kind,
        instanceCount: instances.length,
      });
    } catch (error) {
      logErrorDetails(`${logPrefix}⚠️ Verification failed. `, error);
      break;
    }

    let changed = false;

    const wrongSet = new Set(verification.wrong_indices.filter((i) => i >= 0 && i < instances.length));
    if (wrongSet.size > 0) {
      console.log(`${logPrefix}  ❌ Remove ${wrongSet.size} box(es)`);
      instances = instances.filter((_, i) => !wrongSet.has(i));
      changed = true;
    }

    if (verification.corrections.length > 0) {
      let correctionCount = 0;
      for (const corr of verification.corrections) {
        if (wrongSet.has(corr.index)) continue;
        const removedBelow = [...wrongSet].filter((w) => w < corr.index).length;
        const adjustedIndex = corr.index - removedBelow;
        if (adjustedIndex >= 0 && adjustedIndex < instances.length) {
          instances[adjustedIndex] = { ...instances[adjustedIndex], box_2d: normalizeBox(corr.box_2d) };
          correctionCount++;
        }
      }
      if (correctionCount > 0) {
        console.log(`${logPrefix}  ✏️  Correct ${correctionCount} box(es)`);
        changed = true;
      }
    }

    if (verification.missing.length > 0) {
      console.log(`${logPrefix}  ➕ Add ${verification.missing.length} missing`);
      for (const m of verification.missing) {
        instances.push({ label: kind.kind, type: kind.type, box_2d: normalizeBox(m.box_2d) });
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

    const beforeDedupe = instances.length;
    instances = dedupeObjectsByGeometry(instances) as DetectedObject[];
    if (instances.length < beforeDedupe) {
      console.log(`${logPrefix}  🔄 Deduped ${beforeDedupe} → ${instances.length}`);
    }
  }

  return instances;
}

async function detectAndVerifyAdaptive(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
  kind: Kind;
  maxVerifyRounds: number;
  maxDepth: number;
  logPrefix?: string;
}): Promise<DetectedObject[]> {
  const { pool, imageBuffer, imageWidth, imageHeight, kind, maxVerifyRounds, maxDepth, logPrefix = '' } = options;

  // Phase 1: Adaptive subdivision based on count estimates
  console.log(`${logPrefix}   🔲 Adaptive tiling (estimating counts)...`);
  const tiles = await adaptiveSubdivide({
    pool,
    imageBuffer,
    imageWidth,
    imageHeight,
    kind,
    threshold: 'moderate', // Subdivide if count > moderate
    maxDepth,
    logPrefix: logPrefix + '   ',
  });

  // If only root tile with few objects, detect on full image
  if (tiles.length === 1 && tiles[0].label === 'root') {
    console.log(`${logPrefix}   📍 Full image detect+verify...`);
    return detectAndVerifyRegion({
      pool,
      regionBuffer: imageBuffer,
      regionWidth: imageWidth,
      regionHeight: imageHeight,
      kind,
      maxVerifyRounds,
      logPrefix: logPrefix + '      ',
    });
  }

  // Phase 2: Detect on each tile in parallel
  console.log(`${logPrefix}   📍 Detecting on ${tiles.length} adaptive tile(s)...`);
  const tileResults = await Promise.all(
    tiles.map(async (tile, i) => {
      const tileLabel = `${tile.label} (${i + 1}/${tiles.length})`;
      const tileLogPrefix = `${logPrefix}      [${tileLabel}] `;

      try {
        const tileBuffer = await cropTileBuffer(imageBuffer, tile);

        const tileInstances = await detectAndVerifyRegion({
          pool,
          regionBuffer: tileBuffer,
          regionWidth: tile.width,
          regionHeight: tile.height,
          kind,
          maxVerifyRounds,
          logPrefix: tileLogPrefix,
        });

        // Map tile coords to full image coords and filter boundary-clipped boxes
        const mappedInstances: DetectedObject[] = [];
        let clippedCount = 0;
        for (const inst of tileInstances) {
          const tileBox = normalizeBox(inst.box_2d);
          if (isBoxClippedAtTileBoundary(tileBox, tile, imageWidth, imageHeight)) {
            clippedCount++;
            continue;
          }
          const fullImageBox = mapTileBoxToFullImage(tileBox, tile, imageWidth, imageHeight);
          mappedInstances.push({ ...inst, box_2d: fullImageBox });
        }

        if (clippedCount > 0) {
          console.log(`${tileLogPrefix}🚫 Filtered ${clippedCount} boundary-clipped box(es)`);
        }

        console.log(`${tileLogPrefix}📦 ${mappedInstances.length} verified → global`);
        return mappedInstances;
      } catch (error) {
        logErrorDetails(`${tileLogPrefix}⚠️ Tile failed: `, error);
        return [];
      }
    })
  );

  // Phase 3: Merge and dedupe
  const allInstances = tileResults.flat();
  const beforeDedupe = allInstances.length;
  const deduped = dedupeObjectsByGeometry(allInstances) as DetectedObject[];
  if (deduped.length < beforeDedupe) {
    console.log(`${logPrefix}   🔄 Tile merge: ${beforeDedupe} → ${deduped.length} (removed ${beforeDedupe - deduped.length} duplicates)`);
  }

  return deduped;
}

// For tiny objects: skip adaptive subdivision, immediately detect on 4 quadrants
async function detectOnQuadrants(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
  kind: Kind;
  maxVerifyRounds: number;
  logPrefix?: string;
}): Promise<DetectedObject[]> {
  const { pool, imageBuffer, imageWidth, imageHeight, kind, maxVerifyRounds, logPrefix = '' } = options;

  // Create 4 overlapping quadrants (55% size with 10% overlap in center)
  const quadrantDefs = getQuadrantDefs(imageWidth, imageHeight);

  console.log(`${logPrefix}   🔲 Detecting on 4 quadrants (skipping count estimation for tiny objects)...`);

  const quadrantResults = await Promise.all(
    quadrantDefs.map(async (q) => {
      const quadrantLogPrefix = `${logPrefix}      [${q.name}] `;

      try {
        const quadrantBuffer = await sharp(imageBuffer)
          .extract({ left: q.left, top: q.top, width: q.width, height: q.height })
          .jpeg({ quality: 92 })
          .toBuffer();

        const instances = await detectAndVerifyRegion({
          pool,
          regionBuffer: quadrantBuffer,
          regionWidth: q.width,
          regionHeight: q.height,
          kind,
          maxVerifyRounds,
          logPrefix: quadrantLogPrefix,
        });

        // Map quadrant coordinates back to full image
        const mappedInstances: DetectedObject[] = [];
        for (const inst of instances) {
          const [xmin, ymin, xmax, ymax] = normalizeBox(inst.box_2d);

          // Convert from quadrant's 0-1000 coords to pixel coords in quadrant
          const pxXmin = (xmin / 1000) * q.width;
          const pxYmin = (ymin / 1000) * q.height;
          const pxXmax = (xmax / 1000) * q.width;
          const pxYmax = (ymax / 1000) * q.height;

          // Add quadrant offset to get full image pixel coords
          const fullPxXmin = q.left + pxXmin;
          const fullPxYmin = q.top + pxYmin;
          const fullPxXmax = q.left + pxXmax;
          const fullPxYmax = q.top + pxYmax;

          // Convert back to 0-1000 normalized coords
          const normXmin = Math.round((fullPxXmin / imageWidth) * 1000);
          const normYmin = Math.round((fullPxYmin / imageHeight) * 1000);
          const normXmax = Math.round((fullPxXmax / imageWidth) * 1000);
          const normYmax = Math.round((fullPxYmax / imageHeight) * 1000);

          mappedInstances.push({
            ...inst,
            box_2d: [
              clamp(normXmin, 0, 1000),
              clamp(normYmin, 0, 1000),
              clamp(normXmax, 0, 1000),
              clamp(normYmax, 0, 1000),
            ],
          });
        }

        console.log(`${quadrantLogPrefix}📦 ${mappedInstances.length} instance(s) → global`);
        return mappedInstances;
      } catch (error) {
        logErrorDetails(`${quadrantLogPrefix}⚠️ Quadrant failed: `, error);
        return [];
      }
    })
  );

  // Merge and dedupe
  const allInstances = quadrantResults.flat();
  const beforeDedupe = allInstances.length;
  const deduped = dedupeObjectsByGeometry(allInstances) as DetectedObject[];
  if (deduped.length < beforeDedupe) {
    console.log(`${logPrefix}   🔄 Quadrant merge: ${beforeDedupe} → ${deduped.length} (removed ${beforeDedupe - deduped.length} duplicates)`);
  }

  return deduped;
}

async function detectAndVerifyKind(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
  kind: Kind;
  maxVerifyRounds: number;
  tileThreshold: number;
  logPrefix?: string;
}): Promise<DetectedObject[]> {
  const { pool, imageBuffer, imageWidth, imageHeight, kind, maxVerifyRounds, tileThreshold, logPrefix = '' } = options;

  const segmentation = kind.segmentation ?? 'individual';
  const importance = kind.importance ?? 'secondary';
  const estimatedSize = kind.estimated_size ?? 'medium';

  // Handle different segmentation strategies
  if (segmentation === 'region') {
    // For regions, detect bounding areas (no individual instances)
    console.log(`${logPrefix}  📍 Region detection (${importance})...`);
    return detectRegions({
      pool,
      imageBuffer,
      kind,
      logPrefix: logPrefix + '     ',
    });
  }

  if (segmentation === 'representative') {
    // Detect a few representative examples (cap at 8)
    console.log(`${logPrefix}  📍 Representative detection (max 8, ${importance})...`);
    const instances = await detectAndVerifyRegion({
      pool,
      regionBuffer: imageBuffer,
      regionWidth: imageWidth,
      regionHeight: imageHeight,
      kind,
      maxVerifyRounds: Math.min(maxVerifyRounds, 1), // Fewer verify rounds for representative
      logPrefix: logPrefix + '     ',
    });
    // Cap at 8 representative instances, preferring diverse positions
    if (instances.length <= 8) return instances;
    return selectRepresentativeInstances(instances, 8);
  }

  // segmentation === 'individual': detect every instance
  // Use tiled detection when:
  // 1. Objects are tiny - they MUST be zoomed, skip count estimation entirely
  // 2. Objects are small - they need zoom but can use adaptive subdivision
  // 3. High count (many/very_many) - too many to detect reliably at once
  const isTinyObject = estimatedSize === 'tiny';
  const isSmallObject = estimatedSize === 'small';
  const isHighCount = isCountAboveThreshold(kind.estimated_count, 'moderate');

  if (tileThreshold > 0 && isTinyObject) {
    // Tiny objects: skip adaptive subdivision entirely, immediately split into quadrants
    // Count estimation at full scale is unreliable for tiny objects
    console.log(`${logPrefix}  📍 Quadrant detect+verify (size: tiny, ${importance})...`);
    return detectOnQuadrants({
      pool,
      imageBuffer,
      imageWidth,
      imageHeight,
      kind,
      maxVerifyRounds,
      logPrefix: logPrefix + '  ',
    });
  }

  if (tileThreshold > 0 && (isSmallObject || isHighCount)) {
    const reason = isSmallObject
      ? `size: ${estimatedSize}`
      : `count: ${kind.estimated_count}`;
    const maxDepth = isSmallObject && !isHighCount ? 2 : 3;
    console.log(`${logPrefix}  📍 Adaptive tiled detect+verify (${reason}, ${importance})...`);
    return detectAndVerifyAdaptive({
      pool,
      imageBuffer,
      imageWidth,
      imageHeight,
      kind,
      maxVerifyRounds,
      maxDepth,
      logPrefix: logPrefix + '  ',
    });
  }

  console.log(`${logPrefix}  📍 Full image detect+verify (size: ${estimatedSize}, count: ${kind.estimated_count}, ${importance})...`);
  return detectAndVerifyRegion({
    pool,
    regionBuffer: imageBuffer,
    regionWidth: imageWidth,
    regionHeight: imageHeight,
    kind,
    maxVerifyRounds,
    logPrefix: logPrefix + '     ',
  });
}

// Select spatially diverse representative instances
function selectRepresentativeInstances(instances: DetectedObject[], count: number): DetectedObject[] {
  if (instances.length <= count) return instances;

  // Sort by area (larger first) as a proxy for importance
  const sorted = [...instances].sort((a, b) => {
    const areaA = (a.box_2d[2] - a.box_2d[0]) * (a.box_2d[3] - a.box_2d[1]);
    const areaB = (b.box_2d[2] - b.box_2d[0]) * (b.box_2d[3] - b.box_2d[1]);
    return areaB - areaA;
  });

  // Greedily select instances that are spatially diverse
  const selected: DetectedObject[] = [sorted[0]];
  const getCenter = (obj: DetectedObject) => ({
    x: (obj.box_2d[0] + obj.box_2d[2]) / 2,
    y: (obj.box_2d[1] + obj.box_2d[3]) / 2,
  });

  for (const candidate of sorted.slice(1)) {
    if (selected.length >= count) break;
    const candCenter = getCenter(candidate);
    // Check minimum distance from already selected
    const minDist = Math.min(
      ...selected.map((s) => {
        const sCenter = getCenter(s);
        return Math.sqrt((candCenter.x - sCenter.x) ** 2 + (candCenter.y - sCenter.y) ** 2);
      })
    );
    // Require some minimum separation (100 units in normalized coords)
    if (minDist > 80 || selected.length < 3) {
      selected.push(candidate);
    }
  }

  // If we still need more, add remaining largest ones
  for (const candidate of sorted) {
    if (selected.length >= count) break;
    if (!selected.includes(candidate)) {
      selected.push(candidate);
    }
  }

  return selected.slice(0, count);
}

// Detect regions/areas rather than individual instances
async function detectRegions(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  kind: Kind;
  logPrefix?: string;
}): Promise<DetectedObject[]> {
  const { pool, imageBuffer, kind, logPrefix = '' } = options;

  const prompt = `
You are detecting REGIONS/AREAS in an artwork, not individual instances.

Task: Find the main AREAS where "${kind.kind}" (${kind.type}) appears in this image.

Rules:
- Draw bounding boxes around REGIONS/MASSES, not individual items
- For a forest: one box around the forested area, not each tree
- For clouds: boxes around cloud masses, not each cloud
- For grass/fields: boxes around grassy areas
- Typically 1-5 regions maximum
- Boxes can be large and encompassing

Output JSON only:
{"objects":[{"label":"","type":"","box_2d":[xmin,ymin,xmax,ymax]}, ...]}
box_2d is normalized 0-1000.
`;

  try {
    const result = await pool.generateObject({
      schema: InstancesSchema,
      temperature: 0.15,
      messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
    });

    const regions = result.objects.map((o) =>
      normalizeObject({ ...o, label: kind.kind, type: kind.type })
    );
    console.log(`${logPrefix}Found ${regions.length} region(s)`);
    return regions;
  } catch (error) {
    logErrorDetails(`${logPrefix}⚠️ Region detection failed. `, error);
    return [];
  }
}

// ==========================================
// MAIN ENTRY POINT
// ==========================================
export async function runDetection(config: DetectionConfig): Promise<OutputPayload> {
  console.log(`   Model: ${config.modelName}`);
  console.log(`   Verify rounds: ${config.verifyRounds} | Tile threshold: ${config.tileThreshold === 0 ? 'off' : `>${config.tileThreshold}`} | Multi-scale: ${config.multiScaleDiscovery ? 'on' : 'off'} | Concurrency: ${config.concurrency}`);

  if (!config.mock && !process.env.GOOGLE_GENERATIVE_AI_API_KEY) {
    throw new Error('GOOGLE_GENERATIVE_AI_API_KEY is missing. Add it to .env or set it in your environment.');
  }

  const pool = createAIPool({
    maxConcurrency: config.concurrency,
    model: config.modelName,
    debug: process.env.DEBUG_POOL === '1',
  });
  const descriptionPool = createAIPool({
    maxConcurrency: 1,
    model: config.descriptionModelName,
    debug: process.env.DEBUG_POOL === '1',
  });

  const resolvedImagePath = path.resolve(config.imagePath);
  if (!fs.existsSync(resolvedImagePath)) {
    throw new Error(`Image not found at ${resolvedImagePath}`);
  }

  const imageBuffer = fs.readFileSync(resolvedImagePath);
  const metadata = await sharp(imageBuffer).metadata();
  const imageWidth = metadata.width;
  const imageHeight = metadata.height;
  if (!imageWidth || !imageHeight) {
    throw new Error('Unable to read image dimensions.');
  }

  if (config.mock) {
    console.log('🧪 Mock mode: skipping API calls.');
  }

  // Phase 1: Discover kinds
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('PHASE 1: Discovering object kinds...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  let reconciledKinds: ReconciledKind[];

  if (config.mock) {
    reconciledKinds = [
      { kind: 'hound', type: 'animal', is_real: true, quadrants: ['bottom-left', 'bottom-right'], estimated_count: 'many', estimated_size: 'small', segmentation: 'individual', importance: 'primary', detection_scale: 'quadrant' },
      { kind: 'stag', type: 'animal', is_real: true, quadrants: ['bottom-right'], estimated_count: 'moderate', estimated_size: 'medium', segmentation: 'individual', importance: 'primary', detection_scale: 'full' },
      { kind: 'hunter', type: 'person', is_real: true, quadrants: ['bottom-left', 'bottom-right'], estimated_count: 'moderate', estimated_size: 'small', segmentation: 'individual', importance: 'primary', detection_scale: 'quadrant' },
    ];
  } else {
    // Step 1a: Full image discovery
    console.log('\n🔍 Step 1a: Full image analysis...');
    const fullImageKinds = await discoverKinds(pool, imageBuffer, config.maxKinds);
    console.log(`   Found ${fullImageKinds.length} kind(s)`);

    // Step 1b: Quadrant discovery (if multi-scale enabled)
    let quadrantDiscoveries: QuadrantDiscovery[] = [];

    if (config.multiScaleDiscovery) {
      console.log('\n🔬 Step 1b: Quadrant analysis...');

      const quadrantDefs = getQuadrantDefs(imageWidth, imageHeight);

      quadrantDiscoveries = await Promise.all(
        quadrantDefs.map(async (q): Promise<QuadrantDiscovery> => {
          try {
            const quadrantBuffer = await sharp(imageBuffer)
              .extract({ left: q.left, top: q.top, width: q.width, height: q.height })
              .jpeg({ quality: 92 })
              .toBuffer();

            const quadrantKinds = await discoverKinds(pool, quadrantBuffer, Math.ceil(config.maxKinds / 2));
            console.log(`   [${q.name}] Found ${quadrantKinds.length} kind(s)`);
            return { name: q.name, kinds: quadrantKinds };
          } catch (error) {
            console.warn(`   [${q.name}] ⚠️ Failed: ${error instanceof Error ? error.message : String(error)}`);
            return { name: q.name, kinds: [] };
          }
        })
      );

      const totalQuadrantKinds = quadrantDiscoveries.reduce((sum, q) => sum + q.kinds.length, 0);
      console.log(`   Total: ${totalQuadrantKinds} kind(s) from quadrants`);
    }

    // Step 1c: Reconciliation - filter artifacts and get spatial info
    console.log('\n🔄 Step 1c: Reconciling discoveries with full image context...');
    reconciledKinds = await reconcileKinds({
      pool,
      imageBuffer,
      fullImageKinds,
      quadrantDiscoveries,
      maxKinds: config.maxKinds,
    });
    console.log(`   Reconciled to ${reconciledKinds.length} confirmed kind(s)`);
  }

  console.log(`\n🧭 Reconciled ${reconciledKinds.length} kind(s):`);
  for (const k of reconciledKinds.slice(0, 60)) {
    const seg = k.segmentation === 'individual' ? '⊡' : k.segmentation === 'representative' ? '◇' : '▢';
    const imp = k.importance === 'primary' ? '★' : k.importance === 'secondary' ? '☆' : '·';
    const size = k.estimated_size ?? 'medium';
    const sizeIcon = size === 'tiny' ? '🔬' : size === 'small' ? '🔹' : size === 'large' ? '🔷' : size === 'giant' ? '⬛' : '';
    const scale = k.detection_scale === 'quadrant' ? `[${k.quadrants.join(', ')}]` : '[full]';
    console.log(`   ${imp} ${k.kind} (${k.type}) — ${k.estimated_count}, ${size} ${seg}${sizeIcon ? ' ' + sizeIcon : ''} ${scale}`);
  }
  if (reconciledKinds.length > 60) {
    console.log(`   ... and ${reconciledKinds.length - 60} more`);
  }

  // Convert to Kind array for payload (backwards compat)
  const kinds: Kind[] = reconciledKinds.map(k => ({
    kind: k.kind,
    type: k.type,
    estimated_count: k.estimated_count,
    estimated_size: k.estimated_size,
    segmentation: k.segmentation,
    importance: k.importance,
  }));

  // Phase 2: Detect + Verify each kind
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('PHASE 2: Detecting and verifying instances per kind...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  let kindsToProcess = reconciledKinds;
  if (config.onlyKinds && config.onlyKinds.length > 0) {
    kindsToProcess = reconciledKinds.filter((k) => config.onlyKinds!.includes(k.kind.toLowerCase()));
    console.log(`\n⚠️  --only-kinds: processing ${kindsToProcess.length} of ${reconciledKinds.length} kinds: ${config.onlyKinds.join(', ')}`);
  }

  let allObjects: StoredObject[] = [];

  // Pre-compute quadrant buffers for reuse
  const quadrantBuffers = new Map<QuadrantNameType, { buffer: Buffer; width: number; height: number; left: number; top: number }>();
  const quadrantDefs = getQuadrantDefs(imageWidth, imageHeight);

  if (!config.mock) {
    const total = kindsToProcess.length;
    console.log(`\n🔀 Processing ${total} kinds (pool concurrency: ${config.concurrency})`);

    const results = await Promise.all(
      kindsToProcess.map(async (rKind, i) => {
        const kindLabel = `[${i + 1}/${total}] ${rKind.kind}`;
        const kind: Kind = {
          kind: rKind.kind,
          type: rKind.type,
          estimated_count: rKind.estimated_count,
          estimated_size: rKind.estimated_size,
          segmentation: rKind.segmentation,
          importance: rKind.importance,
        };

        // Handle different segmentation strategies first
        if (rKind.segmentation === 'region') {
          console.log(`\n${kindLabel} (${kind.type}) → region detection`);
          const regions = await detectRegions({ pool, imageBuffer, kind, logPrefix: `${kindLabel}   ` });
          console.log(`${kindLabel}   📦 Result: ${regions.length} region(s)`);
          return regions;
        }

        if (rKind.segmentation === 'representative') {
          console.log(`\n${kindLabel} (${kind.type}) → representative (max 8)`);
          const instances = await detectAndVerifyRegion({
            pool,
            regionBuffer: imageBuffer,
            regionWidth: imageWidth,
            regionHeight: imageHeight,
            kind,
            maxVerifyRounds: Math.min(config.verifyRounds, 1),
            logPrefix: `${kindLabel}   `,
          });
          const selected = instances.length <= 8 ? instances : selectRepresentativeInstances(instances, 8);
          console.log(`${kindLabel}   📦 Result: ${selected.length} representative instance(s)`);
          return selected;
        }

        // Individual detection - use detection_scale from reconciliation
        if (rKind.detection_scale === 'full') {
          console.log(`\n${kindLabel} (${kind.type}) → full image`);
          const instances = await detectAndVerifyRegion({
            pool,
            regionBuffer: imageBuffer,
            regionWidth: imageWidth,
            regionHeight: imageHeight,
            kind,
            maxVerifyRounds: config.verifyRounds,
            logPrefix: `${kindLabel}   `,
          });
          console.log(`${kindLabel}   📦 Result: ${instances.length} verified instance(s)`);
          return instances;
        }

        // Quadrant-level detection
        console.log(`\n${kindLabel} (${kind.type}) → quadrants: [${rKind.quadrants.join(', ')}]`);

        if (rKind.quadrants.length === 0) {
          console.log(`${kindLabel}   ⚠️ No quadrants specified, skipping`);
          return [];
        }

        const quadrantResults = await Promise.all(
          rKind.quadrants.map(async (qName) => {
            const qDef = quadrantDefs.find(q => q.name === qName);
            if (!qDef) return [];

            const quadrantLogPrefix = `${kindLabel}   [${qName}] `;

            try {
              // Get or create quadrant buffer
              let qData = quadrantBuffers.get(qName);
              if (!qData) {
                const buffer = await sharp(imageBuffer)
                  .extract({ left: qDef.left, top: qDef.top, width: qDef.width, height: qDef.height })
                  .jpeg({ quality: 92 })
                  .toBuffer();
                qData = { buffer, width: qDef.width, height: qDef.height, left: qDef.left, top: qDef.top };
                quadrantBuffers.set(qName, qData);
              }

              const instances = await detectAndVerifyRegion({
                pool,
                regionBuffer: qData.buffer,
                regionWidth: qData.width,
                regionHeight: qData.height,
                kind,
                maxVerifyRounds: config.verifyRounds,
                logPrefix: quadrantLogPrefix,
              });

              // Map quadrant coordinates back to full image
              const mappedInstances: DetectedObject[] = instances.map(inst => {
                const [xmin, ymin, xmax, ymax] = normalizeBox(inst.box_2d);

                // Convert from quadrant's 0-1000 coords to pixel coords in quadrant
                const pxXmin = (xmin / 1000) * qData!.width;
                const pxYmin = (ymin / 1000) * qData!.height;
                const pxXmax = (xmax / 1000) * qData!.width;
                const pxYmax = (ymax / 1000) * qData!.height;

                // Add quadrant offset to get full image pixel coords
                const fullPxXmin = qData!.left + pxXmin;
                const fullPxYmin = qData!.top + pxYmin;
                const fullPxXmax = qData!.left + pxXmax;
                const fullPxYmax = qData!.top + pxYmax;

                // Convert back to 0-1000 normalized coords
                return {
                  ...inst,
                  box_2d: [
                    clamp(Math.round((fullPxXmin / imageWidth) * 1000), 0, 1000),
                    clamp(Math.round((fullPxYmin / imageHeight) * 1000), 0, 1000),
                    clamp(Math.round((fullPxXmax / imageWidth) * 1000), 0, 1000),
                    clamp(Math.round((fullPxYmax / imageHeight) * 1000), 0, 1000),
                  ] as [number, number, number, number],
                };
              });

              console.log(`${quadrantLogPrefix}📦 ${mappedInstances.length} instance(s) → global`);
              return mappedInstances;
            } catch (error) {
              logErrorDetails(`${quadrantLogPrefix}⚠️ Failed: `, error);
              return [];
            }
          })
        );

        // Merge and dedupe across quadrants
        const allInstances = quadrantResults.flat();
        const beforeDedupe = allInstances.length;
        const deduped = dedupeObjectsByGeometry(allInstances) as DetectedObject[];
        if (deduped.length < beforeDedupe) {
          console.log(`${kindLabel}   🔄 Quadrant merge: ${beforeDedupe} → ${deduped.length}`);
        }
        console.log(`${kindLabel}   📦 Result: ${deduped.length} verified instance(s)`);
        return deduped;
      })
    );

    allObjects = results.flat();
  }

  // Phase 3: Global deduplication
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('PHASE 3: Global deduplication & scoring...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  const beforeDedupe = allObjects.length;
  allObjects = dedupeObjectsByGeometry(allObjects);
  console.log(`Deduped: ${beforeDedupe} → ${allObjects.length} objects`);

  allObjects = computeImportanceGeom(allObjects);
  console.log(`Computed importance scores for ${allObjects.length} objects`);

  // Cutouts
  let cutoutsStats: OutputPayload['cutouts'] | undefined = undefined;
  if (!config.mock && config.cutouts) {
    try {
      const result = await generateCutouts({
        imageBuffer,
        imagePath: config.imagePath,
        imageWidth,
        imageHeight,
        objects: allObjects,
        outputFile: config.outputFile,
        format: config.cutoutsFormat,
        thumbSize: config.cutoutsThumbSize,
        max: config.cutoutsMax,
        concurrency: config.cutoutsConcurrency,
        padding: config.cutoutsPadding,
      });

      cutoutsStats = {
        enabled: true,
        format: config.cutoutsFormat,
        thumb_size: config.cutoutsThumbSize,
        max: config.cutoutsMax,
        concurrency: config.cutoutsConcurrency,
        index_path: path.basename(result.indexPath),
        directory_path: path.basename(result.directoryPath),
        count: result.count,
      };
    } catch (error) {
      logErrorDetails('⚠️ Cutout generation failed. ', error);
      const paths = getCutoutsPaths(config.outputFile);
      cutoutsStats = {
        enabled: false,
        format: config.cutoutsFormat,
        thumb_size: config.cutoutsThumbSize,
        max: config.cutoutsMax,
        concurrency: config.cutoutsConcurrency,
        index_path: path.basename(paths.indexPath),
        directory_path: path.basename(paths.directoryPath),
        count: 0,
      };
    }
  }

  console.log(`\n✨ Detection Complete. Total objects: ${allObjects.length}`);

  // Phase 4: Description generation
  let descriptions: DescriptionPayload | null = null;
  if (!config.mock) {
    console.log('\n📝 Generating accessibility descriptions...');
    try {
      descriptions = await generateDescription(descriptionPool, imageBuffer, allObjects);
      console.log('\nAlt text:\n' + descriptions.alt_text);
      console.log('\nLong description:\n' + descriptions.long_description);
    } catch (error) {
      logErrorDetails('❌ Description generation failed. ', error);
    }
  }

  // Write output
  const payload: OutputPayload = {
    strategy: 'hybrid-detect-verify',
    image_path: config.imagePath,
    model_name: config.modelName,
    description_model_name: config.descriptionModelName,
    max_kinds: config.maxKinds,
    verify_rounds: config.verifyRounds,
    tile_threshold: config.tileThreshold,
    cutouts: cutoutsStats,
    kinds,
    objects: allObjects,
    descriptions,
    generated_at: new Date().toISOString(),
  };

  fs.writeFileSync(config.outputFile, JSON.stringify(payload, null, 2));
  console.log(`\n📂 Output written to ${config.outputFile}`);

  // Annotated image
  if (config.annotate) {
    try {
      await annotateImage({
        imagePath: config.imagePath,
        objects: allObjects.map((o) => ({ label: o.label, type: o.type, box_2d: normalizeBox(o.box_2d) })),
        outputPath: config.annotatedOutput,
      });
      console.log(`🖼️ Annotated image written to ${config.annotatedOutput}`);
    } catch (error) {
      logErrorDetails('⚠️ Annotation step failed. ', error);
    }
  }

  return payload;
}
