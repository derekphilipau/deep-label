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

const KindSchema = z.object({
  kinds: z.array(
    z.object({
      kind: z.string().min(1),
      type: z.enum(['person', 'animal', 'building', 'landscape', 'object', 'other']),
      estimated_count: CountEstimate,
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
type TileConfig = { rows: number; cols: number };
type TileDefinition = {
  row: number;
  col: number;
  left: number;
  top: number;
  width: number;
  height: number;
  label: string;
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
function estimateToNumber(estimate: CountEstimateType): number {
  switch (estimate) {
    case 'few': return 5;
    case 'moderate': return 18;
    case 'many': return 38;
    case 'very_many': return 75;
  }
}

function getTileConfig(instanceCount: number, threshold: number): TileConfig {
  if (threshold === 0 || instanceCount <= threshold) return { rows: 1, cols: 1 };
  const ratio = instanceCount / threshold;
  if (ratio <= 2) return { rows: 1, cols: 2 };
  if (ratio <= 4) return { rows: 2, cols: 2 };
  if (ratio <= 6) return { rows: 2, cols: 3 };
  return { rows: 3, cols: 3 };
}

const TILE_OVERLAP_PCT = 0.25;

function generateTiles(
  imageWidth: number,
  imageHeight: number,
  rows: number,
  cols: number
): TileDefinition[] {
  if (rows === 1 && cols === 1) {
    return [{
      row: 0,
      col: 0,
      left: 0,
      top: 0,
      width: imageWidth,
      height: imageHeight,
      label: 'full',
    }];
  }

  const tiles: TileDefinition[] = [];
  const baseTileWidth = imageWidth / cols;
  const baseTileHeight = imageHeight / rows;
  const overlapX = Math.round(baseTileWidth * TILE_OVERLAP_PCT);
  const overlapY = Math.round(baseTileHeight * TILE_OVERLAP_PCT);

  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      let left = Math.round(col * baseTileWidth) - (col > 0 ? overlapX : 0);
      let top = Math.round(row * baseTileHeight) - (row > 0 ? overlapY : 0);
      let right = Math.round((col + 1) * baseTileWidth) + (col < cols - 1 ? overlapX : 0);
      let bottom = Math.round((row + 1) * baseTileHeight) + (row < rows - 1 ? overlapY : 0);

      left = Math.max(0, left);
      top = Math.max(0, top);
      right = Math.min(imageWidth, right);
      bottom = Math.min(imageHeight, bottom);

      tiles.push({
        row,
        col,
        left,
        top,
        width: right - left,
        height: bottom - top,
        label: `r${row}c${col}`,
      });
    }
  }

  return tiles;
}

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

function dedupeKinds(kinds: Kind[]) {
  const seen = new Set<string>();
  const out: Kind[] = [];
  for (const k of kinds) {
    const normalizedKind = k.kind.trim().toLowerCase();
    const key = `${k.type}:${normalizedKind}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ kind: normalizedKind, type: k.type, estimated_count: k.estimated_count });
  }
  return out;
}

async function discoverKinds(pool: AIPool, imageBuffer: Buffer, maxKinds: number): Promise<Kind[]> {
  const prompt = `
You are labeling a complex artwork.

Task: list the unique OBJECT KINDS that are clearly visible in this image, with estimated counts.

Definitions:
- A "kind" is a noun phrase category (e.g., "hound", "crossbowman", "stag", "boat", "castle").
- Each kind MUST correspond to at least one clearly visible instance.

Label guidelines:
- Use SINGULAR form (e.g., "demon" not "demons", "tree" not "trees").
- Use lowercase (e.g., "hound" not "Hound").
- Prefer concise noun phrases.
- Specificity is good when visual (e.g., "crossbowman", "noblewoman", "stag", "hound").
- Do NOT invent things from patterns in water/clouds/foliage.

Estimated count categories:
- "few": 1-10 instances
- "moderate": 11-25 instances
- "many": 26-50 instances
- "very_many": 50+ instances

Output JSON only:
{"kinds":[{"kind":"","type":"","estimated_count":""}, ...]}
- kind must be singular lowercase (e.g., "demon", "tree", "person")
- type must be one of: person, animal, building, landscape, object, other
- estimated_count must be one of: few, moderate, many, very_many
- Keep the list <= ${maxKinds}.
`;

  const result = await pool.generateObject({
    schema: KindSchema,
    messages: [{ role: 'user', content: [{ type: 'text', text: prompt }, { type: 'image', image: imageBuffer }] }],
  });

  return dedupeKinds(result.kinds).slice(0, maxKinds);
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

Your tasks:

1. WRONG BOXES (wrong_indices):
   List indices of boxes that should be REMOVED because:
   - They don't actually contain a "${kind.kind}"
   - They are severely misplaced (box doesn't cover the object)
   Only flag boxes that are clearly wrong.

2. CORRECTIONS (corrections):
   For boxes that exist but need POSITION ADJUSTMENT:
   - Provide the index and corrected [xmin, ymin, xmax, ymax] in 0-1000 coords
   - Only correct if the box is notably off; minor imperfections are OK

3. MISSING (missing):
   Find any "${kind.kind}" instances that have NO box yet:
   - Provide box coordinates [xmin, ymin, xmax, ymax] for each
   - Be conservative — only add clearly visible instances
   - Do NOT hallucinate from textures/patterns

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

async function detectAndVerifyTiled(options: {
  pool: AIPool;
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
  kind: Kind;
  maxVerifyRounds: number;
  tileConfig: TileConfig;
  logPrefix?: string;
}): Promise<DetectedObject[]> {
  const { pool, imageBuffer, imageWidth, imageHeight, kind, maxVerifyRounds, tileConfig, logPrefix = '' } = options;

  const tiles = generateTiles(imageWidth, imageHeight, tileConfig.rows, tileConfig.cols);
  const totalTiles = tiles.length;

  console.log(`${logPrefix}   🔲 Tiled: ${tileConfig.rows}×${tileConfig.cols} = ${totalTiles} tiles`);

  const tileResults = await Promise.all(
    tiles.map(async (tile, i) => {
      const tileLabel = `${tile.label} (${i + 1}/${totalTiles})`;
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

  const allInstances = tileResults.flat();
  const beforeDedupe = allInstances.length;
  const deduped = dedupeObjectsByGeometry(allInstances) as DetectedObject[];
  if (deduped.length < beforeDedupe) {
    console.log(`${logPrefix}   🔄 Tile merge: ${beforeDedupe} → ${deduped.length} (removed ${beforeDedupe - deduped.length} duplicates)`);
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

  const estimatedCount = estimateToNumber(kind.estimated_count);
  const tileConfig = getTileConfig(estimatedCount, tileThreshold);

  if (tileConfig.rows > 1 || tileConfig.cols > 1) {
    console.log(`${logPrefix}  📍 Tiled detect+verify (estimated: ${kind.estimated_count})...`);
    return detectAndVerifyTiled({
      pool,
      imageBuffer,
      imageWidth,
      imageHeight,
      kind,
      maxVerifyRounds,
      tileConfig,
      logPrefix: logPrefix + '  ',
    });
  } else {
    console.log(`${logPrefix}  📍 Full image detect+verify (estimated: ${kind.estimated_count})...`);
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
}

// ==========================================
// MAIN ENTRY POINT
// ==========================================
export async function runDetection(config: DetectionConfig): Promise<OutputPayload> {
  console.log(`   Model: ${config.modelName}`);
  console.log(`   Verify rounds: ${config.verifyRounds} | Tile threshold: ${config.tileThreshold === 0 ? 'off' : `>${config.tileThreshold}`} | Concurrency: ${config.concurrency}`);

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

  const kinds: Kind[] = config.mock
    ? [
        { kind: 'Hound', type: 'animal', estimated_count: 'many' },
        { kind: 'Stag', type: 'animal', estimated_count: 'moderate' },
        { kind: 'Hunter', type: 'person', estimated_count: 'moderate' },
      ]
    : await discoverKinds(pool, imageBuffer, config.maxKinds);

  console.log(`\n🧭 Discovered ${kinds.length} kind(s):`);
  for (const k of kinds.slice(0, 60)) {
    console.log(`   • ${k.kind} (${k.type}) — ${k.estimated_count}`);
  }
  if (kinds.length > 60) {
    console.log(`   ... and ${kinds.length - 60} more`);
  }

  // Phase 2: Detect + Verify each kind
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('PHASE 2: Detecting and verifying instances per kind...');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

  let kindsToProcess = kinds;
  if (config.onlyKinds && config.onlyKinds.length > 0) {
    kindsToProcess = kinds.filter((k) => config.onlyKinds!.includes(k.kind.toLowerCase()));
    console.log(`\n⚠️  --only-kinds: processing ${kindsToProcess.length} of ${kinds.length} kinds: ${config.onlyKinds.join(', ')}`);
  }

  let allObjects: StoredObject[] = [];

  if (!config.mock) {
    const total = kindsToProcess.length;
    console.log(`\n🔀 Processing ${total} kinds (pool concurrency: ${config.concurrency})`);

    const results = await Promise.all(
      kindsToProcess.map(async (kind, i) => {
        const kindLabel = `[${i + 1}/${total}] ${kind.kind}`;
        console.log(`\n${kindLabel} (${kind.type})`);

        const instances = await detectAndVerifyKind({
          pool,
          imageBuffer,
          imageWidth,
          imageHeight,
          kind,
          maxVerifyRounds: config.verifyRounds,
          tileThreshold: config.tileThreshold,
          logPrefix: `${kindLabel} `,
        });

        console.log(`${kindLabel}   📦 Result: ${instances.length} verified instance(s)`);
        return instances;
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
