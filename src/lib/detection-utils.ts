/**
 * Shared utilities for object detection
 * Extracted from detector.ts for reuse in recursive_detector.ts
 */
import sharp from 'sharp';

// ==========================================
// TYPES
// ==========================================

export type Box2D = [number, number, number, number]; // [xmin, ymin, xmax, ymax] normalized 0-1000

export type DetectedObject = {
  label: string;
  type: string;
  box_2d: Box2D;
};

export type StoredObject = DetectedObject & {
  importance?: number;
  importance_geom?: number;
  importance_rank?: number | null;
  aliases?: string[];
};

// ==========================================
// GEOMETRY HELPERS
// ==========================================

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function normalizeBox(box: number[]): Box2D {
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

export function normalizeObject(object: DetectedObject): DetectedObject {
  return {
    label: object.label.trim(),
    type: object.type.trim(),
    box_2d: normalizeBox(object.box_2d),
  };
}

export function getBoxArea(box: Box2D): number {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

export function intersectionArea(a: Box2D, b: Box2D): number {
  const interX1 = Math.max(a[0], b[0]);
  const interY1 = Math.max(a[1], b[1]);
  const interX2 = Math.min(a[2], b[2]);
  const interY2 = Math.min(a[3], b[3]);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  return interW * interH;
}

export function boxSimilarity(a: Box2D, b: Box2D): { iou: number; coverMin: number; areaRatio: number } {
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

// ==========================================
// LABEL/TYPE HELPERS
// ==========================================

export function normalizeType(value: string): string {
  return (value || 'other').trim().toLowerCase();
}

export function getLabelFamily(label: string): string {
  return label
    .trim()
    .toLowerCase()
    .replace(/\s+#\d+$/g, '')
    .replace(/\s+\d+$/g, '')
    .trim();
}

// ==========================================
// DEDUPLICATION
// ==========================================

export function mergeAliases(target: StoredObject, incoming: StoredObject): StoredObject {
  const set = new Set<string>();
  if (target.aliases) for (const a of target.aliases) set.add(a);
  if (incoming.aliases) for (const a of incoming.aliases) set.add(a);
  if (incoming.label && incoming.label !== target.label) set.add(incoming.label);

  const aliases = Array.from(set.values()).sort();
  return aliases.length ? { ...target, aliases } : target;
}

export function dedupeObjectsByGeometry(objects: StoredObject[]): StoredObject[] {
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

// ==========================================
// IMPORTANCE SCORING
// ==========================================

export function computeImportanceGeom(objects: StoredObject[]): StoredObject[] {
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

// ==========================================
// ERROR HANDLING
// ==========================================

const DEBUG_ERRORS =
  process.env.DEBUG_ERRORS === '1' ||
  process.env.DEBUG_ERRORS === 'true' ||
  process.env.DEBUG_ERRORS === 'yes';

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function formatError(error: unknown): string {
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

export function logErrorDetails(prefix: string, error: unknown): void {
  console.warn(prefix + formatError(error));
  if (DEBUG_ERRORS && error instanceof Error && error.stack) {
    console.warn(error.stack);
  }
}

// ==========================================
// FILE/PATH HELPERS
// ==========================================

export function sanitizeFilePart(value: string): string {
  const trimmed = value.trim().toLowerCase();
  return trimmed
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-+/g, '-')
    .replace(/^-|-$/g, '')
    .slice(0, 80);
}

// ==========================================
// CONCURRENCY HELPERS
// ==========================================

export async function runWithConcurrency<T, R>(
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
// COORDINATE TRANSFORMATION
// ==========================================

export type TileRect = {
  left: number;
  top: number;
  width: number;
  height: number;
};

/**
 * Map a box from tile-local coordinates (0-1000) to full-image coordinates (0-1000)
 */
export function mapTileBoxToFullImage(
  tileBox: Box2D,
  tile: TileRect,
  fullImageWidth: number,
  fullImageHeight: number
): Box2D {
  const [txmin, tymin, txmax, tymax] = tileBox;

  // Tile-local normalized (0-1000) -> tile pixel coords
  const pxXmin = (txmin / 1000) * tile.width;
  const pxYmin = (tymin / 1000) * tile.height;
  const pxXmax = (txmax / 1000) * tile.width;
  const pxYmax = (tymax / 1000) * tile.height;

  // Tile pixel -> full image pixel
  const fullPxXmin = tile.left + pxXmin;
  const fullPxYmin = tile.top + pxYmin;
  const fullPxXmax = tile.left + pxXmax;
  const fullPxYmax = tile.top + pxYmax;

  // Full image pixel -> full image normalized (0-1000)
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

// ==========================================
// XML/SVG HELPERS
// ==========================================

export function escapeXml(text: string): string {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

export function getTypeColorHex(type: string): string {
  const key = normalizeType(type);
  if (key === 'person') return '#e63946';
  if (key === 'animal') return '#2ec4b6';
  if (key === 'building') return '#457b9d';
  if (key === 'landscape') return '#f4a261';
  if (key === 'object') return '#9b5de5';
  return '#ffc300';
}

// ==========================================
// VERIFICATION IMAGE BUILDER
// ==========================================

export type VerificationContext = {
  kind: string;
  type: string;
};

/**
 * Build an image with numbered bounding box overlays for verification
 */
export async function buildVerificationImage(options: {
  imageBuffer: Buffer;
  imageWidth: number;
  imageHeight: number;
  instances: DetectedObject[];
  context: VerificationContext;
}): Promise<Buffer> {
  const { imageBuffer, imageWidth, imageHeight, instances, context } = options;

  const scaleX = imageWidth / 1000;
  const scaleY = imageHeight / 1000;
  const color = getTypeColorHex(context.type);
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

  const headerText = escapeXml(`Verify: ${instances.length} "${context.kind}" box(es) — check for errors & missing`);
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
