import sharp from 'sharp';
import path from 'path';

export type Box2D = [number, number, number, number];
export type AnnotatedObject = {
  label: string;
  type: string;
  box_2d: Box2D;
};

const COLORS: Record<string, string> = {
  person: '#e63946',
  animal: '#2ec4b6',
  building: '#457b9d',
  landscape: '#f4a261',
  object: '#9b5de5',
  other: '#ffc300',
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function normalizeBox(box: Box2D): Box2D {
  const [x1, y1, x2, y2] = box.map((v) =>
    clamp(Math.round(v), 0, 1000)
  ) as Box2D;
  const xmin = Math.min(x1, x2);
  const xmax = Math.max(x1, x2);
  const ymin = Math.min(y1, y2);
  const ymax = Math.max(y1, y2);
  return [xmin, ymin, xmax, ymax];
}

function escapeXml(text: string) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function getColor(type: string) {
  const key = type.trim().toLowerCase();
  return COLORS[key] || COLORS.other;
}

function buildSvg(
  width: number,
  height: number,
  objects: AnnotatedObject[]
) {
  const thickness = Math.max(2, Math.round(Math.min(width, height) / 400));
  const fontSize = Math.max(12, Math.round(Math.min(width, height) / 70));
  const scaleX = width / 1000;
  const scaleY = height / 1000;

  const elements = objects
    .map((obj) => {
      const label = obj.label?.trim() || 'unknown';
      const type = obj.type?.trim() || 'other';
      const [xmin, ymin, xmax, ymax] = normalizeBox(obj.box_2d);

      const x = xmin * scaleX;
      const y = ymin * scaleY;
      const w = (xmax - xmin) * scaleX;
      const h = (ymax - ymin) * scaleY;

      const color = getColor(type);
      const text = escapeXml(`${label} (${type})`);
      const approxTextWidth = text.length * fontSize * 0.6 + 8;
      const approxTextHeight = fontSize + 8;

      const textX = clamp(x, 0, width - 1);
      const textY =
        y - approxTextHeight - 2 >= 0 ? y - approxTextHeight - 2 : y + 2;
      const bgX = clamp(textX, 0, width - approxTextWidth);
      const bgY = clamp(textY, 0, height - approxTextHeight);

      return `
  <rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" fill="none" stroke="${color}" stroke-width="${thickness}" />
  <rect x="${bgX.toFixed(1)}" y="${bgY.toFixed(1)}" width="${approxTextWidth.toFixed(1)}" height="${approxTextHeight.toFixed(1)}" fill="rgba(0,0,0,0.7)" />
  <text x="${(bgX + 4).toFixed(1)}" y="${(bgY + 4).toFixed(1)}" font-size="${fontSize}" font-family="system-ui, -apple-system, Segoe UI, sans-serif" fill="#ffffff" dominant-baseline="hanging">${text}</text>
`;
    })
    .join('\n');

  return `
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
${elements}
</svg>
`.trim();
}

function applyOutputFormat(
  pipeline: sharp.Sharp,
  outputPath: string
): sharp.Sharp {
  const ext = path.extname(outputPath).toLowerCase();
  if (ext === '.jpg' || ext === '.jpeg') return pipeline.jpeg();
  if (ext === '.webp') return pipeline.webp();
  if (ext === '.avif') return pipeline.avif();
  return pipeline.png();
}

export async function annotateImage(options: {
  imagePath: string;
  objects: AnnotatedObject[];
  outputPath: string;
}) {
  const base = sharp(options.imagePath);
  const metadata = await base.metadata();

  if (!metadata.width || !metadata.height) {
    throw new Error('Unable to read image dimensions.');
  }

  if (!options.objects || options.objects.length === 0) {
    const pipeline = applyOutputFormat(sharp(options.imagePath), options.outputPath);
    await pipeline.toFile(options.outputPath);
    return;
  }

  const svg = buildSvg(metadata.width, metadata.height, options.objects);
  const overlay = Buffer.from(svg);
  const pipeline = applyOutputFormat(
    base.composite([{ input: overlay, top: 0, left: 0 }]),
    options.outputPath
  );

  await pipeline.toFile(options.outputPath);
}

