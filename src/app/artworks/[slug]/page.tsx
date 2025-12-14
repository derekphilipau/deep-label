import { notFound } from 'next/navigation';
import Link from 'next/link';
import fs from 'fs/promises';
import path from 'path';
import { ARTWORKS } from '@/lib/artworks';
import { loadArtworkPayloadFromPublic } from '@/lib/load-artwork';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import ViewerClient from './viewer-client';

type CutoutEntry = {
  index: number;
  label: string;
  type: string;
  thumb_path: string | null;
  cutout_path: string;
};

type CutoutsIndex = {
  count: number;
  entries: CutoutEntry[];
};

async function loadCutoutsIndex(cutoutsJsonSrc: string | undefined): Promise<CutoutsIndex | null> {
  if (!cutoutsJsonSrc) return null;
  try {
    const cleaned = cutoutsJsonSrc.startsWith('/') ? cutoutsJsonSrc.slice(1) : cutoutsJsonSrc;
    const fullPath = path.join(process.cwd(), 'public', cleaned);
    const raw = await fs.readFile(fullPath, 'utf-8');
    return JSON.parse(raw) as CutoutsIndex;
  } catch {
    return null;
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function boxAreaNorm(box: number[]) {
  const x1 = clamp(Math.round(box[0] ?? 0), 0, 1000);
  const y1 = clamp(Math.round(box[1] ?? 0), 0, 1000);
  const x2 = clamp(Math.round(box[2] ?? 0), 0, 1000);
  const y2 = clamp(Math.round(box[3] ?? 0), 0, 1000);
  const xmin = Math.min(x1, x2);
  const xmax = Math.max(x1, x2);
  const ymin = Math.min(y1, y2);
  const ymax = Math.max(y1, y2);
  const w = Math.max(0, xmax - xmin);
  const h = Math.max(0, ymax - ymin);
  return (w * h) / (1000 * 1000);
}

export default async function ArtworkPage({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const artwork = ARTWORKS.find((a) => a.slug === slug);
  if (!artwork) return notFound();

  const payload = await loadArtworkPayloadFromPublic(artwork.jsonSrc);
  const cutoutsIndex = await loadCutoutsIndex(artwork.cutoutsJsonSrc);

  // Get the base directory for cutout paths (e.g., /artworks/hunting-scene)
  const artworkDir = artwork.jsonSrc.replace(/\/[^/]+$/, '');

  const topObjects = payload.objects
    .map((o, index) => ({ ...o, index }))
    .sort((a, b) => {
      const ar = a.importance_rank ?? Number.POSITIVE_INFINITY;
      const br = b.importance_rank ?? Number.POSITIVE_INFINITY;
      if (ar !== br) return ar - br;
      const ai = a.importance ?? a.importance_geom ?? boxAreaNorm(a.box_2d);
      const bi = b.importance ?? b.importance_geom ?? boxAreaNorm(b.box_2d);
      return bi - ai;
    })
    .slice(0, 25);

  return (
    <main className="min-h-dvh">
      <div className="container py-6">
        <div className="flex items-center justify-between gap-4">
          <Link href="/" className="text-sm text-muted-foreground hover:underline">
            ← Back
          </Link>
          <div className="flex items-center gap-2">
            {artwork.sourceUrl ? (
              <a
                className="text-sm text-muted-foreground hover:underline"
                href={artwork.sourceUrl}
                target="_blank"
                rel="noopener noreferrer"
              >
                View source ↗
              </a>
            ) : null}
            <a
              className="text-sm text-muted-foreground hover:underline"
              href={artwork.jsonSrc}
              download
            >
              Download JSON
            </a>
            <a
              className="text-sm text-muted-foreground hover:underline"
              href={artwork.imageSrc}
              download
            >
              Download image
            </a>
          </div>
        </div>

        <div className="mt-4 grid gap-6 lg:grid-cols-[1fr_360px]">
          <div className="rounded-lg border bg-card">
            <ViewerClient imageSrc={artwork.imageSrc} objects={payload.objects} />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-xl">{artwork.title}</CardTitle>
              <div className="flex flex-wrap gap-2">
                {artwork.artist ? <Badge variant="secondary">{artwork.artist}</Badge> : null}
                {artwork.date ? <Badge variant="secondary">{artwork.date}</Badge> : null}
                <Badge variant="outline">{payload.objects.length} objects</Badge>
                {payload.kinds?.length ? (
                  <Badge variant="outline">{payload.kinds.length} kinds</Badge>
                ) : null}
                {payload.strategy ? (
                  <Badge variant="outline">{payload.strategy}</Badge>
                ) : null}
                {payload.cutouts?.enabled ? (
                  <Badge variant="outline">{payload.cutouts.count} cutouts</Badge>
                ) : null}
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {topObjects.length ? (
                <div>
                  <div className="text-xs font-semibold">Top objects</div>
                  <div className="mt-2 max-h-[280px] overflow-y-auto space-y-2 pr-1">
                    {topObjects.map((o) => (
                      <div key={o.index} className="flex items-start gap-2 text-xs">
                        <div className="mt-[2px] w-8 shrink-0 text-muted-foreground">
                          #{o.importance_rank ?? '—'}
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="truncate">{o.label}</div>
                          <div className="mt-0.5 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                            <span>{o.type}</span>
                            {typeof o.importance === 'number' ? (
                              <span>importance {o.importance.toFixed(3)}</span>
                            ) : typeof o.importance_geom === 'number' ? (
                              <span>importance {o.importance_geom.toFixed(3)}</span>
                            ) : null}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {payload.descriptions?.alt_text ? (
                <div>
                  <div className="text-xs font-semibold">Alt text</div>
                  <div className="mt-1 text-sm text-muted-foreground">
                    {payload.descriptions.alt_text}
                  </div>
                </div>
              ) : null}

              {payload.descriptions?.long_description ? (
                <div>
                  <div className="text-xs font-semibold">Long description</div>
                  <div className="mt-1 text-sm text-muted-foreground">
                    {payload.descriptions.long_description}
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>

        {cutoutsIndex && cutoutsIndex.entries.length > 0 ? (
          <div className="mt-8">
            <h2 className="text-lg font-semibold">
              Cutouts ({cutoutsIndex.count})
            </h2>
            <div className="mt-4 flex flex-wrap gap-3">
              {cutoutsIndex.entries.map((entry) => {
                const thumbSrc = entry.thumb_path
                  ? `${artworkDir}/${entry.thumb_path}`
                  : `${artworkDir}/${entry.cutout_path}`;
                const filename = entry.cutout_path.split('/').pop() || '';
                return (
                  <div key={entry.index} className="group">
                    <div className="overflow-hidden rounded-md border bg-muted">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={thumbSrc}
                        alt={entry.label}
                        className="h-24 w-auto"
                      />
                    </div>
                    <div className="mt-1 max-w-24 truncate text-[10px] text-muted-foreground" title={filename}>
                      {entry.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}
      </div>
    </main>
  );
}
