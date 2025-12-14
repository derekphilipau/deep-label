import Link from 'next/link';
import Image from 'next/image';
import { ARTWORKS } from '@/lib/artworks';
import { loadArtworkPayloadFromPublic } from '@/lib/load-artwork';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

export default async function HomePage() {
  // Load payload data for each artwork
  const artworksWithData = await Promise.all(
    ARTWORKS.map(async (artwork) => {
      try {
        const payload = await loadArtworkPayloadFromPublic(artwork.jsonSrc);
        return { artwork, payload };
      } catch {
        return { artwork, payload: null };
      }
    })
  );

  return (
    <main className="min-h-dvh">
      <div className="container py-10">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">deep-label</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Browse artworks and explore high-density bounding boxes.
          </p>
        </div>

        <div className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {artworksWithData.map(({ artwork, payload }) => (
            <Link key={artwork.slug} href={`/artworks/${artwork.slug}`}>
              <Card className="group h-full overflow-hidden transition hover:shadow-md">
                <div className="relative aspect-[4/3] overflow-hidden bg-muted">
                  <Image
                    src={artwork.imageSrc}
                    alt={artwork.title}
                    fill
                    className="object-cover transition group-hover:scale-105"
                    sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
                  />
                </div>
                <CardHeader className="pb-2">
                  <CardTitle className="line-clamp-1 text-lg">{artwork.title}</CardTitle>
                  <div className="text-sm text-muted-foreground">
                    {[artwork.artist, artwork.date].filter(Boolean).join(' • ') || '—'}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-1.5">
                    {payload ? (
                      <>
                        <Badge variant="secondary" className="text-xs">
                          {payload.objects?.length ?? 0} objects
                        </Badge>
                        {payload.kinds?.length ? (
                          <Badge variant="secondary" className="text-xs">
                            {payload.kinds.length} kinds
                          </Badge>
                        ) : null}
                        {payload.strategy ? (
                          <Badge variant="outline" className="text-xs">
                            {payload.strategy}
                          </Badge>
                        ) : null}
                      </>
                    ) : (
                      <Badge variant="outline" className="text-xs">
                        No data
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </main>
  );
}

