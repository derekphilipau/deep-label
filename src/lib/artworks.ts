import fs from 'fs';
import path from 'path';
import type { Artwork } from './types';

// Auto-discover artworks from public/artworks/{slug}/
// Each artwork directory should contain:
//   - artwork.json - metadata (title, artist, date, etc.)
//   - image.jpg (or .png) - the artwork image
//   - detected_objects.json - detection payload (optional)
//   - cutouts/ - cutout images (optional)
//   - cutouts.json - cutout index (optional)

type ArtworkMeta = {
  title?: string;
  artist?: string;
  date?: string;
  medium?: string;
  url?: string;
};

function findImageFile(artworkDir: string): string | null {
  const candidates = ['image.jpg', 'image.jpeg', 'image.png'];
  for (const name of candidates) {
    if (fs.existsSync(path.join(artworkDir, name))) {
      return name;
    }
  }
  return null;
}

function discoverArtworks(): Artwork[] {
  const artworksDir = path.join(process.cwd(), 'public', 'artworks');

  if (!fs.existsSync(artworksDir)) {
    return [];
  }

  const slugs = fs.readdirSync(artworksDir).filter((name) => {
    const fullPath = path.join(artworksDir, name);
    return fs.statSync(fullPath).isDirectory() && !name.startsWith('.');
  });

  const artworks: Artwork[] = [];

  for (const slug of slugs) {
    const artworkDir = path.join(artworksDir, slug);
    const metaPath = path.join(artworkDir, 'artwork.json');

    // Find image file
    const imageFile = findImageFile(artworkDir);
    if (!imageFile) continue; // Skip directories without images

    // Load metadata
    let meta: ArtworkMeta = {};
    if (fs.existsSync(metaPath)) {
      try {
        meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
      } catch {
        // Invalid JSON, use defaults
      }
    }

    // Check for optional files
    const hasDetections = fs.existsSync(path.join(artworkDir, 'detected_objects.json'));
    const hasCutouts = fs.existsSync(path.join(artworkDir, 'cutouts.json'));

    artworks.push({
      slug,
      title: meta.title || slug.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
      artist: meta.artist,
      date: meta.date,
      medium: meta.medium,
      sourceUrl: meta.url,
      imageSrc: `/artworks/${slug}/${imageFile}`,
      jsonSrc: hasDetections ? `/artworks/${slug}/detected_objects.json` : '',
      cutoutsJsonSrc: hasCutouts ? `/artworks/${slug}/cutouts.json` : undefined,
    });
  }

  // Sort by title
  return artworks.sort((a, b) => a.title.localeCompare(b.title));
}

// Cache the result at module load time (server-side)
export const ARTWORKS: Artwork[] = discoverArtworks();
