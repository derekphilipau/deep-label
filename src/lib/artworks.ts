import type { Artwork } from './types';

// Each artwork lives in its own directory: /artworks/{slug}/
// Expected files:
//   - image.jpg (or .png) - the artwork image
//   - detected_objects.json - detection payload
//   - cutouts/ - cutout images (optional)
//   - cutouts.json - cutout index (optional)

export const ARTWORKS: Artwork[] = [
  {
    slug: 'hunting-scene',
    title: 'Hunting Scene',
    artist: 'Unknown',
    date: 'Unknown',
    medium: 'Oil on canvas',
    imageSrc: '/artworks/hunting-scene/image.jpg',
    jsonSrc: '/artworks/hunting-scene/detected_objects.json',
    cutoutsJsonSrc: '/artworks/hunting-scene/cutouts.json',
  },
];

