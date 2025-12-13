export type DetectedObject = {
  label: string;
  type: string;
  box_2d: number[]; // [xmin, ymin, xmax, ymax] normalized to 0–1000
  importance?: number;
  importance_geom?: number;
  importance_rank?: number | null;
  aliases?: string[];
};

export type ObjectKind = {
  kind: string;
  type: 'person' | 'animal' | 'building' | 'landscape' | 'object' | 'other';
  estimated_count?: 'few' | 'moderate' | 'many' | 'very_many';
};

export type DeepLabelPayload = {
  strategy: 'hybrid-detect-verify';
  image_path?: string;
  model_name?: string;
  description_model_name?: string;
  generated_at?: string;

  // Detection config
  max_kinds?: number;
  verify_rounds?: number;
  tile_threshold?: number;

  // Results
  kinds?: ObjectKind[];
  objects: DetectedObject[];
  descriptions?: {
    alt_text: string;
    long_description: string;
  } | null;

  // Cutouts
  cutouts?: {
    enabled: boolean;
    format: 'webp' | 'png';
    thumb_size: number;
    max: number | null;
    concurrency: number;
    index_path: string;
    directory_path: string;
    count: number;
  };
};

export type Artwork = {
  slug: string;
  title: string;
  artist?: string;
  date?: string;
  medium?: string;
  imageSrc: string;
  jsonSrc: string;
  cutoutsJsonSrc?: string;
  sourceUrl?: string;
};
