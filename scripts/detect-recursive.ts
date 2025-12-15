#!/usr/bin/env npx tsx
/**
 * CLI wrapper for recursive artwork object detection (v2)
 */
import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import { RecursiveDetector, RecursiveConfig } from '../src/lib/recursive_detector';

dotenv.config();

const ARTWORKS_DIR = 'public/artworks';

// Defaults from environment
const DEFAULT_MAX_DEPTH = Number(process.env.MAX_DEPTH || 3);
const DEFAULT_MIN_TILE_SIZE = Number(process.env.MIN_TILE_SIZE || 256);
const DEFAULT_VERIFY_ROUNDS = Number(process.env.VERIFY_ROUNDS || 2);
const DEFAULT_CUTOUTS = process.env.CUTOUTS !== '0' && process.env.CUTOUTS !== 'false';
const DEFAULT_CUTOUTS_FORMAT = (process.env.CUTOUTS_FORMAT || 'webp') === 'png' ? 'png' : 'webp';
const DEFAULT_CUTOUTS_THUMB_SIZE = Number(process.env.CUTOUTS_THUMB_SIZE || 256);
const DEFAULT_CUTOUTS_MAX = Number(process.env.CUTOUTS_MAX || 100);
const DEFAULT_CUTOUTS_CONCURRENCY = Number(process.env.CUTOUTS_CONCURRENCY || 8);
const DEFAULT_CUTOUTS_PADDING = Number(process.env.CUTOUTS_PADDING || 0.10);
const DEFAULT_CONCURRENCY = Number(process.env.CONCURRENCY || 6);
const DEFAULT_MODEL_NAME = process.env.MODEL_NAME || 'gemini-3-pro-preview';
const DEFAULT_DESCRIPTION_MODEL_NAME = process.env.DESCRIPTION_MODEL_NAME || 'gemini-3-pro-preview';

function printHelp() {
  console.log(`
Usage: npx tsx scripts/detect-recursive.ts <artwork-slug> [options]
       npx tsx scripts/detect-recursive.ts -i <image> -o <output> [options]

Recursive detector v2: True N-level recursive detection with verification.

Artwork mode (recommended):
  Create a folder in ${ARTWORKS_DIR}/<slug>/ with an image.jpg, then run:
    npx tsx scripts/detect-recursive.ts hunting-scene

Custom paths mode:
  npx tsx scripts/detect-recursive.ts -i path/to/image.jpg -o path/to/output.json

Options:
  -i, --image <path>          Input image path
  -o, --output <path>         Output JSON path
      --force                 Overwrite existing output files
      --max-depth <n>         Maximum recursion depth (default: ${DEFAULT_MAX_DEPTH})
      --min-tile-size <px>    Minimum tile size in pixels (default: ${DEFAULT_MIN_TILE_SIZE})
      --verify-rounds <n>     Verification rounds per detection (default: ${DEFAULT_VERIFY_ROUNDS})
      --concurrency <n>       Max concurrent API calls (default: ${DEFAULT_CONCURRENCY})
      --no-cutouts            Disable cutout generation
      --cutouts-format <fmt>  Cutout format: webp|png (default: ${DEFAULT_CUTOUTS_FORMAT})
      --cutouts-thumb-size <n> Thumbnail height (default: ${DEFAULT_CUTOUTS_THUMB_SIZE})
      --cutouts-padding <pct> Padding as decimal (default: ${DEFAULT_CUTOUTS_PADDING})
      --cutouts-max <n>       Max cutouts to generate (default: ${DEFAULT_CUTOUTS_MAX})
      --model <name>          Detection model (default: ${DEFAULT_MODEL_NAME})
      --no-annotate           Skip annotated image generation
      --no-descriptions       Skip description generation
  -h, --help                  Show help
`);
}

function requireValue(args: string[], index: number, flag: string): string {
  const value = args[index + 1];
  if (!value || value.startsWith('-')) {
    console.error(`❌ Missing value for ${flag}`);
    process.exit(1);
  }
  return value;
}

function findArtworkImage(artworkDir: string): string | null {
  const candidates = ['image.jpg', 'image.jpeg', 'image.png'];
  for (const name of candidates) {
    const p = path.join(artworkDir, name);
    if (fs.existsSync(p)) return p;
  }
  try {
    const files = fs.readdirSync(artworkDir);
    for (const f of files) {
      if (/\.(jpg|jpeg|png)$/i.test(f)) {
        return path.join(artworkDir, f);
      }
    }
  } catch {
    // Directory doesn't exist
  }
  return null;
}

type CLIConfig = RecursiveConfig & { slug: string | null; force: boolean };

function parseArgs(): CLIConfig {
  const args = process.argv.slice(2);
  const config: CLIConfig = {
    slug: null,
    imagePath: '',
    outputFile: '',
    maxDepth: DEFAULT_MAX_DEPTH,
    minTileSize: DEFAULT_MIN_TILE_SIZE,
    verifyRounds: DEFAULT_VERIFY_ROUNDS,
    concurrency: DEFAULT_CONCURRENCY,
    modelName: DEFAULT_MODEL_NAME,
    annotate: true,
    annotatedOutput: '',
    generateDescriptions: true,
    descriptionModelName: DEFAULT_DESCRIPTION_MODEL_NAME,
    cutouts: DEFAULT_CUTOUTS,
    cutoutsFormat: DEFAULT_CUTOUTS_FORMAT as 'webp' | 'png',
    cutoutsThumbSize: DEFAULT_CUTOUTS_THUMB_SIZE,
    cutoutsMax: DEFAULT_CUTOUTS_MAX,
    cutoutsConcurrency: DEFAULT_CUTOUTS_CONCURRENCY,
    cutoutsPadding: DEFAULT_CUTOUTS_PADDING,
    force: false,
  };

  const positionalArgs: string[] = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
      case '-i':
      case '--image':
        config.imagePath = requireValue(args, i, arg);
        i++;
        break;
      case '-o':
      case '--output':
        config.outputFile = requireValue(args, i, arg);
        i++;
        break;
      case '--annotated-output':
        config.annotatedOutput = requireValue(args, i, arg);
        i++;
        break;
      case '--force':
        config.force = true;
        break;
      case '--max-depth': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 1 || value > 10) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]} (must be 1-10)`);
          process.exit(1);
        }
        config.maxDepth = value;
        i++;
        break;
      }
      case '--min-tile-size': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 64 || value > 2048) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]} (must be 64-2048)`);
          process.exit(1);
        }
        config.minTileSize = value;
        i++;
        break;
      }
      case '--verify-rounds': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 0 || value > 10) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]} (must be 0-10)`);
          process.exit(1);
        }
        config.verifyRounds = value;
        i++;
        break;
      }
      case '--concurrency': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 1 || value > 20) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]} (must be 1-20)`);
          process.exit(1);
        }
        config.concurrency = value;
        i++;
        break;
      }
      case '--cutouts':
        config.cutouts = true;
        break;
      case '--no-cutouts':
        config.cutouts = false;
        break;
      case '--cutouts-format': {
        const value = requireValue(args, i, arg);
        if (value !== 'webp' && value !== 'png') {
          console.error(`❌ Invalid value for ${arg}: ${value} (must be webp or png)`);
          process.exit(1);
        }
        config.cutoutsFormat = value;
        i++;
        break;
      }
      case '--cutouts-thumb-size': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 0) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]}`);
          process.exit(1);
        }
        config.cutoutsThumbSize = value;
        i++;
        break;
      }
      case '--cutouts-max': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 1) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]}`);
          process.exit(1);
        }
        config.cutoutsMax = value;
        i++;
        break;
      }
      case '--cutouts-padding': {
        const value = Number(requireValue(args, i, arg));
        if (!Number.isFinite(value) || value < 0 || value > 1) {
          console.error(`❌ Invalid value for ${arg}: ${args[i + 1]} (must be 0-1)`);
          process.exit(1);
        }
        config.cutoutsPadding = value;
        i++;
        break;
      }
      case '--model':
        config.modelName = requireValue(args, i, arg);
        i++;
        break;
      case '--description-model':
        config.descriptionModelName = requireValue(args, i, arg);
        i++;
        break;
      case '--no-annotate':
        config.annotate = false;
        break;
      case '--annotate':
        config.annotate = true;
        break;
      case '--no-descriptions':
        config.generateDescriptions = false;
        break;
      case '--descriptions':
        config.generateDescriptions = true;
        break;
      case '-h':
      case '--help':
        printHelp();
        process.exit(0);
      default:
        if (arg.startsWith('-')) {
          console.warn(`⚠️ Unknown argument ignored: ${arg}`);
        } else {
          positionalArgs.push(arg);
        }
    }
  }

  // Process artwork slug if provided
  if (positionalArgs.length > 0 && !config.imagePath) {
    const slug = positionalArgs[0];
    const artworkDir = path.join(ARTWORKS_DIR, slug);

    if (!fs.existsSync(artworkDir)) {
      console.error(`❌ Artwork directory not found: ${artworkDir}`);
      console.error(`   Create the directory and add an image.jpg file.`);
      process.exit(1);
    }

    const imagePath = findArtworkImage(artworkDir);
    if (!imagePath) {
      console.error(`❌ No image found in ${artworkDir}`);
      console.error(`   Add an image.jpg, image.png, or any .jpg/.png file.`);
      process.exit(1);
    }

    config.slug = slug;
    config.imagePath = imagePath;
    config.outputFile = path.join(artworkDir, 'detected_objects.json');
    config.annotatedOutput = path.join(artworkDir, 'annotated.png');

    // Check for existing outputs
    const existingFiles: string[] = [];
    if (fs.existsSync(config.outputFile)) existingFiles.push('detected_objects.json');
    if (fs.existsSync(config.annotatedOutput)) existingFiles.push('annotated.png');
    const cutoutsDir = path.join(artworkDir, 'cutouts');
    if (fs.existsSync(cutoutsDir)) existingFiles.push('cutouts/');

    if (existingFiles.length > 0 && !config.force) {
      console.error(`❌ Output files already exist in ${artworkDir}:`);
      for (const f of existingFiles) {
        console.error(`   - ${f}`);
      }
      console.error(`   Use --force to overwrite.`);
      process.exit(1);
    }
  }

  // Validate required paths
  if (!config.imagePath) {
    console.error(`❌ No input specified.`);
    console.error(`   Usage: npx tsx scripts/detect-recursive.ts <artwork-slug>`);
    console.error(`   Or:    npx tsx scripts/detect-recursive.ts -i <image> -o <output>`);
    process.exit(1);
  }

  // Set defaults for output paths
  if (!config.outputFile) {
    config.outputFile = 'detected_objects.json';
  }
  if (!config.annotatedOutput) {
    config.annotatedOutput = config.outputFile.replace(/\.json$/, '.annotated.png');
  }

  return config;
}

async function main() {
  const cliConfig = parseArgs();

  if (cliConfig.slug) {
    console.log(`🎨 Processing artwork: ${cliConfig.slug}`);
    console.log(`   Input:  ${cliConfig.imagePath}`);
    console.log(`   Output: ${path.dirname(cliConfig.outputFile)}/`);
  } else {
    console.log(`🚀 Starting recursive detection...`);
    console.log(`   Input:  ${cliConfig.imagePath}`);
    console.log(`   Output: ${cliConfig.outputFile}`);
  }

  console.log(`   Max Depth: ${cliConfig.maxDepth}, Min Tile: ${cliConfig.minTileSize}px, Verify Rounds: ${cliConfig.verifyRounds}`);

  // Build RecursiveConfig (exclude CLI-only fields)
  const { slug, force, ...config } = cliConfig;

  try {
    const detector = new RecursiveDetector(config);
    await detector.run();
    console.log(`\n🎉 Done!`);
  } catch (error) {
    console.error(`❌ ${error instanceof Error ? error.message : String(error)}`);
    if (error instanceof Error && error.stack) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

main();
