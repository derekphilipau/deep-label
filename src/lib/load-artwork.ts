import fs from 'fs/promises';
import path from 'path';
import type { DeepLabelPayload } from './types';

export async function loadArtworkPayloadFromPublic(jsonSrc: string) {
  const cleaned = jsonSrc.startsWith('/') ? jsonSrc.slice(1) : jsonSrc;
  const fullPath = path.join(process.cwd(), 'public', cleaned);
  const raw = await fs.readFile(fullPath, 'utf-8');
  return JSON.parse(raw) as DeepLabelPayload;
}

