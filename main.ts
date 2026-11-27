import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { constants, crc32, deflateRawSync, deflateSync, inflateSync } from 'node:zlib';
import { optimiseDeflate } from './deflate.mjs';

const SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const MAGNITUDE = Uint8Array.from({ length: 256 }, (_, byte) => Math.min(byte, 256 - byte));
const BAND_HEIGHTS = [12, 24, 48];

function paeth(left, up, corner) {
  const leftDistance = Math.abs(up - corner);
  const upDistance = Math.abs(left - corner);
  const cornerDistance = Math.abs(left + up - 2 * corner);
  if (leftDistance <= upDistance && leftDistance <= cornerDistance) return left;
  return upDistance <= cornerDistance ? up : corner;
}

function readScreenshot(png) {
  if (!png.subarray(0, 8).equals(SIGNATURE) || png.toString('ascii', 12, 16) !== 'IHDR' ||
      png.readUInt32BE(8) !== 13 || !png.subarray(24, 29).equals(Buffer.from([8, 6, 0, 0, 0]))) {
    throw new Error('Expected a non-interlaced, 8-bit RGBA screenshot');
  }
  const width = png.readUInt32BE(16);
  const height = png.readUInt32BE(20);
  const stride = width * 4;
  const rowLength = stride + 1;
  const rawLength = height * rowLength;
  if (width === 0 || height === 0 || rawLength > 8 * 1024 * 1024) {
    throw new Error('Unexpected screenshot dimensions');
  }

  const compressed = [];
  let ended = false;
  for (let at = 33; at + 12 <= png.length;) {
    const length = png.readUInt32BE(at);
    const end = at + 8 + length;
    if (end + 4 > png.length) throw new Error('Truncated PNG chunk');
    const type = png.toString('ascii', at + 4, at + 8);
    if (type === 'IDAT') compressed.push(png.subarray(at + 8, end));
    if (type === 'IEND') {
      ended = true;
      break;
    }
    at = end + 4;
  }
  if (!ended || compressed.length === 0) throw new Error('Missing IDAT or IEND');
  const raw = inflateSync(Buffer.concat(compressed), { maxOutputLength: rawLength });
  if (raw.length !== rawLength) throw new Error('Unexpected scanline length');

  const rgba = Buffer.allocUnsafe(height * stride);
  for (let y = 0; y < height; y++) {
    const source = y * rowLength;
    const target = y * stride;
    const filter = raw[source];
    if (filter === 0) {
      raw.copy(rgba, target, source + 1, source + rowLength);
      continue;
    }
    if (filter > 4) throw new Error(`Invalid input filter ${filter}`);
    for (let x = 0; x < stride; x++) {
      const left = x >= 4 ? rgba[target + x - 4] : 0;
      const up = y ? rgba[target + x - stride] : 0;
      const corner = y && x >= 4 ? rgba[target + x - stride - 4] : 0;
      let predictor = left;
      if (filter === 2) predictor = up;
      if (filter === 3) predictor = (left + up) >> 1;
      if (filter === 4) predictor = paeth(left, up, corner);
      rgba[target + x] = raw[source + 1 + x] + predictor;
    }
  }
  return { width, height, rgba };
}

function pngChunk(name, data) {
  const chunk = Buffer.allocUnsafe(data.length + 12);
  chunk.writeUInt32BE(data.length, 0);
  chunk.write(name, 4, 4, 'ascii');
  data.copy(chunk, 8);
  chunk.writeUInt32BE(crc32(chunk.subarray(4, chunk.length - 4)), chunk.length - 4);
  return chunk;
}

function toPng(image, encoding, zlibStream) {
  const header = Buffer.alloc(13);
  header.writeUInt32BE(image.width, 0);
  header.writeUInt32BE(image.height, 4);
  header[8] = 8;
  header[9] = encoding.type;
  const chunks = [SIGNATURE, pngChunk('IHDR', header)];
  if (encoding.palette) chunks.push(pngChunk('PLTE', encoding.palette));
  if (encoding.transparency.length) chunks.push(pngChunk('tRNS', encoding.transparency));
  chunks.push(pngChunk('IDAT', zlibStream), pngChunk('IEND', Buffer.alloc(0)));
  return Buffer.concat(chunks);
}

function directEncoding(rgba, gray, keepAlpha, transparentRgb = -1) {
  const bpp = (gray ? 1 : 3) + Number(keepAlpha);
  const type = gray ? (keepAlpha ? 4 : 0) : (keepAlpha ? 6 : 2);
  const transparency = Buffer.alloc(transparentRgb < 0 ? 0 : bpp * 2);
  if (transparency.length) {
    transparency[1] = transparentRgb >>> 16;
    if (!gray) {
      transparency[3] = transparentRgb >>> 8;
      transparency[5] = transparentRgb;
    }
  }
  if (bpp === 4) return { bpp, type, samples: rgba, transparency };
  const samples = Buffer.allocUnsafe(rgba.length / 4 * bpp);
  for (let i = 0, j = 0; i < rgba.length; i += 4) {
    samples[j++] = rgba[i];
    if (!gray) {
      samples[j++] = rgba[i + 1];
      samples[j++] = rgba[i + 2];
    }
    if (keepAlpha) samples[j++] = rgba[i + 3];
  }
  return { bpp, type, samples, transparency };
}

function indexedEncoding(indices, colors, order) {
  const samples = Buffer.allocUnsafe(indices.length);
  const palette = Buffer.allocUnsafe(3 * order.length);
  const alpha = Buffer.allocUnsafe(order.length);
  const remap = new Uint8Array(order.length);
  let alphaLength = 0;
  for (let i = 0; i < order.length; i++) {
    const originalIndex = order[i];
    const word = colors[originalIndex];
    remap[originalIndex] = i;
    palette[i * 3] = word >>> 24;
    palette[i * 3 + 1] = word >>> 16;
    palette[i * 3 + 2] = word >>> 8;
    alpha[i] = word;
    if (alpha[i] !== 255) alphaLength = i + 1;
  }
  for (let i = 0; i < indices.length; i++) samples[i] = remap[indices[i]];
  return { bpp: 1, type: 3, samples, palette, transparency: alpha.subarray(0, alphaLength) };
}

function encodings({ rgba }) {
  let gray = true;
  let opaque = true;
  let singleTransparentColor = true;
  let transparentRgb = -1;
  let hasPalette = true;
  const paletteMap = new Map();
  const colors = [];
  const frequency = new Uint32Array(256);
  const indices = Buffer.allocUnsafe(rgba.length / 4);

  for (let i = 0, p = 0; i < rgba.length; i += 4, p++) {
    const r = rgba[i], g = rgba[i + 1], b = rgba[i + 2], a = rgba[i + 3];
    if (r !== g || g !== b) gray = false;
    const rgb = (r << 16) | (g << 8) | b;
    if (a !== 255) {
      opaque = false;
      if (a !== 0 || (transparentRgb >= 0 && rgb !== transparentRgb)) singleTransparentColor = false;
      transparentRgb = rgb;
    }
    if (hasPalette) {
      const word = (rgb << 8) | a;
      let index = paletteMap.get(word);
      if (index === undefined) {
        if (colors.length === 256) {
          hasPalette = false;
          continue;
        }
        index = colors.length;
        colors.push(word);
        paletteMap.set(word, index);
      }
      indices[p] = index;
      frequency[index]++;
    }
  }

  const options = [directEncoding(rgba, gray, !opaque)];
  if (!opaque && singleTransparentColor) {
    const r = transparentRgb >>> 16, g = (transparentRgb >>> 8) & 255, b = transparentRgb & 255;
    let conflicts = false;
    for (let i = 0; i < rgba.length; i += 4) {
      if (rgba[i + 3] === 255 && rgba[i] === r && rgba[i + 1] === g && rgba[i + 2] === b) {
        conflicts = true;
        break;
      }
    }
    if (!conflicts) options.push(directEncoding(rgba, gray, false, transparentRgb));
  }
  if (hasPalette) {
    const order = colors.map((_, index) => index);
    order.sort((a, b) => frequency[b] - frequency[a] || a - b);
    if (gray) {
      // For a dense grayscale range the direct encoding already has one byte per pixel.
      // Try the palette only when it compresses a genuinely sparse set of values.
      if (colors.length <= 128 || !opaque) options.push(indexedEncoding(indices, colors, order));
    } else {
      const frequent = indexedEncoding(indices, colors, order);
      order.sort((a, b) => (colors[a] >>> 0) - (colors[b] >>> 0));
      options.unshift(frequent, indexedEncoding(indices, colors, order));
    }
  }
  return options;
}

function filterImage({ samples, bpp }, width, height) {
  const stride = width * bpp;
  const rowLength = stride + 1;
  const filtered = Array.from({ length: 5 }, () => Buffer.allocUnsafe(height * rowLength));
  const [none, sub, up, average, paethRows] = filtered;
  const zeroRow = Buffer.alloc(stride);
  for (let y = 0; y < height; y++) {
    const row = y * rowLength;
    const start = y * stride;
    const current = samples.subarray(start, start + stride);
    const previous = y ? samples.subarray(start - stride, start) : zeroRow;
    for (let f = 0; f < 5; f++) filtered[f][row] = f;
    current.copy(none, row + 1);
    for (let x = 0, to = row + 1; x < stride; x++, to++) {
      const value = current[x];
      const left = x >= bpp ? current[x - bpp] : 0;
      const corner = x >= bpp ? previous[x - bpp] : 0;
      const above = previous[x];
      sub[to] = value - left;
      up[to] = value - above;
      average[to] = value - ((left + above) >> 1);
      paethRows[to] = value - paeth(left, above, corner);
    }
  }
  return filtered;
}

function chooseRows(filtered, rowLength, height) {
  const magnitude = new Uint8Array(height);
  const runCost = new Uint8Array(height);
  for (let y = 0; y < height; y++) {
    const start = y * rowLength + 1;
    const end = start + rowLength - 1;
    let bestMagnitude = Infinity;
    let bestRunCost = Infinity;
    for (let f = 0; f < 5; f++) {
      const data = filtered[f];
      let sum = 0;
      let cost = f === runCost[y - 1] ? 0 : 4;
      let previous = -1;
      let repeats = 0;
      for (let i = start; i < end; i++) {
        const value = data[i];
        sum += MAGNITUDE[value];
        if (value === previous) {
          if (++repeats === 258) {
            cost += 2;
            repeats = 0;
          }
        } else {
          cost += 3 + Math.min(5, MAGNITUDE[value]);
          previous = value;
          repeats = 0;
        }
      }
      if (sum < bestMagnitude) {
        bestMagnitude = sum;
        magnitude[y] = f;
      }
      if (cost < bestRunCost) {
        bestRunCost = cost;
        runCost[y] = f;
      }
    }
  }
  return [magnitude, runCost].map((choices) => {
    const raw = Buffer.allocUnsafe(height * rowLength);
    for (let y = 0; y < height; y++) {
      const start = y * rowLength;
      filtered[choices[y]].copy(raw, start, start, start + rowLength);
    }
    return raw;
  });
}

function chooseBands(candidates, rowLength, height, bandHeight, lookahead = 0) {
  // Compare actual compressed bands with the 32 KiB prefix that DEFLATE can see.
  const raw = Buffer.allocUnsafe(height * rowLength);
  for (let y = 0; y < height; y += bandHeight) {
    const start = y * rowLength;
    const end = Math.min(y + bandHeight, height) * rowLength;
    const horizon = Math.min(end + lookahead * rowLength, raw.length);
    const dictionary = raw.subarray(Math.max(0, start - 32768), start);
    let bestLength = Infinity;
    let bestBand;
    for (const candidate of candidates) {
      const size = deflateRawSync(candidate.subarray(start, horizon), { level: 4, dictionary }).length;
      if (size < bestLength) {
        bestLength = size;
        bestBand = candidate.subarray(start, end);
      }
    }
    bestBand.copy(raw, start);
  }
  return raw;
}

function compress(image, extra) {
  const finalists = [];
  for (const encoding of encodings(image)) {
    const { width, height } = image;
    const rowLength = width * encoding.bpp + 1;
    const filtered = filterImage(encoding, width, height);
    const sources = [...filtered, ...chooseRows(filtered, rowLength, height)];
    const overhead = 57 + (encoding.palette ? 12 + encoding.palette.length : 0) +
      (encoding.transparency.length ? 12 + encoding.transparency.length : 0);
    const trial = (raw) => ({ raw, stream: deflateSync(raw, { level: 6 }) });
    const ranked = sources.map(trial);
    const simpleSize = Math.min(...ranked.map(({ stream }) => stream.length)) + overhead;
    if (finalists.length && simpleSize > finalists[0].size * 1.18) continue;
    for (const rows of BAND_HEIGHTS) {
      ranked.push(trial(chooseBands(sources, rowLength, height, rows)));
    }
    if (encoding.type !== 3) ranked.push(trial(chooseBands(sources, rowLength, height, 6, 6)));
    ranked.sort((a, b) => a.stream.length - b.stream.length);
    for (const [index, candidate] of ranked.slice(0, 2).entries()) {
      const { raw, stream: quick } = candidate;
      const streams = [quick, deflateSync(raw, { level: 9 })];
      if (index === 0 && encoding.type !== 3) {
        streams.push(deflateSync(raw, { level: 8, strategy: constants.Z_FILTERED }));
      }
      let stream = quick;
      for (const next of streams) if (next.length < stream.length) stream = next;
      const alternate = streams.find((next) => !stream.equals(next));
      const size = overhead + stream.length;
      if (finalists.length === 2 && size >= finalists[1].size) continue;
      const duplicate = finalists.findIndex((other) => raw.equals(other.raw));
      const result = { size, overhead, encoding, stream, alternate, raw, rowLength };
      if (duplicate < 0) finalists.push(result);
      else if (size < finalists[duplicate].size) finalists[duplicate] = result;
      finalists.sort((a, b) => a.size - b.size);
      if (finalists.length > 2) finalists.pop();
    }
  }

  let best = finalists[0];
  for (const [rank, { raw, rowLength, encoding, stream, alternate, overhead, size }] of finalists.entries()) {
    if (stream.length * 3 >= raw.length) continue;
    if (rank) {
      // Spend a fixed amount of extra work across the set, independent of timing.
      if (raw.length > extra.remaining || size > finalists[0].size * 1.04) continue;
      extra.remaining -= raw.length;
    }
    const optimised = optimiseDeflate(raw, rowLength, encoding.bpp, best.size - overhead, stream, alternate);
    if (optimised && inflateSync(optimised, { maxOutputLength: raw.length }).equals(raw)) {
      best = { encoding, stream: optimised, size: overhead + optimised.length };
    }
  }
  return toPng(image, best.encoding, best.stream);
}

const input = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const output = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const extra = { remaining: 3 * 1024 * 1024 };
for (let index = 1; index <= 8; index++) {
  const id = `i${String(index).padStart(3, '0')}`;
  const image = readScreenshot(readFileSync(join(input, id, 'image.png')));
  const answer = compress(image, extra);
  const target = join(output, id);
  mkdirSync(target, { recursive: true });
  writeFileSync(join(target, 'image.png'), answer);
}
