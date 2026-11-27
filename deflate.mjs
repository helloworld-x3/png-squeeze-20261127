// RFC 1951 lengths and backward distances. Tokens below pack length into the
// low nine bits and distance above it. Values below 256 are literals.
const LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 259];
const DISTANCE_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 32769];
const LENGTH_EXTRA = LENGTH_BASE.slice(0, -1).map((_, i) => i === 28 ? 0 : Math.max(0, (i >> 2) - 1));
const DISTANCE_EXTRA = DISTANCE_BASE.slice(0, -1).map((_, i) => Math.max(0, (i >> 1) - 1));
const CODE_LENGTH_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function lookupCodes(bases) {
  const codes = new Uint8Array(bases.at(-1));
  for (let i = 0; i < bases.length - 1; i++) codes.fill(i, bases[i], bases[i + 1]);
  return codes;
}

const LENGTH_CODE = lookupCodes(LENGTH_BASE);
const DISTANCE_CODE = lookupCodes(DISTANCE_BASE);
const MATCH_LIMIT = 258;
const WINDOW = 32768;
const MATCH_INDEX_BITS = 18;
const MATCH_INDEX_SIZE = 1 << MATCH_INDEX_BITS;
const CHAIN = new Int32Array(WINDOW);
const HEAD = new Int32Array(MATCH_INDEX_SIZE);
const RUN_HEAD = new Int32Array(MATCH_INDEX_SIZE);
const SHORT_PREFIX_HEAD = new Int32Array(MATCH_INDEX_SIZE);
const PREFIX_HEAD = new Int32Array(MATCH_INDEX_SIZE);
let matchScratch;

function scratchFor(size) {
  if (matchScratch && matchScratch.runs.length >= size) return matchScratch;
  const capacity = 2 ** (32 - Math.clz32(size - 1));
  matchScratch = {
    runs: new Uint16Array(capacity),
    nearest: new Uint32Array(capacity),
    nearby: new Uint32Array(capacity),
    medium: new Uint32Array(capacity),
    longest: new Uint32Array(capacity),
    decisions: new Uint32Array(capacity),
    hints: new Uint16Array(capacity),
    alternateHints: new Uint16Array(capacity),
  };
  return matchScratch;
}

function matchLength(bytes, words, runs, pos, earlier, limit) {
  if (bytes[pos] !== bytes[earlier] || bytes[pos + 1] !== bytes[earlier + 1] ||
      bytes[pos + 2] !== bytes[earlier + 2]) return 0;
  const run = runs[pos];
  const earlierRun = runs[earlier];
  let length = 3;
  if (run >= 3 && earlierRun >= 3) {
    length = Math.min(run, earlierRun, limit);
    if (run !== earlierRun) return length;
  }
  while (length + 4 <= limit && words.getUint32(pos + length) === words.getUint32(earlier + length)) {
    length += 4;
  }
  while (length < limit && bytes[pos + length] === bytes[earlier + length]) length++;
  return length;
}

function findMatches(bytes, rowLength, bpp, nativeStream, alternateStream) {
  const size = bytes.length;
  const words = new DataView(bytes.buffer, bytes.byteOffset, size);
  // Searches run serially. Reuse their large tables and copy out only the
  // actual token sequences, which are usually much shorter than the input.
  const scratch = scratchFor(size);
  const { runs, nearest, nearby, medium, longest, hints, alternateHints } = scratch;
  hints.fill(0, 0, size);
  if (nativeStream) collectMatchHints(nativeStream, hints.subarray(0, size));
  if (alternateStream) collectMatchHints(alternateStream, alternateHints.fill(0, 0, size).subarray(0, size));
  let remaining = 0;
  for (let pos = size - 1; pos >= 0; pos--) {
    remaining = bytes[pos] === bytes[pos + 1] ? Math.min(MATCH_LIMIT, remaining + 1) : 1;
    runs[pos] = remaining;
  }

  const chain = CHAIN;
  const head = HEAD.fill(-1);
  const runHead = RUN_HEAD.fill(-1);
  const shortPrefixHead = SHORT_PREFIX_HEAD.fill(-1);
  const prefixHead = PREFIX_HEAD.fill(-1);
  longest.fill(0, Math.max(0, size - 2), size);
  const mask = WINDOW - 1;
  const mediumRadius = Math.max(2048, 2 * rowLength);
  for (let pos = 0; pos + 2 < size; pos++) {
    const key = (bytes[pos] << 16) | (bytes[pos + 1] << 8) | bytes[pos + 2];
    const hash = Math.imul(key, 0x9e3779b1) >>> (32 - MATCH_INDEX_BITS);
    const previousHead = head[hash];
    let previous = previousHead;
    head[hash] = pos;
    const boundary = Math.max(0, pos - WINDOW);
    const limit = Math.min(MATCH_LIMIT, size - pos);
    const run = runs[pos];

    if (run === MATCH_LIMIT && pos > 0 && bytes[pos - 1] === bytes[pos]) {
      const fill = (1 << 9) | MATCH_LIMIT;
      nearest[pos] = nearby[pos] = medium[pos] = longest[pos] = fill;
      chain[pos & mask] = previousHead;
      continue;
    }

    let maxLength = 0, maxDistance = 0;
    const previousMatch = pos ? longest[pos - 1] : 0;
    if ((previousMatch & 511) > 3) {
      // Shifting both sides of the previous match preserves every byte after
      // its first, including matches that have fallen out of the short chain.
      maxDistance = previousMatch >>> 9;
      maxLength = (previousMatch & 511) - 1;
      if (maxLength === MATCH_LIMIT - 1 && limit === MATCH_LIMIT &&
          bytes[pos + maxLength] === bytes[pos + maxLength - maxDistance]) maxLength++;
    }
    let nearLength = maxLength, nearDistance = maxDistance;
    let midLength = maxDistance <= 256 ? maxLength : 0;
    let midDistance = midLength ? maxDistance : 0;
    let mediumLength = maxDistance <= mediumRadius ? maxLength : 0;
    let mediumDistance = mediumLength ? maxDistance : 0;
    let alignedRun = -1;
    if (run >= 3 && run < MATCH_LIMIT && pos + run + 2 < size) {
      // Group starts with the same run length and following bytes. This exposes
      // split runs that a short chain of identical three-byte heads would hide.
      const suffix = (bytes[pos + run] << 16) | (bytes[pos + run + 1] << 8) | bytes[pos + run + 2];
      const runKey = suffix ^ (run << 15) ^ (bytes[pos] << 7);
      const runHash = Math.imul(runKey, 0x85ebca6b) >>> (32 - MATCH_INDEX_BITS);
      alignedRun = runHead[runHash];
      runHead[runHash] = pos;
    }

    let shortPrefix = -1, longerPrefix = -1;
    const firstWord = limit >= 5 && run < 12 ? Math.imul(words.getUint32(pos), 0x9e3779b1) : 0;
    if (limit >= 5 && run < 5) {
      const hash5 = (firstWord ^ Math.imul(bytes[pos + 4], 0x85ebca6b)) >>> (32 - MATCH_INDEX_BITS);
      shortPrefix = shortPrefixHead[hash5];
      shortPrefixHead[hash5] = pos;
    }
    if (limit >= 12 && run < 12) {
      // A longer prefix can surface matches buried in a crowded 3-byte chain.
      const hash12 = (firstWord ^
        Math.imul(words.getUint32(pos + 4), 0x85ebca6b) ^ Math.imul(words.getUint32(pos + 8), 0xc2b2ae35)) >>> (32 - MATCH_INDEX_BITS);
      longerPrefix = prefixHead[hash12];
      prefixHead[hash12] = pos;
    }

    const chainLimit = run >= 3 ? 6 : 24;
    for (let probe = alternateStream ? -8 : -7; probe < chainLimit; probe++) {
      let earlier;
      if (probe === -8) {
        const distance = alternateHints[pos];
        if (!distance || distance === hints[pos]) continue;
        earlier = pos - distance;
      } else if (probe === -7) {
        if (!hints[pos]) continue;
        earlier = pos - hints[pos];
      } else if (probe === -6) earlier = pos - bpp;
      else if (probe === -5) earlier = pos - rowLength;
      else if (probe === -4) earlier = pos - 2 * rowLength;
      else if (probe === -3) earlier = alignedRun;
      else if (probe === -2) earlier = shortPrefix;
      else if (probe === -1) earlier = longerPrefix;
      else {
        earlier = previous;
        if (earlier < boundary) break;
        previous = chain[earlier & mask];
      }
      if (earlier < boundary) continue;
      const distance = pos - earlier;
      if (probe >= 0 && maxLength === limit && distance >= maxDistance && distance >= nearDistance &&
          (distance > 256 || (midLength === limit && distance >= midDistance)) &&
          (distance > mediumRadius || (mediumLength === limit && distance >= mediumDistance))) break;
      const canImproveNear = nearDistance === 0 || distance < nearDistance;
      const canImproveMid = distance <= 256 && (distance < midDistance ||
        (midLength < limit && bytes[pos + midLength] === bytes[earlier + midLength]));
      const canImproveMedium = distance <= mediumRadius && (distance < mediumDistance ||
        (mediumLength < limit && bytes[pos + mediumLength] === bytes[earlier + mediumLength]));
      const canImproveBest = distance < maxDistance ||
        (maxLength < limit && bytes[pos + maxLength] === bytes[earlier + maxLength]);
      if (!canImproveNear && !canImproveMid && !canImproveMedium && !canImproveBest) continue;
      const length = matchLength(bytes, words, runs, pos, earlier, limit);
      if (length < 3) continue;
      if (canImproveNear) {
        nearLength = length;
        nearDistance = distance;
      }
      if (distance <= 256 && (length > midLength || (length === midLength && distance < midDistance))) {
        midLength = length;
        midDistance = distance;
      }
      if (distance <= mediumRadius && (length > mediumLength || (length === mediumLength && distance < mediumDistance))) {
        mediumLength = length;
        mediumDistance = distance;
      }
      if (length > maxLength || (length === maxLength && distance < maxDistance)) {
        maxLength = length;
        maxDistance = distance;
      }
      if (length === limit && distance === 1) break;
    }

    // Save the previous head even if probing moved farther down the chain.
    chain[pos & mask] = previousHead;
    nearest[pos] = (nearDistance << 9) | nearLength;
    nearby[pos] = midLength > nearLength ? (midDistance << 9) | midLength : nearest[pos];
    medium[pos] = mediumLength > (nearby[pos] & 511) ? (mediumDistance << 9) | mediumLength : nearby[pos];
    longest[pos] = (maxDistance << 9) | maxLength;
  }
  return scratch;
}

function greedyTokens(bytes, { longest, decisions: tokens }) {
  let count = 0;
  for (let pos = 0; pos < bytes.length;) {
    const match = longest[pos];
    const length = match & 511;
    if (length >= 3 && (length >= 64 || (longest[pos + 1] & 511) <= length + 1)) {
      tokens[count++] = match;
      pos += length;
    } else {
      tokens[count++] = bytes[pos++];
    }
  }
  return tokens.slice(0, count);
}

function countSymbols(tokens) {
  const literals = new Uint32Array(286);
  const distances = new Uint32Array(30);
  literals[256] = 1;
  let extra = 0;
  let matches = 0;
  for (const token of tokens) {
    if (token < 256) {
      literals[token]++;
    } else {
      const lc = LENGTH_CODE[token & 511];
      const dc = DISTANCE_CODE[token >>> 9];
      literals[257 + lc]++;
      distances[dc]++;
      extra += LENGTH_EXTRA[lc] + DISTANCE_EXTRA[dc];
      matches++;
    }
  }
  return { literals, distances, extra, matches };
}

const GROUP_SIZE = 1024;
const MAX_TOKENS_PER_BLOCK = 24 * GROUP_SIZE;
const COUNT_LOG_COUNT = Float64Array.from({ length: MAX_TOKENS_PER_BLOCK + 2 }, (_, count) =>
  count <= 1 ? 0 : count * Math.log2(count));

function splitBlocks(tokens, groupSize) {
  const groups = [];
  let position = 0;
  for (let start = 0; start < tokens.length; start += groupSize) {
    const group = tokens.subarray(start, start + groupSize);
    const { literals, distances, matches } = countSymbols(group);
    literals[256] = 0;
    const rawStart = position;
    for (const token of group) position += token < 256 ? 1 : token & 511;
    groups.push({ literals, distances, matches, size: group.length, rawStart });
  }

  const cost = new Float64Array(groups.length + 1).fill(Infinity);
  const previous = new Uint32Array(groups.length + 1);
  cost[0] = 0;
  for (let start = 0; start < groups.length; start++) {
    const literals = new Uint32Array(286);
    const distances = new Uint32Array(30);
    literals[256] = 1;
    let symbols = 0;
    let matches = 0;
    const limit = Math.min(groups.length, start + MAX_TOKENS_PER_BLOCK / groupSize);
    for (let end = start; end < limit; end++) {
      const group = groups[end];
      symbols += group.size;
      matches += group.matches;
      let literalEntropy = COUNT_LOG_COUNT[symbols + 1];
      let distanceEntropy = COUNT_LOG_COUNT[matches];
      let alphabet = 0;
      for (let i = 0; i < literals.length; i++) {
        literals[i] += group.literals[i];
        literalEntropy -= COUNT_LOG_COUNT[literals[i]];
        alphabet += Number(literals[i] !== 0);
      }
      for (let i = 0; i < distances.length; i++) {
        distances[i] += group.distances[i];
        distanceEntropy -= COUNT_LOG_COUNT[distances[i]];
        alphabet += Number(distances[i] !== 0);
      }
      const headerEstimate = 150 + 2 * alphabet;
      const next = cost[start] + headerEstimate + Math.max(symbols, literalEntropy) + Math.max(matches, distanceEntropy);
      if (next < cost[end + 1]) {
        cost[end + 1] = next;
        previous[end + 1] = start;
      }
    }
  }

  const blocks = [];
  for (let end = groups.length; end > 0;) {
    const start = previous[end];
    const stop = end === groups.length ? position : groups[end].rawStart;
    blocks.push({ start: groups[start].rawStart, stop, tokens: tokens.subarray(start * groupSize, end * groupSize) });
    end = start;
  }
  return blocks.reverse();
}

function huffmanLengths(frequencies, limit) {
  const alphabetSize = frequencies.length;
  const weights = new Uint32Array(alphabetSize * 2);
  weights.set(frequencies);
  const symbols = [];
  for (let i = 0; i < alphabetSize; i++) if (weights[i]) symbols.push(i);
  for (let i = 0; symbols.length < 2; i++) {
    if (weights[i] === 0) {
      weights[i] = 1;
      symbols.push(i);
    }
  }

  const parent = new Uint16Array(alphabetSize * 2);
  const depths = new Uint8Array(alphabetSize * 2);
  const take = () => {
    if (leaf < symbols.length && (branch === next || weights[symbols[leaf]] <= weights[branch])) return symbols[leaf++];
    return branch++;
  };
  let leaf, branch, next;
  for (;;) {
    symbols.sort((a, b) => weights[a] - weights[b] || a - b);
    leaf = 0;
    branch = next = alphabetSize;
    for (let merged = 1; merged < symbols.length; merged++) {
      const a = take();
      const b = take();
      weights[next] = weights[a] + weights[b];
      parent[a] = parent[b] = next++;
    }
    depths[next - 1] = 0;
    for (let i = next - 2; i >= alphabetSize; i--) depths[i] = depths[parent[i]] + 1;
    let maxDepth = 0;
    for (const symbol of symbols) {
      const depth = depths[parent[symbol]] + 1;
      depths[symbol] = depth;
      maxDepth = Math.max(maxDepth, depth);
    }
    if (maxDepth <= limit) return depths.subarray(0, alphabetSize);
    // Even weights give a balanced tree. Gradually flatten skewed histograms
    // until the same Huffman construction fits the format's depth limit.
    for (const symbol of symbols) weights[symbol] = (weights[symbol] + 1) >>> 1;
  }
}

function canonicalCodes(lengths) {
  const counts = new Uint16Array(16);
  const next = new Uint16Array(16);
  for (const length of lengths) if (length) counts[length]++;
  for (let i = 1; i < 16; i++) next[i] = (next[i - 1] + counts[i - 1]) << 1;
  return Uint16Array.from(lengths, (length) => {
    let code = next[length]++;
    let reversed = 0;
    for (let bit = 0; bit < length; bit++, code >>>= 1) reversed = (reversed << 1) | (code & 1);
    return reversed;
  });
}

function decodingTable(lengths) {
  const width = Math.max(...lengths);
  const codes = canonicalCodes(lengths);
  const entries = new Uint16Array(1 << width);
  for (let symbol = 0; symbol < lengths.length; symbol++) {
    const bits = lengths[symbol];
    if (bits === 0) continue;
    for (let i = codes[symbol]; i < entries.length; i += 1 << bits) entries[i] = (symbol << 4) | bits;
  }
  return { width, entries };
}

const FIXED_LITERALS = decodingTable(Uint8Array.from({ length: 288 }, (_, i) => i < 144 ? 8 : i < 256 ? 9 : i < 280 ? 7 : 8));
const FIXED_DISTANCES = decodingTable(new Uint8Array(32).fill(5));

function collectMatchHints(stream, hints) {
  // The native compressor tries a different match search. Import its distance
  // choices and test them alongside the byte-equality checks in findMatches.
  let bit = 16;
  const bitEnd = (stream.length - 4) * 8;
  const peek = (width) => {
    const octet = bit >>> 3;
    const word = stream[octet] | (stream[octet + 1] << 8) | (stream[octet + 2] << 16);
    return (word >>> (bit & 7)) & ((1 << width) - 1);
  };
  const advance = (width) => {
    bit += width;
    if (bit > bitEnd) throw new Error('Truncated native DEFLATE stream');
  };
  const read = (width) => {
    const value = peek(width);
    advance(width);
    return value;
  };
  const nextSymbol = (table) => {
    const entry = table.entries[peek(table.width)];
    if (!entry) throw new Error('Invalid code in native DEFLATE stream');
    advance(entry & 15);
    return entry >>> 4;
  };

  let cursor = 0;
  let last;
  do {
    last = read(1);
    const type = read(2);
    if (type === 0) {
      bit = (bit + 7) & ~7;
      const length = read(16);
      if (read(16) !== (length ^ 65535)) throw new Error('Invalid native stored block');
      advance(length * 8);
      cursor += length;
      continue;
    }

    let literals = FIXED_LITERALS;
    let distances = FIXED_DISTANCES;
    if (type === 2) {
      const literalCount = read(5) + 257;
      const distanceCount = read(5) + 1;
      const headerCount = read(4) + 4;
      const headerLengths = new Uint8Array(19);
      for (let i = 0; i < headerCount; i++) headerLengths[CODE_LENGTH_ORDER[i]] = read(3);
      const header = decodingTable(headerLengths);
      const lengths = new Uint8Array(literalCount + distanceCount);
      for (let i = 0; i < lengths.length;) {
        const code = nextSymbol(header);
        if (code < 16) {
          lengths[i++] = code;
          continue;
        }
        const count = code === 16 ? read(2) + 3 : code === 17 ? read(3) + 3 : read(7) + 11;
        if (i + count > lengths.length || (i === 0 && code === 16)) throw new Error('Invalid native code lengths');
        if (code === 16) lengths.fill(lengths[i - 1], i, i + count);
        i += count;
      }
      literals = decodingTable(lengths.subarray(0, literalCount));
      distances = decodingTable(lengths.subarray(literalCount));
    } else if (type !== 1) throw new Error('Invalid native block type');

    for (;;) {
      const literal = nextSymbol(literals);
      if (literal === 256) break;
      if (literal < 256) {
        cursor++;
        continue;
      }
      const lc = literal - 257;
      if (lc >= LENGTH_EXTRA.length) throw new Error('Invalid native match length');
      const length = LENGTH_BASE[lc] + read(LENGTH_EXTRA[lc]);
      const dc = nextSymbol(distances);
      if (dc >= DISTANCE_EXTRA.length) throw new Error('Invalid native match distance');
      const distance = DISTANCE_BASE[dc] + read(DISTANCE_EXTRA[dc]);
      if (distance > cursor || cursor + length > hints.length) throw new Error('Native match is outside the input');
      const preceding = Math.max(distance, cursor - 2);
      for (let i = cursor - 1; i >= preceding && hints[i] === 0; i--) hints[i] = distance;
      hints.fill(distance, cursor, cursor + length);
      cursor += length;
    }
  } while (!last);
  if (cursor !== hints.length) throw new Error('Native stream has the wrong input length');
}

function encodeLengthRuns(lengths) {
  const result = [];
  for (let at = 0; at < lengths.length;) {
    const value = lengths[at];
    let end = at + 1;
    while (end < lengths.length && lengths[end] === value) end++;
    let count = end - at;
    at = end;
    if (value !== 0) {
      result.push([value, 0, 0]);
      count--;
    }
    while (value === 0 && count >= 11) {
      const repeat = Math.min(138, count);
      result.push([18, repeat - 11, 7]);
      count -= repeat;
    }
    if (value === 0 && count >= 3) {
      result.push([17, count - 3, 3]);
      count = 0;
    }
    while (value !== 0 && count >= 3) {
      let repeat = Math.min(6, count);
      if (count === 7 || count === 8) repeat = count - 3;
      result.push([16, repeat - 3, 2]);
      count -= repeat;
    }
    while (count-- > 0) result.push([value, 0, 0]);
  }
  return result;
}

function describeLengthRuns(runs) {
  const runCounts = new Uint32Array(19);
  for (const [symbol] of runs) runCounts[symbol]++;
  const runLengths = huffmanLengths(runCounts, 7);
  const runCount = Math.max(4, CODE_LENGTH_ORDER.findLastIndex((code) => runLengths[code] !== 0) + 1);
  let bits = 3 * runCount;
  for (const [symbol, , width] of runs) bits += runLengths[symbol] + width;
  return { runs, runLengths, runCount, bits };
}

function optimiseLengthRuns(lengths, codeLengths) {
  const prices = Uint8Array.from(codeLengths, (bits) => bits || 5);
  const count = new Uint16Array(lengths.length);
  const symbol = new Uint8Array(lengths.length);
  const cost = new Uint16Array(lengths.length + 1);
  let run = 0;
  for (let i = lengths.length - 1; i >= 0; i--) {
    const value = lengths[i];
    run = value === lengths[i + 1] ? run + 1 : 1;
    let best = prices[value] + cost[i + 1];
    let bestCount = 1;
    let bestSymbol = value;
    for (let n = 3; n <= Math.min(run, value === 0 ? 138 : 6); n++) {
      if (n <= 6 && lengths[i - 1] === value) {
        const bits = prices[16] + 2 + cost[i + n];
        if (bits < best) [best, bestCount, bestSymbol] = [bits, n, 16];
      }
      if (value === 0) {
        const zeros = n <= 10 ? 17 : 18;
        const bits = prices[zeros] + (zeros === 17 ? 3 : 7) + cost[i + n];
        if (bits < best) [best, bestCount, bestSymbol] = [bits, n, zeros];
      }
    }
    cost[i] = best;
    count[i] = bestCount;
    symbol[i] = bestSymbol;
  }
  const runs = [];
  for (let i = 0; i < lengths.length; i += count[i]) {
    const code = symbol[i];
    if (code < 16) runs.push([code, 0, 0]);
    else if (code === 18) runs.push([18, count[i] - 11, 7]);
    else runs.push([code, count[i] - 3, code === 16 ? 2 : 3]);
  }
  return runs;
}

function describeBlock(tokens, stats) {
  const lit = huffmanLengths(stats.literals, 15);
  const dist = huffmanLengths(stats.distances, 15);
  const litCount = lit.findLastIndex(Boolean) + 1;
  const distCount = dist.findLastIndex(Boolean) + 1;
  const lengths = [...lit.subarray(0, litCount), ...dist.subarray(0, distCount)];
  const initial = describeLengthRuns(encodeLengthRuns(lengths));
  const improved = describeLengthRuns(optimiseLengthRuns(lengths, initial.runLengths));
  const header = improved.bits < initial.bits ? improved : initial;
  let bits = 17 + header.bits + stats.extra;
  for (let i = 0; i < litCount; i++) bits += stats.literals[i] * lit[i];
  for (let i = 0; i < distCount; i++) bits += stats.distances[i] * dist[i];
  return { tokens, lit, dist, litCount, distCount, ...header, bits };
}

const RING = 512;
const RING_MASK = RING - 1;
const COST_UNIT = 8;
const DISTANCE_PRICES = new Uint32Array(WINDOW + 1);
const MINIMA = new Float64Array(9 * RING);

function estimatedCosts(counts, total, lengths, exact) {
  return Uint16Array.from(counts, (frequency, i) => {
    if (exact && frequency) return lengths[i] * COST_UNIT;
    const entropy = Math.log2((total + counts.length) / (1 + frequency));
    const estimate = lengths[i] && frequency ? (lengths[i] + Math.min(16, entropy)) / 2 : Math.min(16, entropy);
    return Math.max(1, Math.round(estimate * COST_UNIT));
  });
}

function lengthPriceRanges(prices) {
  const ends = new Uint16Array(259);
  const costs = new Uint32Array(259);
  let maxWidth = 1;
  for (let first = 3; first <= MATCH_LIMIT;) {
    const price = prices[LENGTH_CODE[first]];
    let end = first + 1;
    while (end <= MATCH_LIMIT && prices[LENGTH_CODE[end]] === price) end++;
    ends.fill(end - 1, first, end);
    costs.fill(price, first, end);
    maxWidth = Math.max(maxWidth, end - first);
    first = end;
  }
  return { ends, costs, layers: 32 - Math.clz32(maxWidth) };
}

function optimiseTokens(bytes, matches, start, stop, seed, stats, exact) {
  const literal = estimatedCosts(stats.literals, seed.tokens.length + 1, seed.lit, exact);
  const distance = estimatedCosts(stats.distances, stats.matches, seed.dist, exact);
  const length = lengthPriceRanges(LENGTH_EXTRA.map((extra, i) => RING * (literal[257 + i] + COST_UNIT * extra)));
  const distanceCost = DISTANCE_PRICES;
  for (let code = 0; code < DISTANCE_EXTRA.length; code++) {
    distanceCost.fill(RING * (distance[code] + COST_UNIT * DISTANCE_EXTRA[code]), DISTANCE_BASE[code], DISTANCE_BASE[code + 1]);
  }
  const chosen = matches.decisions;
  const minima = MINIMA.fill(Infinity, 0, length.layers * RING);
  minima[stop & RING_MASK] = stop & RING_MASK;

  for (let pos = stop - 1; pos >= start; pos--) {
    const slot = pos & RING_MASK;
    let best = RING * literal[bytes[pos]] + minima[(slot + 1) & RING_MASK];
    let choice = bytes[pos];
    const maxMatch = matches.longest[pos];
    const upper = Math.min(maxMatch & 511, stop - pos);
    if (upper >= 3) {
      const mediumMatch = matches.medium[pos];
      const midMatch = matches.nearby[pos];
      const nearMatch = matches.nearest[pos];
      const furthest = maxMatch >>> 9;
      const mediumDist = mediumMatch >>> 9;
      const midDist = midMatch >>> 9;
      const nearDist = nearMatch >>> 9;
      const mediumLimit = Math.min(mediumMatch & 511, upper);
      const midLimit = Math.min(midMatch & 511, mediumLimit);
      const nearLimit = Math.min(nearMatch & 511, midLimit);
      const bestMedium = distanceCost[mediumDist] <= distanceCost[furthest] ? mediumDist : furthest;
      const bestMid = distanceCost[midDist] <= distanceCost[bestMedium] ? midDist : bestMedium;
      const bestNear = distanceCost[nearDist] <= distanceCost[bestMid] ? nearDist : bestMid;

      // One distance price applies to each interval of reachable lengths.
      // Adjacent lengths with the same price can share a query for the
      // cheapest suffix in that range in constant time using rolling minima.
      for (let from = 3; from <= upper;) {
        let to, dist;
        if (from <= nearLimit && bestNear !== bestMid) {
          to = nearLimit;
          dist = bestNear;
        } else if (from <= midLimit && bestMid !== bestMedium) {
          to = midLimit;
          dist = bestMid;
        } else if (from <= mediumLimit && bestMedium !== furthest) {
          to = mediumLimit;
          dist = bestMedium;
        } else {
          to = upper;
          dist = furthest;
        }
        const distBits = distanceCost[dist];
        while (from <= to) {
          const end = Math.min(to, length.ends[from]);
          const power = 31 - Math.clz32(end - from + 1);
          const layer = power * RING;
          const left = minima[layer + ((slot + from) & RING_MASK)];
          const right = minima[layer + ((slot + end + 1 - (1 << power)) & RING_MASK)];
          const endpoint = Math.min(left, right);
          const value = endpoint + length.costs[from] + distBits;
          if (value < best) {
            best = value;
            choice = (dist << 9) | (((endpoint & RING_MASK) - slot) & RING_MASK);
          }
          from = end + 1;
        }
      }
    }
    chosen[pos - start] = choice;
    // The low bits retain the winning suffix's index for reconstructing tokens.
    minima[slot] = best - (best & RING_MASK) + slot;
    for (let layer = 1, half = 1; layer < length.layers; layer++, half <<= 1) {
      const base = (layer - 1) * RING;
      minima[base + RING + slot] = Math.min(minima[base + slot], minima[base + ((slot + half) & RING_MASK)]);
    }
  }

  // Output indices stay at or behind the next decision, so compact in place.
  let count = 0;
  for (let at = 0; at < stop - start;) {
    const token = chosen[at];
    chosen[count++] = token;
    at += token < 256 ? 1 : token & 511;
  }
  return chosen.slice(0, count);
}

class BitWriter {
  constructor(capacity) {
    this.output = Buffer.allocUnsafe(capacity);
    this.at = 2;
    this.pending = 0;
    this.bits = 0;
    this.output[0] = 0x78;
    this.output[1] = 0xda;
  }

  write(bits, width) {
    this.bits |= bits << this.pending;
    this.pending += width;
    while (this.pending >= 8) {
      this.output[this.at++] = this.bits;
      this.bits >>>= 8;
      this.pending -= 8;
    }
  }

  finish(bytes) {
    if (this.pending) this.output[this.at++] = this.bits;
    let a = 1, b = 0;
    for (let start = 0; start < bytes.length; start += 4096) {
      const end = Math.min(start + 4096, bytes.length);
      for (let i = start; i < end; i++) {
        a += bytes[i];
        b += a;
      }
      a %= 65521;
      b %= 65521;
    }
    this.output.writeUInt32BE((b * 65536) + a, this.at);
    return this.output.subarray(0, this.at + 4);
  }
}

function writeBlock(writer, block, final) {
  const { tokens, lit, dist, litCount, distCount, runs, runLengths, runCount } = block;
  const litCodes = canonicalCodes(lit);
  const distCodes = canonicalCodes(dist);
  const runCodes = canonicalCodes(runLengths);
  writer.write(Number(final) | 4, 3);
  writer.write(litCount - 257, 5);
  writer.write(distCount - 1, 5);
  writer.write(runCount - 4, 4);
  for (let i = 0; i < runCount; i++) writer.write(runLengths[CODE_LENGTH_ORDER[i]], 3);
  for (const [symbol, extra, width] of runs) {
    writer.write(runCodes[symbol], runLengths[symbol]);
    writer.write(extra, width);
  }
  for (const token of tokens) {
    if (token < 256) {
      writer.write(litCodes[token], lit[token]);
    } else {
      const length = token & 511;
      const distance = token >>> 9;
      const lc = LENGTH_CODE[length];
      const dc = DISTANCE_CODE[distance];
      writer.write(litCodes[257 + lc], lit[257 + lc]);
      writer.write(length - LENGTH_BASE[lc], LENGTH_EXTRA[lc]);
      writer.write(distCodes[dc], dist[dc]);
      writer.write(distance - DISTANCE_BASE[dc], DISTANCE_EXTRA[dc]);
    }
  }
  writer.write(litCodes[256], lit[256]);
}

function refineBoundaries(tokens, blocks) {
  let start = 0;
  for (let i = 0; i + 1 < blocks.length; i++) {
    const cut = start + blocks[i].tokens.length;
    const stop = cut + blocks[i + 1].tokens.length;
    let bits = blocks[i].bits + blocks[i + 1].bits;
    for (const shift of [-256, -128, 128, 256]) {
      const boundary = cut + shift;
      if (boundary - start < 128 || stop - boundary < 128) continue;
      const first = tokens.subarray(start, boundary);
      const second = tokens.subarray(boundary, stop);
      const left = describeBlock(first, countSymbols(first));
      const right = describeBlock(second, countSymbols(second));
      if (left.bits + right.bits < bits) {
        bits = left.bits + right.bits;
        blocks[i] = left;
        blocks[i + 1] = right;
      }
    }
    start += blocks[i].tokens.length;
  }
}

export function optimiseDeflate(bytes, rowLength, bpp, maxBytes, passes, nativeStream, alternateStream) {
  if (bytes.length === 0 || maxBytes < 7) return null;
  const matches = findMatches(bytes, rowLength, bpp, nativeStream, alternateStream);
  const greedy = greedyTokens(bytes, matches);
  const initial = splitBlocks(greedy, GROUP_SIZE);
  let blocks = initial.map(({ start, stop, tokens }) => {
    let stats = countSymbols(tokens);
    let seed = describeBlock(tokens, stats);
    let best = seed;
    for (let iteration = 0; iteration < passes; iteration++) {
      const exact = iteration === 2;
      if (exact && best !== seed) {
        seed = best;
        stats = countSymbols(seed.tokens);
      }
      const optimised = optimiseTokens(bytes, matches, start, stop, seed, stats, exact);
      stats = countSymbols(optimised);
      seed = describeBlock(optimised, stats);
      if (seed.bits < best.bits) best = seed;
    }
    return best;
  });
  let bits = blocks.reduce((sum, block) => sum + block.bits, 0);

  // The final token distribution can move useful boundaries away from those
  // selected for the initial greedy parse. Distance history spans both layouts.
  const tokenCount = blocks.reduce((sum, block) => sum + block.tokens.length, 0);
  if (tokenCount >= 2 * GROUP_SIZE) {
    const tokens = new Uint32Array(tokenCount);
    let offset = 0;
    for (const block of blocks) {
      tokens.set(block.tokens, offset);
      offset += block.tokens.length;
    }
    const repartitioned = splitBlocks(tokens, GROUP_SIZE / 2).map(({ tokens }) => describeBlock(tokens, countSymbols(tokens)));
    const updatedBits = repartitioned.reduce((sum, block) => sum + block.bits, 0);
    if (updatedBits < bits) {
      blocks = repartitioned;
    }
    refineBoundaries(tokens, blocks);
    bits = blocks.reduce((sum, block) => sum + block.bits, 0);
  }

  if (Math.ceil(bits / 8) + 6 >= maxBytes) return null;
  const writer = new BitWriter(maxBytes);
  for (let i = 0; i < blocks.length; i++) {
    writeBlock(writer, blocks[i], i === blocks.length - 1);
  }
  return writer.finish(bytes);
}
