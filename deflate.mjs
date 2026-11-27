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
const MAX_TOKENS = 245_000;
const MAX_POSITIONS = 285_000;
const MAX_EDGES = 2_750_000;
const MATCH_OPTIONS = 12;
const HASH_SIZE = 1 << 16;

function tokenPositions(tokens) {
  const positions = new Uint32Array(tokens.length + 1);
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    positions[i + 1] = positions[i] + (token < 256 ? 1 : token & 511);
  }
  return positions;
}

function mergePositions(first, second, expand) {
  if (!second) return first;
  const positions = new Uint32Array(MAX_POSITIONS);
  let i = 1, j = 1, count = 1;
  while (i < first.length || j < second.length) {
    const point = Math.min(first[i] ?? Infinity, second[j] ?? Infinity);
    const previous = positions[count - 1];
    const gap = point - previous;
    const padding = expand ? Math.min(5, (gap - 1) >> 1) : gap >= 8 ? 1 : 0;
    if (count + 2 * padding >= positions.length) {
      return expand ? mergePositions(first, second, false) : first;
    }
    for (let k = 1; k <= padding; k++) positions[count++] = previous + k;
    for (let k = padding; k > 0; k--) positions[count++] = point - k;
    positions[count++] = point;
    if (first[i] === point) i++;
    if (second[j] === point) j++;
  }
  return positions.subarray(0, count);
}

function addMatchEnds(positions, longest) {
  const ends = new Uint32Array(positions.length);
  let count = 0;
  for (let i = 0; i + 1 < positions.length; i++) {
    if (longest[i] >= 8) ends[count++] = positions[i] + longest[i];
  }
  const sorted = ends.subarray(0, count).sort();
  const merged = new Uint32Array(positions.length + count);
  let a = 0, b = 0, size = 0;
  while (a < positions.length || b < count) {
    const point = Math.min(positions[a] ?? Infinity, sorted[b] ?? Infinity);
    if (point !== merged[size - 1]) merged[size++] = point;
    if (positions[a] === point) a++;
    while (sorted[b] === point) b++;
  }
  return merged.subarray(0, size);
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

function hintedDistance(tokens, positions, index, pos) {
  const token = tokens[index];
  if (token >= 256) return token >>> 9;
  if (positions[index + 1] - pos <= 2) return tokens[index + 1] >>> 9;
  return 0;
}

function shortHash(bytes, pos) {
  const key = (bytes[pos] << 16) | (bytes[pos + 1] << 8) | bytes[pos + 2];
  return Math.imul(key, 0x9e3779b1) >>> 16;
}

function longHash(words, pos) {
  return (Math.imul(words.getUint32(pos), 0x85ebca6b) ^ Math.imul(words.getUint32(pos + 4), 0xc2b2ae35)) >>> 16;
}

function runHash(bytes, pos, length) {
  const end = pos + length;
  const suffix = (bytes[end] << 16) | (bytes[end + 1] << 8) | bytes[end + 2];
  return Math.imul(suffix ^ (length << 15) ^ (bytes[pos] << 7), 0x85ebca6b) >>> 16;
}

function matchOptions(bytes, rowLength, bpp, positions, first, firstEnds, second, secondEnds) {
  const words = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
  const runs = new Uint16Array(bytes.length);
  let remaining = 0;
  for (let p = bytes.length - 1; p >= 0; p--) {
    remaining = bytes[p] === bytes[p + 1] ? Math.min(MATCH_LIMIT, remaining + 1) : 1;
    runs[p] = remaining;
  }
  const options = new Uint32Array(positions.length * MATCH_OPTIONS);
  const counts = new Uint8Array(positions.length);
  const longest = new Uint16Array(positions.length);
  const shortHeads = new Int32Array(HASH_SIZE).fill(-1);
  const longHeads = new Int32Array(HASH_SIZE).fill(-1);
  const runHeads = new Int32Array(HASH_SIZE).fill(-1);
  const shortChain = new Int32Array(WINDOW);
  const probes = new Int32Array(MATCH_OPTIONS);
  const mask = WINDOW - 1;
  let a = 0, b = 0, indexed = 0;
  for (let i = 0; i + 1 < positions.length; i++) {
    const pos = positions[i];
    const limit = Math.min(MATCH_LIMIT, bytes.length - pos);
    if (limit < 3) break;
    while (firstEnds[a + 1] <= pos) a++;
    while (second && secondEnds[b + 1] <= pos) b++;
    const from = Math.max(0, pos - WINDOW);
    // The dictionary may refer to any byte, even though the parse only visits
    // selected positions. Index every source with constant work per byte.
    while (indexed < pos) {
      const short = shortHash(bytes, indexed);
      shortChain[indexed & mask] = shortHeads[short];
      shortHeads[short] = indexed;
      const run = runs[indexed];
      if (run < 8 && indexed + 8 <= bytes.length) longHeads[longHash(words, indexed)] = indexed;
      if (run >= 3 && run < MATCH_LIMIT && indexed + run + 2 < bytes.length) runHeads[runHash(bytes, indexed, run)] = indexed;
      indexed++;
    }
    probes[0] = pos - 1;
    probes[1] = pos - bpp;
    probes[2] = pos - rowLength;
    probes[3] = pos - 2 * rowLength;
    probes[4] = pos - hintedDistance(first, firstEnds, a, pos);
    probes[5] = second ? pos - hintedDistance(second, secondEnds, b, pos) : -1;
    let prev = shortHeads[shortHash(bytes, pos)];
    for (let k = 6; k < 10; k++) {
      probes[k] = prev;
      prev = prev > from ? shortChain[prev & mask] : -1;
    }
    const run = runs[pos];
    probes[10] = limit >= 8 && run < 8 ? longHeads[longHash(words, pos)] : -1;
    probes[11] = run >= 3 && run < MATCH_LIMIT && pos + run + 2 < bytes.length ? runHeads[runHash(bytes, pos, run)] : -1;

    const base = i * MATCH_OPTIONS;
    let count = 0;
    for (let index = 0; index < probes.length; index++) {
      const earlier = probes[index];
      if (earlier < from || earlier >= pos) continue;
      const distance = pos - earlier;
      let duplicate = false;
      for (let j = 0; j < count; j++) if ((options[base + j] >>> 9) === distance) duplicate = true;
      if (duplicate) continue;
      const length = matchLength(bytes, words, runs, pos, earlier, limit);
      if (length >= 3) {
        options[base + count++] = (distance << 9) | length;
        longest[i] = Math.max(longest[i], length);
        if (distance === 1 && length === limit) break;
      }
    }
    counts[i] = count;
  }
  return { options, counts, longest };
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

function readNativeTokens(stream, bytes) {
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

  const tokens = [];
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
      if (tokens.length + length > MAX_TOKENS) return null;
      if (cursor + length > bytes.length) throw new Error('Native stored block is outside the input');
      for (let i = cursor; i < cursor + length; i++) tokens.push(bytes[i]);
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
      if (tokens.length === MAX_TOKENS) return null;
      if (literal < 256) {
        tokens.push(literal);
        cursor++;
        continue;
      }
      const lc = literal - 257;
      if (lc >= LENGTH_EXTRA.length) throw new Error('Invalid native match length');
      const length = LENGTH_BASE[lc] + read(LENGTH_EXTRA[lc]);
      const dc = nextSymbol(distances);
      if (dc >= DISTANCE_EXTRA.length) throw new Error('Invalid native match distance');
      const distance = DISTANCE_BASE[dc] + read(DISTANCE_EXTRA[dc]);
      if (distance > cursor || cursor + length > bytes.length) throw new Error('Native match is outside the input');
      tokens.push((distance << 9) | length);
      cursor += length;
    }
  } while (!last);
  if (cursor !== bytes.length) throw new Error('Native stream has the wrong input length');
  return Uint32Array.from(tokens);
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

const COST_UNIT = 8;

function estimatedCosts(counts, total, lengths, exact) {
  return Uint16Array.from(counts, (frequency, i) => {
    if (exact && frequency) return lengths[i] * COST_UNIT;
    const entropy = Math.min(16, Math.log2((total + counts.length) / (1 + frequency)));
    const estimate = frequency ? (lengths[i] + entropy) / 2 : entropy;
    return Math.max(1, Math.round(estimate * COST_UNIT));
  });
}

function optimiseTokens(bytes, positions, graph, start, stop, seed, stats, budget, exact) {
  const { options, counts, longest, costs, next, distances } = graph;
  const literal = estimatedCosts(stats.literals, seed.tokens.length + 1, seed.lit, exact);
  const distance = estimatedCosts(stats.distances, stats.matches, seed.dist, exact);
  const lengthPrice = Uint16Array.from(LENGTH_EXTRA, (extra, i) => literal[257 + i] + COST_UNIT * extra);
  const distancePrice = Uint16Array.from(DISTANCE_EXTRA, (extra, i) => distance[i] + COST_UNIT * extra);
  const prices = new Uint16Array(MATCH_OPTIONS);
  let remaining = budget.remaining;
  costs[stop] = 0;
  for (let i = stop - 1; i >= start; i--) {
    const pos = positions[i];
    const following = positions[i + 1];
    let best = costs[i + 1];
    for (let byte = pos; byte < following; byte++) best += literal[bytes[byte]];
    let bestNext = i + 1;
    let bestDistance = 0;
    const end = Math.min(pos + longest[i], positions[stop]);
    if (end >= following) {
      const base = i * MATCH_OPTIONS;
      const optionsCount = counts[i];
      for (let k = 0; k < optionsCount; k++) prices[k] = distancePrice[DISTANCE_CODE[options[base + k] >>> 9]];
      let target = i + 1;
      while (target <= stop && positions[target] - pos < 3) target++;
      while (target <= stop && positions[target] <= end) {
        const length = positions[target] - pos;
        let cheapest = Infinity;
        let chosenDistance = 0;
        let reach = 0;
        for (let k = 0; k < optionsCount; k++) {
          const match = options[base + k];
          const limit = match & 511;
          if (limit >= length && (prices[k] < cheapest || (prices[k] === cheapest && limit > reach))) {
            cheapest = prices[k];
            chosenDistance = match >>> 9;
            reach = limit;
          }
        }
        // The cheapest distance stays valid until its match length expires.
        const boundary = Math.min(end, pos + reach);
        while (target <= stop && positions[target] <= boundary) {
          if (--remaining < 0) {
            budget.remaining = 0;
            return null;
          }
          const span = positions[target] - pos;
          const cost = costs[target] + cheapest + lengthPrice[LENGTH_CODE[span]];
          if (cost < best) {
            best = cost;
            bestNext = target;
            bestDistance = chosenDistance;
          }
          target++;
        }
      }
    }
    next[i] = bestNext;
    distances[i] = bestDistance;
    costs[i] = best;
  }
  budget.remaining = remaining;

  const tokens = [];
  for (let i = start; i < stop; i = next[i]) {
    const pos = positions[i];
    const end = positions[next[i]];
    const distance = distances[i];
    if (distance) tokens.push((distance << 9) | (end - pos));
    else for (let byte = pos; byte < end; byte++) tokens.push(bytes[byte]);
  }
  return Uint32Array.from(tokens);
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

export function optimiseDeflate(bytes, rowLength, bpp, maxBytes, nativeStream, alternateStream) {
  if (bytes.length === 0 || maxBytes < 7) return null;
  const native = readNativeTokens(nativeStream, bytes);
  if (!native) return null;
  const alternate = alternateStream ? readNativeTokens(alternateStream, bytes) : null;
  // A full byte-by-byte search is expensive on long repeated spans. Native
  // token boundaries retain both parses while keeping the graph small.
  const nativePositions = tokenPositions(native);
  const alternatePositions = alternate && tokenPositions(alternate);
  let positions = mergePositions(nativePositions, alternatePositions, native.length <= 62_000);
  let paths = matchOptions(bytes, rowLength, bpp, positions, native, nativePositions, alternate, alternatePositions);
  const refine = bytes.length <= (1 << 20) && positions.length <= 110_000;
  if (refine) {
    // Match ends expose useful cut positions that neither native parse used.
    positions = addMatchEnds(positions, paths.longest);
    paths = matchOptions(bytes, rowLength, bpp, positions, native, nativePositions, alternate, alternatePositions);
  }
  const graph = {
    ...paths,
    costs: new Float64Array(positions.length),
    next: new Uint32Array(positions.length),
    distances: new Uint16Array(positions.length),
  };
  const budget = { remaining: refine ? 3_600_000 : MAX_EDGES };
  const passes = positions.length <= 140_000 ? 2 : 1;
  let end = 0;
  let blocks = splitBlocks(native, GROUP_SIZE).map(({ stop, tokens }) => {
    const start = end;
    while (positions[end] < stop) end++;
    let stats = countSymbols(tokens);
    let seed = describeBlock(tokens, stats);
    let best = seed;
    for (let iteration = 0; iteration < passes && budget.remaining > 0; iteration++) {
      const optimised = optimiseTokens(bytes, positions, graph, start, end, seed, stats, budget, iteration > 0);
      if (!optimised) break;
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
