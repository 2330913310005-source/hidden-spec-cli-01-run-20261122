// main.ts is mend, the CLI SPEC.md specifies: one command-line tool that applies the edits AI models write,
// in four formats (unified diff, SEARCH/REPLACE blocks, V4A patches, JSON tool calls), to a tree of text
// files, to one written contract. Node 22, the standard library alone; run it as `node main.ts [options] [PATCH]`.
//
// How it works, per SPEC.md:
//  - Sections 1-4, 9, 10 (command line, output, safe paths, format detection, JSON edits, all-or-nothing
//    write) are the kit starter's code, kept as handed out.
//  - Section 5: line matching at three tiers (exact; trailing whitespace ignored; leading and trailing
//    whitespace ignored), with tier-2 re-indentation of new lines and context lines written as the file's own.
//  - Section 6: unified diffs — sections, hunks, nearest-occurrence choice with the A-1+d target, empty-O
//    insertion, `\ No newline at end of file`, creations, deletions, moves (reported as `R old -> new`).
//  - Section 7: SEARCH/REPLACE blocks — path from the nearest non-blank, non-fence line above, exactly-once
//    matching (else `ambiguous`), empty old lines create.
//  - Section 8: V4A patches — Add/Delete/Update sections, `*** Move to:`, `@@` anchors with a cursor,
//    `*** End of File` chunks, blank lines at the end of a section dropped.
//  - Every edit is checked for path safety before anything else; a failed edit changes nothing and the
//    edits after it still run; when any edit failed no file is written at all.
//
// In a trial the grader runs `node /bot/main.ts` with MAKEFUN_ENV_URL set, and then this file hands over to
// runner.ts (the last lines below): keep that hand-over and runner.ts as they are.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMATS = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

class Exit2 {
  readonly line: string;
  constructor(line: string) {
    this.line = line;
  }
}
class Fail {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const blank = (s: string): boolean => s.trim() === '';

function patchLines(text: string): string[] {
  if (text === '') return [];
  const lines = text.split('\n');
  if (text.endsWith('\n')) lines.pop();
  return lines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

function safe(p: string): boolean {
  return p !== '' && !p.startsWith('/') && !p.includes('\\') && p.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
}

/** The tree as the edits leave it: changed paths in `now` (null: deleted), the rest read from disk. */
class Tree {
  private readonly orig = new Map<string, string | null>();
  readonly now = new Map<string, string | null>();
  /** Moves that succeeded, in patch order: used to print `R old -> new` instead of a `D`/`A` pair. */
  readonly moves: Array<{ old: string; new: string }> = [];
  original(p: string): string | null {
    if (!this.orig.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.orig.set(p, v);
    }
    return this.orig.get(p)!;
  }
  get(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.original(p);
  }
}

type Edit = (tree: Tree) => void;

/** A file's lines under its own line break, and whether it ends with one (SPEC.md section 3). */
interface FileState {
  br: string;
  lines: string[];
  final: boolean;
}

function splitFile(text: string): FileState {
  if (text === '') return { br: '\n', lines: [], final: false };
  const nl = text.indexOf('\n');
  const br = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const lines = text.split(br);
  let final = false;
  if (text.endsWith(br)) {
    final = true;
    lines.pop();
  }
  return { br, lines, final };
}

function joinFile(f: FileState): string {
  if (f.lines.length === 0) return '';
  return f.lines.join(f.br) + (f.final ? f.br : '');
}

// --- Section 5: matching lines at three tiers -------------------------------------------

const trimEndWS = (s: string): string => s.replace(/[ \t]+$/, '');
const trimWS = (s: string): string => s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');
const leadWS = (s: string): string => {
  const m = /^[ \t]*/.exec(s);
  return m === null ? '' : m[0];
};

function keyAt(line: string, tier: number): string {
  if (tier === 0) return line;
  if (tier === 1) return trimEndWS(line);
  return trimWS(line);
}

/** Every 0-based position where O occurs in `lines` at this tier (occurrences may overlap). */
function occurrences(lines: string[], O: string[], tier: number): number[] {
  const out: number[] = [];
  if (O.length === 0 || O.length > lines.length) return out;
  const keys = O.map((l) => keyAt(l, tier));
  for (let i = 0; i + O.length <= lines.length; i++) {
    let good = true;
    for (let j = 0; j < O.length; j++) {
      if (keyAt(lines[i + j]!, tier) !== keys[j]) {
        good = false;
        break;
      }
    }
    if (good) out.push(i);
  }
  return out;
}

interface TierHit {
  tier: number;
  pos: number[];
}

/**
 * The first tier (0, then 1, then 2; only 0 with --strict) at which O occurs at least once at or
 * after `from`, restricted to positions passing `pred`. Null when O occurs at no tier.
 */
function findTier(
  lines: string[],
  O: string[],
  strict: boolean,
  from: number,
  pred: (p: number) => boolean = () => true,
): TierHit | null {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    const all = occurrences(lines, O, t).filter((p) => p >= from && pred(p));
    if (all.length > 0) return { tier: t, pos: all };
  }
  return null;
}

/** One line of a hunk/chunk body: context (in both O and N), old (only O) or new (only N). */
interface BodyEntry {
  kind: 'ctx' | 'old' | 'new';
  text: string;
}

/** Split body entries into O (context + old) and N (context + new) with context positions. */
function splitEntries(entries: BodyEntry[]): {
  O: string[];
  N: Array<{ text: string; ctx: boolean; oIdx: number }>;
} {
  const O: string[] = [];
  const N: Array<{ text: string; ctx: boolean; oIdx: number }> = [];
  for (const e of entries) {
    if (e.kind === 'ctx') {
      O.push(e.text);
      N.push({ text: e.text, ctx: true, oIdx: O.length - 1 });
    } else if (e.kind === 'old') {
      O.push(e.text);
    } else {
      N.push({ text: e.text, ctx: false, oIdx: -1 });
    }
  }
  return { O, N };
}

/**
 * The lines N writes for a match of O at `pos`: context lines are the file's own matched lines;
 * at tier 2 every other line is re-indented (O's first line's leading whitespace P replaced by the
 * file's first matched line's F); at tiers 0 and 1 new lines are written as they are.
 */
function buildN(
  fileLines: string[],
  pos: number,
  O: string[],
  N: Array<{ text: string; ctx: boolean; oIdx: number }>,
  tier: number,
): string[] {
  if (tier !== 2) {
    return N.map((e) => (e.ctx ? fileLines[pos + e.oIdx]! : e.text));
  }
  const F = leadWS(fileLines[pos]!);
  const P = leadWS(O[0]!);
  return N.map((e) => {
    if (e.ctx) return fileLines[pos + e.oIdx]!;
    const t = e.text;
    if (t === '' || /^[ \t]+$/.test(t)) return t;
    if (t.startsWith(P)) return F + t.slice(P.length);
    return t;
  });
}

function detect(lines: string[]): string | null {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

// --- Section 6: unified diffs ------------------------------------------------------------

/** One hunk: its old start line A, its body, and whether a `\` line follows its last context/`+` line. */
interface Hunk {
  A: number;
  entries: BodyEntry[];
  noEol: boolean;
  plusNoEol: boolean;
}

/**
 * Parse a hunk body's raw lines (trailing empty lines already stripped) into entries. A `\` line
 * attaches to the body line before it; noEol/plusNoEol say whether one follows the last context-or-`+`
 * line / the last `+` line.
 */
function parseHunkBody(raw: string[]): { entries: BodyEntry[]; noEol: boolean; plusNoEol: boolean } {
  const entries: BodyEntry[] = [];
  const nextIsBackslash: boolean[] = [];
  for (let k = 0; k < raw.length; k++) {
    const r = raw[k]!;
    if (r.startsWith('\\')) continue;
    let e: BodyEntry;
    if (r === '') e = { kind: 'ctx', text: '' };
    else if (r.startsWith(' ')) e = { kind: 'ctx', text: r.slice(1) };
    else if (r.startsWith('-')) e = { kind: 'old', text: r.slice(1) };
    else e = { kind: 'new', text: r.slice(1) };
    entries.push(e);
    nextIsBackslash.push(k + 1 < raw.length && raw[k + 1]!.startsWith('\\'));
  }
  let noEol = false;
  let plusNoEol = false;
  for (let k = entries.length - 1; k >= 0; k--) {
    if (entries[k]!.kind === 'ctx' || entries[k]!.kind === 'new') {
      noEol = nextIsBackslash[k]!;
      break;
    }
  }
  for (let k = entries.length - 1; k >= 0; k--) {
    if (entries[k]!.kind === 'new') {
      plusNoEol = nextIsBackslash[k]!;
      break;
    }
  }
  return { entries, noEol, plusNoEol };
}

interface USection {
  oldPath: string | null;
  newPath: string | null;
  create: boolean;
  del: boolean;
  hunks: Hunk[];
}

/** The path after `--- ` or `+++ `: text after the four characters, up to the first tab, trimmed. */
function dashPath(line: string): string | null {
  let p = line.slice(4);
  const tab = p.indexOf('\t');
  if (tab >= 0) p = p.slice(0, tab);
  p = p.trim();
  if (p === '/dev/null') return null;
  if (p.startsWith('a/')) p = p.slice(2);
  else if (p.startsWith('b/')) p = p.slice(2);
  return p;
}

const HUNK_RE = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/;
const DIFFGIT_RE = /^diff --git a\/(\S+) b\/(\S+)$/;

function isDiffGit(line: string): boolean {
  return line === 'diff --git' || line.startsWith('diff --git ');
}

/**
 * Apply a section's hunks in order to a file's text. Throws Fail('no-match', path) when a hunk's
 * old lines are nowhere. The file's final line break is changed only by a hunk whose matched lines
 * run to the end of the file (or, for an empty O, whose insertion is at the end): then it ends with
 * a break unless a `\` line follows the hunk's last context or `+` line.
 */
function applyHunks(text: string, hunks: Hunk[], strict: boolean, failPath: string): string {
  const f = splitFile(text);
  let d = 0;
  for (const h of hunks) {
    const { O, N } = splitEntries(h.entries);
    if (O.length === 0) {
      const oldLen = f.lines.length;
      const at = Math.max(0, Math.min(h.A + d, oldLen));
      const ins = N.map((e) => e.text);
      f.lines.splice(at, 0, ...ins);
      if (at === oldLen) f.final = !h.noEol;
      d += N.length;
      continue;
    }
    const found = findTier(f.lines, O, strict, 0);
    if (found === null) throw new Fail('no-match', failPath);
    const target = h.A - 1 + d;
    let best = found.pos[0]!;
    let bestDist = Math.abs(best - target);
    for (const p of found.pos) {
      const dd = Math.abs(p - target);
      if (dd < bestDist) {
        best = p;
        bestDist = dd;
      }
    }
    const atEnd = best + O.length === f.lines.length;
    const rep = buildN(f.lines, best, O, N, found.tier);
    f.lines.splice(best, O.length, ...rep);
    if (atEnd) f.final = !h.noEol;
    d += N.length - O.length;
  }
  return joinFile(f);
}

function sectionEdit(sec: USection, strict: boolean): Edit {
  return (tree) => {
    if (sec.oldPath === null && sec.newPath === null) return; // both /dev/null: changes nothing
    if (sec.oldPath !== null && !safe(sec.oldPath)) throw new Fail('unsafe', sec.oldPath);
    if (sec.newPath !== null && !safe(sec.newPath)) throw new Fail('unsafe', sec.newPath);
    if (sec.create) {
      const p = sec.newPath!;
      if (tree.get(p) !== null) throw new Fail('exists', p);
      const plus: string[] = [];
      for (const h of sec.hunks) for (const e of h.entries) if (e.kind === 'new') plus.push(e.text);
      let noEol = false;
      for (let hi = sec.hunks.length - 1; hi >= 0; hi--) {
        if (sec.hunks[hi]!.entries.some((e) => e.kind === 'new')) {
          noEol = sec.hunks[hi]!.plusNoEol;
          break;
        }
      }
      tree.now.set(p, plus.length === 0 ? '' : plus.join('\n') + (noEol ? '' : '\n'));
      return;
    }
    if (sec.del) {
      const p = sec.oldPath!;
      if (tree.get(p) === null) throw new Fail('missing', p);
      tree.now.set(p, null);
      return;
    }
    if (sec.oldPath !== sec.newPath) {
      const oldP = sec.oldPath!;
      const newP = sec.newPath!;
      const text = tree.get(oldP);
      if (text === null) throw new Fail('missing', oldP);
      if (tree.get(newP) !== null) throw new Fail('exists', newP);
      const out = applyHunks(text, sec.hunks, strict, oldP);
      tree.now.set(oldP, null);
      tree.now.set(newP, out);
      tree.moves.push({ old: oldP, new: newP });
      return;
    }
    const p = sec.oldPath!;
    const text = tree.get(p);
    if (text === null) throw new Fail('missing', p);
    tree.now.set(p, applyHunks(text, sec.hunks, strict, p));
  };
}

function unifiedEdits(lines: string[], strict: boolean): Edit[] {
  const sections: USection[] = [];
  let cur: USection | null = null;
  // Header state kept while a section is open.
  let pairSeen = false;
  let renameOld: string | null = null;
  let renameNew: string | null = null;
  let diffOld: string | null = null;
  let diffNew: string | null = null;
  let newFileMode = false;
  let deletedFileMode = false;
  // 'header': before the section's first hunk; 'hunks': after it (between hunks, or after a body).
  let phase: 'header' | 'hunks' = 'header';

  const finishSection = (): void => {
    if (cur === null) return;
    let oldPath: string | null;
    let newPath: string | null;
    if (pairSeen) {
      oldPath = cur.oldPath;
      newPath = cur.newPath;
    } else {
      oldPath = renameOld ?? diffOld;
      newPath = renameNew ?? diffNew;
    }
    const create = newFileMode || oldPath === null;
    const del = !create && (deletedFileMode || newPath === null);
    sections.push({ oldPath, newPath, create, del, hunks: cur.hunks });
    cur = null;
  };
  const startSection = (oldP: string | null, newP: string | null, withPair: boolean): void => {
    finishSection();
    cur = { oldPath: oldP, newPath: newP, create: false, del: false, hunks: [] };
    pairSeen = withPair;
    renameOld = null;
    renameNew = null;
    diffOld = null;
    diffNew = null;
    newFileMode = false;
    deletedFileMode = false;
    phase = 'header';
  };

  let i = 0;
  while (i < lines.length) {
    const line = lines[i]!;
    if (isDiffGit(line)) {
      const m = DIFFGIT_RE.exec(line);
      if (m === null) throw new Exit2(`error: parse ${i + 1}`);
      startSection(null, null, false);
      diffOld = m[1]!;
      diffNew = m[2]!;
      i++;
      continue;
    }
    if (line.startsWith('--- ')) {
      const nextIsPlus = (lines[i + 1] ?? '').startsWith('+++ ');
      if (phase === 'header' && cur !== null && !pairSeen) {
        if (!nextIsPlus) throw new Exit2(`error: parse ${i + 2}`);
        cur.oldPath = dashPath(line);
        cur.newPath = dashPath(lines[i + 1]!);
        pairSeen = true;
        i += 2;
        continue;
      }
      if (nextIsPlus) {
        startSection(dashPath(line), dashPath(lines[i + 1]!), true);
        i += 2;
        continue;
      }
      // A lone `--- ` line outside a section header is prose: ignored.
      i++;
      continue;
    }
    if (line.startsWith('@@')) {
      if (cur === null) throw new Exit2(`error: parse ${i + 1}`);
      const m = HUNK_RE.exec(line);
      if (m === null) throw new Exit2(`error: parse ${i + 1}`);
      const hunk: Hunk = { A: Number(m[1]), entries: [], noEol: false, plusNoEol: false };
      // Collect the body: lines starting with ` `, `-`, `+`, `\`, or empty. It ends at the first
      // other line, at a `diff --git ` line, at a `--- ` line followed by a `+++ ` line, or at the
      // end of the patch; empty lines at the end of a body are not part of it.
      let j = i + 1;
      const raw: string[] = [];
      while (j < lines.length) {
        const L = lines[j]!;
        if (L.startsWith('diff --git ')) break;
        if (L.startsWith('--- ') && (lines[j + 1] ?? '').startsWith('+++ ')) break;
        if (L.startsWith('@@')) break;
        if (L === '' || L.startsWith(' ') || L.startsWith('-') || L.startsWith('+') || L.startsWith('\\')) {
          raw.push(L);
          j++;
          continue;
        }
        break;
      }
      while (raw.length > 0 && raw[raw.length - 1] === '') raw.pop();
      const parsed = parseHunkBody(raw);
      hunk.entries = parsed.entries;
      hunk.noEol = parsed.noEol;
      hunk.plusNoEol = parsed.plusNoEol;
      cur.hunks.push(hunk);
      phase = 'hunks';
      i = j;
      continue;
    }
    if (cur !== null && phase === 'header') {
      if (line.startsWith('rename from ')) renameOld = line.slice('rename from '.length).trim();
      else if (line.startsWith('rename to ')) renameNew = line.slice('rename to '.length).trim();
      else if (line.startsWith('new file mode')) newFileMode = true;
      else if (line.startsWith('deleted file mode')) deletedFileMode = true;
      // `index`, `old mode`, `new mode`, `similarity index`, `dissimilarity index`, `Index:` and
      // any other line in the header are ignored.
    }
    // Outside bodies, lines that are not part of a section header are ignored (prose, code fences).
    i++;
  }
  finishSection();
  return sections.map((s) => sectionEdit(s, strict));
}

// --- Section 7: SEARCH/REPLACE blocks ------------------------------------------------------

function srEdit(path: string, oldLines: string[], newLines: string[], strict: boolean): Edit {
  return (tree) => {
    if (!safe(path)) throw new Fail('unsafe', path);
    if (oldLines.length === 0) {
      if (tree.get(path) !== null) throw new Fail('exists', path);
      tree.now.set(path, newLines.length === 0 ? '' : newLines.join('\n') + '\n');
      return;
    }
    const text = tree.get(path);
    if (text === null) throw new Fail('missing', path);
    const f = splitFile(text);
    const found = findTier(f.lines, oldLines, strict, 0);
    if (found === null) throw new Fail('no-match', path);
    if (found.pos.length > 1) throw new Fail('ambiguous', path);
    const pos = found.pos[0]!;
    const N = newLines.map((t) => ({ text: t, ctx: false, oIdx: -1 }));
    const rep = buildN(f.lines, pos, oldLines, N, found.tier);
    f.lines.splice(pos, oldLines.length, ...rep);
    tree.now.set(path, joinFile(f));
  };
}

function searchReplaceEdits(lines: string[], strict: boolean): Edit[] {
  const edits: Edit[] = [];
  let prevEnd = -1;
  let prevPath: string | null = null;
  let i = 0;
  while (i < lines.length) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i++;
      continue;
    }
    const searchIdx = i;
    let path: string | null = null;
    for (let j = searchIdx - 1; j > prevEnd; j--) {
      const t = lines[j]!.trim();
      if (t === '' || t.startsWith('```')) continue;
      path = t;
      break;
    }
    if (path === null) path = prevPath;
    if (path === null) throw new Exit2(`error: parse ${searchIdx + 1}`);
    let k = searchIdx + 1;
    let sep = -1;
    while (k < lines.length) {
      const t = lines[k]!.trimEnd();
      if (t === '=======') {
        sep = k;
        break;
      }
      if (t === '>>>>>>> REPLACE' || t === '<<<<<<< SEARCH') break;
      k++;
    }
    if (sep < 0) throw new Exit2(`error: parse ${searchIdx + 1}`);
    let m = sep + 1;
    let rep = -1;
    while (m < lines.length) {
      const t = lines[m]!.trimEnd();
      if (t === '>>>>>>> REPLACE') {
        rep = m;
        break;
      }
      if (t === '<<<<<<< SEARCH') break;
      m++;
    }
    if (rep < 0) throw new Exit2(`error: parse ${searchIdx + 1}`);
    edits.push(srEdit(path, lines.slice(searchIdx + 1, sep), lines.slice(sep + 1, rep), strict));
    prevPath = path;
    prevEnd = rep;
    i = rep + 1;
  }
  return edits;
}

// --- Section 8: V4A patches ---------------------------------------------------------------

interface VChunk {
  anchor: string;
  eof: boolean;
  entries: BodyEntry[];
}

function isVBody(b: string): boolean {
  return b === '' || b.startsWith(' ') || b.startsWith('-') || b.startsWith('+');
}

function vBodyEntry(b: string): BodyEntry {
  if (b === '') return { kind: 'ctx', text: '' };
  if (b.startsWith(' ')) return { kind: 'ctx', text: b.slice(1) };
  if (b.startsWith('-')) return { kind: 'old', text: b.slice(1) };
  return { kind: 'new', text: b.slice(1) };
}

function vAddEdit(path: string, fileLines: string[]): Edit {
  return (tree) => {
    if (!safe(path)) throw new Fail('unsafe', path);
    if (tree.get(path) !== null) throw new Fail('exists', path);
    tree.now.set(path, fileLines.length === 0 ? '' : fileLines.join('\n') + '\n');
  };
}

function vDeleteEdit(path: string): Edit {
  return (tree) => {
    if (!safe(path)) throw new Fail('unsafe', path);
    if (tree.get(path) === null) throw new Fail('missing', path);
    tree.now.set(path, null);
  };
}

function vUpdateEdit(path: string, moveTo: string | null, chunks: VChunk[], strict: boolean): Edit {
  return (tree) => {
    if (!safe(path)) throw new Fail('unsafe', path);
    if (moveTo !== null && !safe(moveTo)) throw new Fail('unsafe', moveTo);
    const text = tree.get(path);
    if (text === null) throw new Fail('missing', path);
    if (moveTo !== null && tree.get(moveTo) !== null) throw new Fail('exists', moveTo);
    const f = splitFile(text);
    let cursor = 0;
    for (const ch of chunks) {
      if (ch.anchor !== '') {
        const found = findTier(f.lines, [ch.anchor], strict, cursor);
        if (found === null) throw new Fail('no-match', path);
        cursor = found.pos[0]! + 1;
      }
      const { O, N } = splitEntries(ch.entries);
      if (O.length === 0) {
        const at = ch.eof ? f.lines.length : cursor;
        f.lines.splice(at, 0, ...N.map((e) => e.text));
        cursor = at + N.length;
        continue;
      }
      const found = ch.eof
        ? findTier(f.lines, O, strict, cursor, (p) => p + O.length === f.lines.length)
        : findTier(f.lines, O, strict, cursor);
      if (found === null) throw new Fail('no-match', path);
      const pos = found.pos[0]!;
      const rep = buildN(f.lines, pos, O, N, found.tier);
      f.lines.splice(pos, O.length, ...rep);
      cursor = pos + rep.length;
    }
    const out = joinFile(f);
    if (moveTo !== null) {
      tree.now.set(path, null);
      tree.now.set(moveTo, out);
      tree.moves.push({ old: path, new: moveTo });
    } else {
      tree.now.set(path, out);
    }
  };
}

function v4aEdits(lines: string[], strict: boolean): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  if (lines[first]!.trim() !== '*** Begin Patch') throw new Exit2(`error: parse ${first + 1}`);
  let last = -1;
  for (let k = lines.length - 1; k >= 0; k--) {
    if (!blank(lines[k]!)) {
      last = k;
      break;
    }
  }
  if (lines[last]!.trim() !== '*** End Patch') throw new Exit2(`error: parse ${lines.length + 1}`);
  const edits: Edit[] = [];
  let i = first + 1;
  while (i < last) {
    const line = lines[i]!;
    if (blank(line)) {
      i++;
      continue;
    }
    if (line.trimEnd() === '*** End Patch') {
      i++;
      continue;
    }
    let kind: 'add' | 'delete' | 'update' | null = null;
    let path = '';
    if (line.startsWith('*** Add File: ')) {
      kind = 'add';
      path = line.slice('*** Add File: '.length).trim();
    } else if (line.startsWith('*** Delete File: ')) {
      kind = 'delete';
      path = line.slice('*** Delete File: '.length).trim();
    } else if (line.startsWith('*** Update File: ')) {
      kind = 'update';
      path = line.slice('*** Update File: '.length).trim();
    } else {
      throw new Exit2(`error: parse ${i + 1}`);
    }
    const headerIdx = i;
    // The section's lines run to the next section header, an `*** End Patch` line, or the end.
    let j = i + 1;
    while (j < last) {
      const l = lines[j]!;
      if (
        l.startsWith('*** Add File: ') ||
        l.startsWith('*** Delete File: ') ||
        l.startsWith('*** Update File: ') ||
        l.trimEnd() === '*** End Patch'
      ) {
        break;
      }
      j++;
    }
    const raw: Array<{ text: string; idx: number }> = [];
    for (let k = i + 1; k < j; k++) raw.push({ text: lines[k]!, idx: k });
    while (raw.length > 0 && blank(raw[raw.length - 1]!.text)) raw.pop();
    if (kind === 'add') {
      const fileLines: string[] = [];
      for (const r of raw) {
        if (r.text === '') fileLines.push('');
        else if (r.text.startsWith('+')) fileLines.push(r.text.slice(1));
        else throw new Exit2(`error: parse ${r.idx + 1}`);
      }
      edits.push(vAddEdit(path, fileLines));
    } else if (kind === 'delete') {
      if (raw.length > 0) throw new Exit2(`error: parse ${raw[0]!.idx + 1}`);
      edits.push(vDeleteEdit(path));
    } else {
      let k = 0;
      let moveTo: string | null = null;
      if (raw.length > 0 && raw[0]!.text.startsWith('*** Move to: ')) {
        moveTo = raw[0]!.text.slice('*** Move to: '.length).trim();
        k = 1;
      }
      const chunks: VChunk[] = [];
      while (k < raw.length) {
        const l = raw[k]!;
        if (l.text.startsWith('@@')) {
          const anchor = l.text.slice(2).trim();
          k++;
          const chunk: VChunk = { anchor, eof: false, entries: [] };
          while (k < raw.length) {
            const b = raw[k]!;
            if (b.text.startsWith('@@')) break;
            if (b.text.trimEnd() === '*** End of File') break;
            if (isVBody(b.text)) {
              chunk.entries.push(vBodyEntry(b.text));
              k++;
              continue;
            }
            throw new Exit2(`error: parse ${b.idx + 1}`);
          }
          if (k < raw.length && raw[k]!.text.trimEnd() === '*** End of File') {
            chunk.eof = true;
            k++;
          }
          chunks.push(chunk);
          continue;
        }
        if (chunks.length === 0 && isVBody(l.text)) {
          const chunk: VChunk = { anchor: '', eof: false, entries: [] };
          while (k < raw.length) {
            const b = raw[k]!;
            if (b.text.startsWith('@@')) break;
            if (b.text.trimEnd() === '*** End of File') break;
            if (isVBody(b.text)) {
              chunk.entries.push(vBodyEntry(b.text));
              k++;
              continue;
            }
            throw new Exit2(`error: parse ${b.idx + 1}`);
          }
          if (k < raw.length && raw[k]!.text.trimEnd() === '*** End of File') {
            chunk.eof = true;
            k++;
          }
          chunks.push(chunk);
          continue;
        }
        throw new Exit2(`error: parse ${l.idx + 1}`);
      }
      if (chunks.length === 0 && moveTo === null) throw new Exit2(`error: parse ${headerIdx + 1}`);
      edits.push(vUpdateEdit(path, moveTo, chunks, strict));
    }
    i = j;
  }
  return edits;
}

// --- Section 9: JSON edits (the kit starter's code, kept as handed out) -------------------------

/** One JSON edit object (SPEC.md section 9), or null when it is not one. */
function jsonEdit(v: unknown): Edit | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;
  const exists = (tree: Tree): string => {
    if (!safe(p)) throw new Fail('unsafe', p);
    const t = tree.get(p);
    if (t === null) throw new Fail('missing', p);
    return t;
  };
  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      tree.now.set(p, text);
    };
  }
  if (o.command === 'str_replace' && typeof o.old_str === 'string' && o.old_str !== '' && (o.new_str === undefined || typeof o.new_str === 'string')) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const lf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const t = lf(before);
      const old = lf(oldStr);
      let count = 0;
      for (let k = t.indexOf(old); k >= 0; k = t.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new Fail('no-match', p);
      if (count > 1) throw new Fail('ambiguous', p);
      const at = t.indexOf(old);
      const out = t.slice(0, at) + lf(newStr) + t.slice(at + old.length);
      tree.now.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }
  if (o.command === 'insert' && typeof o.insert_line === 'number' && Number.isInteger(o.insert_line) && typeof o.new_str === 'string') {
    const line = o.insert_line;
    const add = patchLines(o.new_str);
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const br = nl > 0 && before[nl - 1] === '\r' ? '\r\n' : '\n';
      const lines = before === '' ? [] : before.split(br);
      const final = before !== '' && before.endsWith(br);
      if (final) lines.pop();
      if (line < 0 || line > lines.length) throw new Fail('no-match', p);
      lines.splice(line, 0, ...add);
      tree.now.set(p, lines.length === 0 ? '' : lines.join(br) + (final ? br : ''));
    };
  }
  return null;
}

function jsonEdits(text: string, lines: string[]): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  const edits: Edit[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new Exit2(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new Exit2(`error: parse ${first + 1}`);
    for (const x of v) {
      const e = jsonEdit(x);
      if (e === null) throw new Exit2(`error: parse ${first + 1}`);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, i) => {
    if (blank(l)) return;
    let e: Edit | null = null;
    try {
      e = jsonEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new Exit2(`error: parse ${i + 1}`);
    edits.push(e);
  });
  return edits;
}

// --- Sections 1, 2, 10: invocation, output, order and atomicity ---------------------------------

function run(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let check = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') ended = true;
    else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMATS.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !check) check = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };
  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = patchLines(text);
  const kind = format === null || format === 'auto' ? detect(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  let edits: Edit[];
  try {
    if (kind === 'json') edits = jsonEdits(text, lines);
    else if (kind === 'unified') edits = unifiedEdits(lines, strict);
    else if (kind === 'search-replace') edits = searchReplaceEdits(lines, strict);
    else edits = v4aEdits(lines, strict);
    if (edits.length === 0) throw new Exit2('error: parse 1');
  } catch (e) {
    if (e instanceof Exit2) return { code: 2, out: [e.line] };
    throw e;
  }
  const tree = new Tree();
  const fails: Fail[] = [];
  for (const edit of edits) {
    const before = new Map(tree.now);
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof Fail)) throw e;
      tree.now.clear();
      for (const [k, v] of before) tree.now.set(k, v);
      fails.push(e);
    }
  }
  if (fails.length > 0) return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  // One line for each path whose state changed, in byte order of the paths (UTF-8). A move that
  // succeeded prints `R <old> -> <new>` (sorting by <new>) instead of its `D` and `A` lines.
  const rows: Array<{ key: string; line: string }> = [];
  const moved = new Set<string>();
  for (const m of tree.moves) {
    if (tree.original(m.old) !== null && tree.original(m.new) === null && tree.get(m.old) === null && tree.get(m.new) !== null) {
      rows.push({ key: m.new, line: `R ${m.old} -> ${m.new}` });
      moved.add(m.old);
      moved.add(m.new);
    }
  }
  for (const p of [...tree.now.keys()].sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)))) {
    if (moved.has(p)) continue;
    const was = tree.original(p);
    const now = tree.now.get(p)!;
    if (was === now) continue;
    rows.push({ key: p, line: `${was === null ? 'A' : now === null ? 'D' : 'M'} ${p}` });
  }
  rows.sort((a, b) => Buffer.compare(Buffer.from(a.key), Buffer.from(b.key)));
  if (!check) {
    for (const p of [...tree.now.keys()].sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)))) {
      const was = tree.original(p);
      const now = tree.now.get(p)!;
      if (was === now) continue;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  return { code: 0, out: [...rows.map((r) => r.line), `ok: ${rows.length} ${rows.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}
