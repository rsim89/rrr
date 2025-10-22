import 'dotenv/config';
import fs from 'fs';
import path from 'path';
import express from 'express';
import OpenAI from 'openai';

const app = express();
const port = process.env.PORT || 7676;

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// === Reason length config ===
const REASON_WORD_LIMIT = Number(process.env.REASON_WORD_LIMIT || 100);   // soft prompt hint
const REASON_CHAR_LIMIT = Number(process.env.REASON_CHAR_LIMIT || 1200);  // hard cap in schema / trimming
const OUTPUT_TOKENS     = Number(process.env.OUTPUT_TOKENS || 384);       // reply budget

// GPT-5 controls (env-driven)
const GPT5_EFFORT    = (process.env.GPT5_EFFORT || 'medium').toLowerCase();     // minimal|low|medium|high
const GPT5_VERBOSITY = (process.env.GPT5_VERBOSITY || 'low').toLowerCase();     // low|medium|high

// ---------- Utility ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Parse any JSON embedded anywhere in a string. Handles balanced braces and quoted strings. */
function tryParseJSON(maybeJSON) {
  if (!maybeJSON) return null;
  let s = String(maybeJSON).trim();
  s = s.replace(/^```(?:json)?\s*/i, '').replace(/```$/i, '').trim();

  // Fast path
  try { return JSON.parse(s); } catch {}

  // Robust path: scan for the first balanced {...} anywhere
  let depth = 0, start = -1, inStr = false, esc = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (inStr) {
      if (!esc && ch === '"') inStr = false;
      esc = !esc && ch === '\\';
      continue;
    }
    if (ch === '"') { inStr = true; continue; }
    if (ch === '{') { if (depth === 0) start = i; depth++; continue; }
    if (ch === '}') {
      depth--;
      if (depth === 0 && start >= 0) {
        const candidate = s.slice(start, i + 1).trim();
        try { return JSON.parse(candidate); } catch {}
        start = -1; // keep scanning
      }
    }
  }
  return null;
}

// ---------- Salvage helpers ----------
const RATING_KEYS = [
  'rating','score','rate','points','point','cuteness','aegyo','aegyo_score',
  '점수','평점','귀여움','애교점수'
];
const REASON_KEYS = [
  'reason','explanation','rationale','justification','why','comment','notes','because','analysis',
  '설명','이유','근거','사유','코멘트'
];
const NEST_KEYS = ['data','result','results','output','outputs','value','values','response','AegyoRating'];

function clampReason(s) {
  return String(s).replace(/\s+/g, ' ').trim().slice(0, REASON_CHAR_LIMIT);
}

function parseKoreanDigit(txt) {
  const t = String(txt);

  // 5점, 5 /7, 5 of 7, 5 out of 7
  let m = t.match(/\b([1-7])\s*(?:점|\s*(?:\/|\bof\b|\bout of\b)\s*7)?\b/i);
  if (m) return Number(m[1]);

  // Sino-Korean: 일 이 삼 사 오 육 칠
  const sino = { 일:1, 이:2, 삼:3, 사:4, 오:5, 육:6, 칠:7 };
  m = t.match(/(일|이|삼|사|오|육|칠)\s*(?:점|\s*(?:\/|\bof\b|\bout of\b)\s*칠)?/);
  if (m) return sino[m[1]];

  // Native: 하나/한, 둘/두, 셋/세, 넷/네, 다섯, 여섯, 일곱
  const native = { 하나:1, 한:1, 둘:2, 두:2, 셋:3, 세:3, 넷:4, 네:4, 다섯:5, 여섯:6, 일곱:7 };
  m = t.match(/(하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱)\s*(?:점)?/);
  if (m) return native[m[1]];

  // Lone digit
  m = t.match(/["']?\b([1-7])\b["']?/);
  if (m) return Number(m[1]);

  return NaN;
}

function coerceRating(raw) {
  if (raw == null) return null;
  if (typeof raw === 'number' && Number.isFinite(raw)) return Math.min(7, Math.max(1, Math.round(raw)));
  if (typeof raw === 'string') {
    const n = parseKoreanDigit(raw);
    if (Number.isInteger(n)) return Math.min(7, Math.max(1, n));
    const m = raw.match(/\b([0-9]+)\b/);
    if (m) return Math.min(7, Math.max(1, Math.round(Number(m[1]))));
  }
  return null;
}

function findPairDeep(value) {
  // Returns {rating, reason} or null; scans nested structures
  if (!value) return null;

  if (Array.isArray(value)) {
    for (const el of value) {
      const got = findPairDeep(el);
      if (got) return got;
    }
    return null;
  }

  if (typeof value === 'object') {
    // direct keys
    let ratingCandidate = null, reasonCandidate = null;

    for (const k of Object.keys(value)) {
      const low = k.toLowerCase();
      if (ratingCandidate == null && RATING_KEYS.some(rk => low.includes(rk))) {
        ratingCandidate = value[k];
      }
      if (reasonCandidate == null && REASON_KEYS.some(rk => low.includes(rk))) {
        reasonCandidate = value[k];
      }
    }
    // If pair found, coerce and return
    const rating = coerceRating(ratingCandidate);
    const reason = reasonCandidate != null ? clampReason(reasonCandidate) : '';
    if (Number.isInteger(rating) && reason.length > 0) {
      return { rating, reason };
    }

    // Common wrappers: data/result/output/...
    for (const nk of NEST_KEYS) {
      if (value && Object.prototype.hasOwnProperty.call(value, nk)) {
        const got = findPairDeep(value[nk]);
        if (got) return got;
      }
    }

    // Otherwise, scan all child values
    for (const v of Object.values(value)) {
      const got = findPairDeep(v);
      if (got) return got;
    }
    return null;
  }

  if (typeof value === 'string') {
    const rating = coerceRating(value);
    if (Number.isInteger(rating)) {
      const reason = clampReason(value);
      if (reason.length > 0) return { rating, reason };
    }
    return null;
  }

  return null;
}

/** Try to coerce any blob into {rating:number 1..7, reason:string} */
function coerceToSchema(objOrText) {
  if (!objOrText) return null;

  // If it's a string, try JSON first; else salvage from text
  if (typeof objOrText === 'string') {
    const j = tryParseJSON(objOrText);
    if (j) return coerceToSchema(j);

    const fallback = findPairDeep(objOrText);
    return fallback;
  }

  // If it's an object/array, search deeply
  if (typeof objOrText === 'object') {
    return findPairDeep(objOrText);
  }

  return null;
}

// ---------- Prompt helpers (no alternation hints) ----------
const jsonHint = `{"rating": <integer 1-7>, "reason": "Concise justification (<= ${REASON_WORD_LIMIT} words). No lists/newlines."}`;

function augmentUserPrompt(prompt) {
  return (
    `${prompt}\n\n` +
    `Respond ONLY as raw JSON (no prose/markdown). Exact format:\n${jsonHint}\n` +
    `If you cannot fully comply for any reason, still return: {"rating":1,"reason":"Cannot evaluate from the given text."}`
  );
}
function augmentUserPromptFallback(prompt) {
  return (
    `${prompt}\n\n` +
    `Return ONLY a single JSON object (no prose/markdown):\n${jsonHint}\n` +
    `If unsure, return {"rating":1,"reason":"Cannot evaluate from the given text."}`
  );
}

// ---------- Structured outputs schema for 4o/o3 ----------
const RATING_SCHEMA = {
  type: 'object',
  additionalProperties: false, // required by API
  required: ['rating', 'reason'],
  properties: {
    rating: {
      anyOf: [
        { type: 'integer', minimum: 1, maximum: 7 },
        { type: 'string',  pattern: '^[1-7]$' }
      ]
    },
    reason: { type: 'string', minLength: 1, maxLength: REASON_CHAR_LIMIT }
  }
};
const ratingFormat = (strict = true) => ({
  type: 'json_schema',
  name: 'AegyoRating',
  strict,
  schema: RATING_SCHEMA
});

// ---------- Build Responses API "input" messages ----------
const buildInputMessages = (devText, userText) => ([
  { role: 'developer', content: [{ type: 'input_text', text: devText }] },
  { role: 'user',      content: [{ type: 'input_text', text: userText }] }
]);

// ---------- Extractor for Responses API ----------
function extractFromResponse(resp) {
  // Best case: model returned parsed JSON
  if (resp.output_parsed) {
    const coerced = coerceToSchema(resp.output_parsed);
    if (coerced) return { parsed: coerced, rawText: JSON.stringify(resp.output_parsed) };
  }

  const segs = [];
  const jsons = [];
  const collectText = (text) => {
    if (typeof text !== 'string') return;
    const t = text.trim();
    if (!t) return;
    segs.push(t);
    const j = tryParseJSON(t);
    if (j) jsons.push(j);
  };
  const collectJSON = (value) => {
    if (value && typeof value === 'object') jsons.push(value);
  };
  const collectAnnotation = (ann) => {
    if (!ann) return;
    if (ann.parsed_json && typeof ann.parsed_json === 'object') jsons.push(ann.parsed_json);
    if (ann.parsed && typeof ann.parsed === 'object') jsons.push(ann.parsed);
    if (ann.json && typeof ann.json === 'object') jsons.push(ann.json);
    collectText(ann.text);
    collectText(ann.output_text);
  };
  if (Array.isArray(resp?.annotations)) {
    for (const ann of resp.annotations) collectAnnotation(ann);
  }
  const out = Array.isArray(resp.output) ? resp.output : [];

  for (const item of out) {
    collectJSON(item?.parsed);
    collectJSON(item?.json);
    if (Array.isArray(item?.annotations)) {
      for (const ann of item.annotations) collectAnnotation(ann);
    }

    if (item?.type === 'message' && Array.isArray(item.content)) {
      for (const c of item.content) {
        if (!c) continue;
        collectJSON(c.parsed);
        collectJSON(c.json);
        collectText(c.text);
        if (Array.isArray(c.annotations)) {
          for (const ann of c.annotations) collectAnnotation(ann);
        }
      }
    }
    collectText(item?.output_text);
  }

  // Prefer exact JSON candidates
  for (const cand of jsons) {
    const coerced = coerceToSchema(cand);
    if (coerced) return { parsed: coerced, rawText: JSON.stringify(cand) };
  }

  // Try salvage from concatenated text
  const rawJoined = segs.join('\n').trim();
  const salvaged = coerceToSchema(rawJoined);
  return { parsed: salvaged, rawText: rawJoined };
}

// ---------- Responses API call helper ----------
async function responsesCall({
  model,
  devText,
  userText,
  useSchema,
  forGpt5,
  strictSchema = true
}) {
  // GPT-5 uses json_object (no schema). 4o/o3 keep json_schema.
  const textObj = forGpt5
    ? (useSchema
        ? { format: { type: 'json_object' }, verbosity: GPT5_VERBOSITY }
        : { format: { type: 'text' },        verbosity: GPT5_VERBOSITY })
    : (useSchema
        ? { format: ratingFormat(strictSchema) }
        : { format: { type: 'text' } });

  const body = {
    model,
    input: buildInputMessages(devText, userText),
    ...(forGpt5 ? { reasoning: { effort: GPT5_EFFORT, summary: 'auto' } } : {}),
    text: textObj,
    max_output_tokens: OUTPUT_TOKENS,
    ...(forGpt5 ? { tools: [] } : {}),
    ...(forGpt5 ? { store: true } : {}),
    ...(forGpt5 ? { include: ['reasoning.encrypted_content', 'web_search_call.action.sources'] } : {})
  };

  const resp = await client.responses.create(body);
  const { parsed, rawText } = extractFromResponse(resp);
  const fallback = (resp.output_text || '').trim();
  return {
    parsed,
    raw_text: rawText || fallback,
    response_id: resp.id,
    request_id: resp._request_id || null
  };
}

// ---------- Unified one-call wrapper with validation ----------
async function callModelOnce(model, instructionText, originalUserPrompt) {
  // For GPT-5: json_object → plain text fallback
  if (model === 'gpt-5') {
    const attempts = [
      { useSchema: true,  userText: augmentUserPrompt(originalUserPrompt) },
      { useSchema: false, userText: augmentUserPromptFallback(originalUserPrompt) }
    ];
    let last = null;
    for (const a of attempts) {
      const r = await responsesCall({
        model: 'gpt-5',
        devText: instructionText,
        userText: a.userText,
        useSchema: a.useSchema,
        forGpt5: true
      });
      // Normalize before validating
      r.parsed = coerceToSchema(r.parsed) || coerceToSchema(r.raw_text) || null;
      if (r.parsed) return r;
      last = r;
      if (!r.parsed && r.raw_text) console.warn('[SALVAGE MISS]', r.raw_text.slice(0, 600));
    }
    const err = new Error('Empty or invalid structured output (gpt-5)');
    err._raw_text = last?.raw_text || '';
    err._response_id = last?.response_id || null;
    throw err;
  }

  // For 4o / o3: json_schema → loose → plain text
  const attempts = [
    { useSchema: true,  strictSchema: true,  userText: augmentUserPrompt(originalUserPrompt) },
    { useSchema: true,  strictSchema: false, userText: augmentUserPrompt(originalUserPrompt) + '\nReturn exactly these two keys. No others.' },
    { useSchema: false,                      userText: augmentUserPromptFallback(originalUserPrompt) }
  ];
  let last = null;
  for (const a of attempts) {
    const r = await responsesCall({
      model,
      devText: instructionText,
      userText: a.userText,
      useSchema: a.useSchema,
      strictSchema: a.strictSchema,
      forGpt5: false
    });
    r.parsed = coerceToSchema(r.parsed) || coerceToSchema(r.raw_text) || null;
    if (r.parsed) return r;
    last = r;
  }
  const err = new Error('Empty or invalid structured output');
  err._raw_text = last?.raw_text || '';
  err._response_id = last?.response_id || null;
  throw err;
}

// ---------- Load/resume ----------
function loadItems({ inputPath, outputPath, resume }) {
  const inputStr = fs.readFileSync(inputPath, 'utf8');
  const inputItems = JSON.parse(inputStr);
  if (!Array.isArray(inputItems)) throw new Error(`Input file "${inputPath}" must be a JSON array.`);
  if (resume && fs.existsSync(outputPath)) {
    try {
      const outStr = fs.readFileSync(outputPath, 'utf8');
      const outItems = JSON.parse(outStr);
      if (Array.isArray(outItems) && outItems.length === inputItems.length) {
        console.log(`Resuming from existing output: ${outputPath}`);
        return outItems;
      } else {
        console.warn(`Output length mismatch; starting from input. (${outItems?.length} vs ${inputItems.length})`);
      }
    } catch {
      console.warn(`Could not parse existing output; starting from input. (${outputPath})`);
    }
  }
  return inputItems;
}

function computeIndicesToProcess(items, { onlyMissing = true, redoDone = false }) {
  const isDone = (it) =>
    Number.isInteger(it?.rating) &&
    typeof it?.reason === 'string' &&
    it.reason.length > 0 &&
    it.format_ok === true;

  const idxs = [];
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    if (!onlyMissing) { idxs.push(i); continue; }
    if (!redoDone && isDone(it)) continue;
    idxs.push(i);
  }
  return idxs;
}

// ---------- Processor ----------
async function processFile({
  inputPath,
  model,
  outputPath,
  concurrency = 5,
  batchWriteEvery = 50,
  resume = true,
  onlyMissing = true,
  redoDone = false,
}) {
  const items = loadItems({ inputPath, outputPath, resume });

  console.log(`Loaded ${items.length} items ${resume && fs.existsSync(outputPath) ? '(resumed)' : '(from input)'} for model ${model}.`);

  const outDir = path.dirname(outputPath);
  if (outDir && outDir !== '.' && !fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const targetIdxs = computeIndicesToProcess(items, { onlyMissing, redoDone });
  const alreadyDone = items.length - targetIdxs.length;
  console.log(`Items already done: ${alreadyDone}; to process: ${targetIdxs.length}`);

  let inFlight = 0, processed = 0, cursor = 0;

  async function handleIndex(i) {
    const item = items[i];
    if (!item || typeof item !== 'object') return;
    if (!item.instruction || !item.prompt) return;

    const MAX_RETRIES = 3;
    let attempt = 0;
    while (attempt < MAX_RETRIES) {
      try {
        const { parsed, raw_text, response_id, request_id } =
          await callModelOnce(model, item.instruction, item.prompt);

        item.model = model;
        item.raw_text = raw_text;
        item.response_id = response_id;
        item.request_id = request_id;

        // Validate parsed (already normalized)
        if (!parsed || !Number.isInteger(parsed.rating) || !parsed.reason) {
          throw new Error('Parsed object missing rating/reason');
        }

        item.rating = parsed.rating;
        item.score  = parsed.rating;
        item.reason = parsed.reason;
        item.format_ok = true;
        break;
      } catch (err) {
        attempt += 1;

        if (err && err._raw_text && !item.raw_text) item.raw_text = err._raw_text;
        if (err && err._response_id) item.response_id = err._response_id;

        const backoffMs = 700 * Math.pow(2, attempt - 1) + Math.floor(Math.random() * 250);
        console.warn(`Index ${i} (attempt ${attempt}/${MAX_RETRIES}) error: ${err?.message || err}. Backoff ${backoffMs|0}ms.`);
        await sleep(backoffMs);

        if (attempt >= MAX_RETRIES) {
          item.format_ok = false;
          item.rating = item.rating ?? null;
          item.score  = item.score  ?? null;
          item.reason = item.reason ?? null;
          item.error  = String(err?.message || err);
        }
      }
    }
  }

  // Bounded concurrency
  const queue = [];
  while (cursor < targetIdxs.length || inFlight > 0) {
    while (inFlight < concurrency && cursor < targetIdxs.length) {
      const idx = targetIdxs[cursor++];
      inFlight++;
      const p = handleIndex(idx)
        .catch((e) => console.error('Unexpected handler error:', e))
        .finally(() => { inFlight--; processed++; });
      queue.push(p);
    }

    if (processed > 0 && processed % batchWriteEvery === 0) {
      fs.writeFileSync(outputPath, JSON.stringify(items, null, 2), 'utf8');
      console.log(`Wrote progress after ${processed}/${targetIdxs.length} processed -> ${outputPath}`);
    }

    await sleep(25);
  }

  // Final write
  fs.writeFileSync(outputPath, JSON.stringify(items, null, 2), 'utf8');
  console.log(`Done. Processed ${targetIdxs.length} items; wrote ${items.length} total -> ${outputPath}`);

  // Minimal ratings-only file (score first)
  const minimal = items.map((it, i) => ({
    index: i,
    model: it.model,
    instruction: it.instruction,
    prompt: it.prompt,
    score: it.score ?? it.rating ?? null,
    reason: it.reason ?? null,
  }));
  const minimalPath = outputPath.replace(/\.json$/i, '-ratings-only.json');
  fs.writeFileSync(minimalPath, JSON.stringify(minimal, null, 2), 'utf8');
  console.log(`Also wrote minimal file -> ${minimalPath}`);

  const missingAfter = items.filter(
    (it) => !(Number.isInteger(it.rating) && typeof it.reason === 'string' && it.reason.length > 0 && it.format_ok === true)
  ).length;

  return {
    total: items.length,
    already_done: alreadyDone,
    processed: targetIdxs.length,
    missing_after: missingAfter,
    outputPath,
    minimalPath,
  };
}

// ---------- Helpers ----------
function parseBool(v, dflt) {
  if (v === undefined) return dflt;
  if (typeof v === 'string') return v === '1' || v.toLowerCase() === 'true';
  return !!v;
}

// ---------- Routes ----------
app.get('/process/4o', async (req, res) => {
  try {
    const result = await processFile({
      inputPath: 'input-4o.json',
      model: 'gpt-4o',
      outputPath: 'output-4o.json',
      concurrency: Number(req.query.c || 5),
      batchWriteEvery: Number(req.query.b || 50),
      resume: parseBool(req.query.resume, true),
      onlyMissing: parseBool(req.query.onlyMissing, true),
      redoDone: parseBool(req.query.redoDone, false),
    });
    res.json(result);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e?.message || e) });
  }
});

app.get('/process/o3', async (req, res) => {
  try {
    const result = await processFile({
      inputPath: 'input-o3.json',
      model: 'o3',
      outputPath: 'output-o3.json',
      concurrency: Number(req.query.c || 5),
      batchWriteEvery: Number(req.query.b || 50),
      resume: parseBool(req.query.resume, true),
      onlyMissing: parseBool(req.query.onlyMissing, true),
      redoDone: parseBool(req.query.redoDone, false),
    });
    res.json(result);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e?.message || e) });
  }
});

app.get('/process/5', async (req, res) => {
  try {
    const result = await processFile({
      inputPath: 'input-5.json',
      model: 'gpt-5',
      outputPath: 'output-5.json',
      concurrency: Number(req.query.c || 5),
      batchWriteEvery: Number(req.query.b || 50),
      resume: parseBool(req.query.resume, true),
      onlyMissing: parseBool(req.query.onlyMissing, true),
      redoDone: parseBool(req.query.redoDone, false),
    });
    res.json(result);
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: String(e?.message || e) });
  }
});

app.listen(port, () => {
  console.log(`Server listening on http://localhost:${port}`);
  console.log(`Reason length: <= ${REASON_WORD_LIMIT} words; schema cap: ${REASON_CHAR_LIMIT} chars; tokens: ${OUTPUT_TOKENS}`);
});

