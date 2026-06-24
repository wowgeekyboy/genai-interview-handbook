// build.mjs — Transform the handbook markdown into a fast, structured website.
// Splits the single markdown file (fence-aware) into per-PART HTML fragments,
// builds a navigation manifest + client search index, copies the UI shell,
// and "softens" SHOUTY ALL-CAPS headings into clean Title Case while keeping
// acronyms (RAG, LLM…) and lowercase repo paths intact.
import { marked } from 'marked';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = __dirname;
const SRC = path.join(ROOT, 'src');
const DOCS = path.join(ROOT, 'docs');
const MD_FILE = path.join(ROOT, 'GenAI_AI_Engineering_Complete_Handbook.md');

// Acronyms kept fully uppercase
const ACRONYMS = new Set(('AI ML DL LLM RAG CRAG MCP PII PHI API HNSW OWASP BM25 RRF SQL JD ' +
  'GDPR SSE CLI SDK IOT CPU GPU KV MOE LORA QLORA DPO RLHF RLAIF PPO NLP CLIP VAE GAN BERT GPT ' +
  'T5 PCA FAQ PDF OAUTH FSDP HIPAA PCI DSS EU OS UI UX TOC STAR ADR JSON HTML CSS REST CI CD NER ' +
  'IVF PQ TPU CUDA HPC VLM GGUF SMS MQTT RASA TTFT ZERO DDP GRPO RMSNORM ROPE GQA SLM LRM BPE ' +
  'XSS A2A DSPY RAGAS OTEL SGLANG VLLM TRL').split(/\s+/));

// Exact mixed-case brand/term forms (keyed by uppercased input)
const SPECIAL = {
  GENAI: 'GenAI', LLMS: 'LLMs', APIS: 'APIs', GPUS: 'GPUs', PDFS: 'PDFs', JDS: 'JDs',
  LANGGRAPH: 'LangGraph', LANGCHAIN: 'LangChain', LANGSMITH: 'LangSmith', LANGFUSE: 'Langfuse',
  GITHUB: 'GitHub', HIPPORAG: 'HippoRAG', GRAPHRAG: 'GraphRAG', SELFRAG: 'SelfRAG',
  OPENAI: 'OpenAI', POSTGRESQL: 'PostgreSQL', POSTGRES: 'Postgres', MONGODB: 'MongoDB',
  NODEJS: 'NodeJS', REACTJS: 'ReactJS', NEXTJS: 'NextJS', FASTAPI: 'FastAPI', QDRANT: 'Qdrant',
  PINECONE: 'Pinecone', WEAVIATE: 'Weaviate', CHROMA: 'Chroma', PGVECTOR: 'pgvector',
  PYTORCH: 'PyTorch', TENSORFLOW: 'TensorFlow', HUGGINGFACE: 'HuggingFace', BRAINTRUST: 'Braintrust',
  DEEPEVAL: 'DeepEval', DATADOG: 'Datadog', SIGNOZ: 'Signoz', HONCHO: 'Honcho', PRESIDIO: 'Presidio',
  TWILIO: 'Twilio', BEDROCK: 'Bedrock', AZURE: 'Azure', VERTEXAI: 'VertexAI', GEMINI: 'Gemini',
  CLAUDE: 'Claude', LLAMA: 'Llama', MISTRAL: 'Mistral', MEGATRON: 'Megatron', DEEPSPEED: 'DeepSpeed',
  MAMBA: 'Mamba', WORDPIECE: 'WordPiece', SENTENCEPIECE: 'SentencePiece', HYDE: 'HyDE',
  COLBERT: 'ColBERT', SPLADE: 'SPLADE', AUTOGEN: 'AutoGen', CREWAI: 'CrewAI', LCEL: 'LCEL',
  RLBASED: 'RL-based', CHATGPT: 'ChatGPT', LLMOPS: 'LLMOps', MLOPS: 'MLOps',
};

const SMALLWORDS = new Set(['of', 'the', 'from', 'and', 'for', 'a', 'an', 'to', 'with', 'in',
  'on', 'vs', 'or', 'at', 'by', 'is', 'as', 'but', 'per']);

function softenCore(core, isFirst, allowSmall = true) {
  if (!core) return core;
  const up = core.toUpperCase();
  if (SPECIAL[up]) return SPECIAL[up];
  if (ACRONYMS.has(up)) return up;
  if (!/[A-Z]/.test(core)) return core;              // numbers / symbols only
  if (core !== up) return core;                       // already mixed/lowercase → keep
  if (core.includes('&')) return core.split('&').map((p, i) => softenCore(p, isFirst && i === 0, false)).join('&');
  if (core.includes('-')) return core.split('-').map((p, i) => softenCore(p, isFirst && i === 0, false)).join('-');
  const low = core.toLowerCase();
  if (allowSmall && SMALLWORDS.has(low) && !isFirst) return low;
  return low.charAt(0).toUpperCase() + low.slice(1);
}
function softenToken(tok, isFirst) {
  const lead = (tok.match(/^[^A-Za-z0-9]+/) || [''])[0];
  const rest = tok.slice(lead.length);
  const trail = (rest.match(/[^A-Za-z0-9]+$/) || [''])[0];
  const core = trail ? rest.slice(0, rest.length - trail.length) : rest;
  return lead + softenCore(core, isFirst) + trail;
}
function softenHeading(text) {
  let firstSeen = false;
  return text.split(/(\s+)/).map(t => {
    if (t === '' || /^\s+$/.test(t)) return t;
    const isFirst = !firstSeen; firstSeen = true;
    return softenToken(t, isFirst);
  }).join('');
}

function slugify(text) {
  return String(text).toLowerCase()
    .replace(/<[^>]+>/g, '').replace(/&[a-z]+;/g, '')
    .replace(/[^a-z0-9\s-]/g, '').trim()
    .replace(/\s+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '') || 'section';
}
function stripTags(html) {
  return html.replace(/<[^>]+>/g, '').replace(/&amp;/g, '&').replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/\s+/g, ' ').trim();
}

// --- Read the markdown -----------------------------------------------------
const md = fs.readFileSync(MD_FILE, 'utf8');
const rawLines = md.split('\n');

// Pre-pass: soften heading lines (fence-aware), leaving code/comments untouched.
const lines = [];
{
  let inFence = false, fm = '';
  for (const line of rawLines) {
    const f = line.match(/^(\s*)(```+|~~~+)/);
    if (f) { const m = f[2][0]; if (!inFence) { inFence = true; fm = m; } else if (m === fm) { inFence = false; fm = ''; } lines.push(line); continue; }
    if (!inFence) {
      const hm = line.match(/^(#{1,6})\s+(.*)$/);
      if (hm) { lines.push(hm[1] + ' ' + softenHeading(hm[2])); continue; }
    }
    lines.push(line);
  }
}

// --- Split into chunks by top-level PART heading (fence-aware) --------------
const chunks = [];
let current = null;
{
  let inFence = false, fm = '';
  for (const line of lines) {
    const f = line.match(/^(\s*)(```+|~~~+)/);
    if (f) { const m = f[2][0]; if (!inFence) { inFence = true; fm = m; } else if (m === fm) { inFence = false; fm = ''; } }
    const isTop = !inFence && /^#\s+/.test(line);
    if (isTop) {
      const text = line.replace(/^#\s+/, '').trim();
      if (current === null) current = { headingRaw: text, body: [line] };
      else if (/^part\b/i.test(text)) { chunks.push(current); current = { headingRaw: text, body: [line] }; }
      else current.body.push(line);
    } else {
      if (current === null) current = { headingRaw: 'Introduction', body: [] };
      current.body.push(line);
    }
  }
  if (current) chunks.push(current);
}

// --- marked config + heading id injection ----------------------------------
marked.setOptions({ gfm: true, breaks: false });
function injectHeadingIds(html) {
  return html.replace(/<h([2-6])>([\s\S]*?)<\/h\1>/g, (m, lvl, inner) => {
    const slug = slugify(stripTags(inner));
    return `<h${lvl} id="${slug}"><a class="anchor" href="#${slug}" aria-label="Link to section">#</a>${inner}</h${lvl}>`;
  });
}

// --- Build fragments + manifest + search index -----------------------------
const sections = [];
const searchIndex = [];
let totalWords = 0;

chunks.forEach((chunk, i) => {
  const raw = chunk.headingRaw;
  let id, num, title;
  const pm = raw.match(/^part\s+(\d+)\s*[:\-]\s*(.+)$/i);
  if (pm) { num = pm[1]; id = `part-${num}`; title = pm[2].trim(); }
  else if (i === 0) { id = 'intro'; num = ''; title = 'Start Here'; }
  else { num = ''; id = slugify(raw); title = raw; }

  const bodyText = chunk.body.join('\n');
  totalWords += bodyText.split(/\s+/).filter(Boolean).length;

  // ## subsections (fence-aware)
  const subs = [];
  { let inF = false, fm = '';
    for (const line of chunk.body) {
      const f = line.match(/^(\s*)(```+|~~~+)/);
      if (f) { const m = f[2][0]; if (!inF) { inF = true; fm = m; } else if (m === fm) { inF = false; fm = ''; } continue; }
      if (!inF) { const h2 = line.match(/^##\s+(.+)$/); if (h2) subs.push({ title: h2[1].trim(), slug: slugify(h2[1].trim()) }); }
    }
  }

  let html = injectHeadingIds(marked.parse(bodyText));
  const file = `part-${String(i).padStart(2, '0')}.html`;
  fs.writeFileSync(path.join(DOCS, 'data', 'parts', file), html, 'utf8');
  sections.push({ id, num, title, file, subs });

  // search: subsections
  for (const block of bodyText.split(/\n(?=##\s+)/)) {
    const h = block.match(/^##\s+(.+)$/m);
    if (!h) continue;
    const plain = stripTags(marked.parse(block.replace(/^##\s+.+$/m, ''))).slice(0, 360);
    searchIndex.push({ part: id, partTitle: title, type: 'section', title: h[1].trim(), slug: slugify(h[1].trim()), file, text: plain });
  }
  searchIndex.push({ part: id, partTitle: title, type: 'part', title, slug: '', file, text: stripTags(marked.parse(chunk.body.slice(0, 30).join('\n'))).slice(0, 240) });
});

// --- Stats -----------------------------------------------------------------
const stats = {
  lines: rawLines.length,
  words: totalWords,
  parts: sections.filter(s => s.num).length,
  questions: (md.match(/\bQ\d+:/g) || []).length,
  codeBlocks: Math.round((md.match(/```/g) || []).length / 2),
  diagrams: lines.filter(l => /[┌└┐┘├┤┬┴┼]/.test(l)).length,
};

const manifest = {
  title: 'The Complete GenAI & AI Engineering Handbook',
  tagline: 'From fundamentals to production — crack any AI/GenAI engineering interview.',
  repo: 'https://github.com/wowgeekyboy/genai-interview-handbook',
  generatedAt: new Date().toISOString().slice(0, 10),
  stats, sections,
};
fs.writeFileSync(path.join(DOCS, 'data', 'manifest.json'), JSON.stringify(manifest, null, 2));
fs.writeFileSync(path.join(DOCS, 'data', 'search-index.json'), JSON.stringify(searchIndex));

// --- Copy UI shell + assets ------------------------------------------------
fs.copyFileSync(path.join(SRC, 'template.html'), path.join(DOCS, 'index.html'));
fs.copyFileSync(path.join(SRC, 'styles.css'), path.join(DOCS, 'assets', 'css', 'styles.css'));
fs.copyFileSync(path.join(SRC, 'app.js'), path.join(DOCS, 'assets', 'js', 'app.js'));
if (fs.existsSync(path.join(SRC, 'og-image.svg'))) fs.copyFileSync(path.join(SRC, 'og-image.svg'), path.join(DOCS, 'assets', 'og-image.svg'));
fs.copyFileSync(MD_FILE, path.join(DOCS, 'GenAI_AI_Engineering_Complete_Handbook.md'));
fs.writeFileSync(path.join(DOCS, '.nojekyll'), '');

console.log('✓ Build complete');
console.log(`  Sections: ${sections.length} (${stats.parts} parts), ${searchIndex.length} search entries`);
console.log(`  ${stats.lines} lines · ~${stats.words.toLocaleString()} words · ${stats.questions} Q's · ${stats.codeBlocks} code blocks`);
sections.forEach(s => console.log(`   ${(s.num || '•').padStart(2)}  ${s.title}`));
