// apify.js
import { BasicCrawler, log } from 'crawlee';
import { chromium } from 'playwright-core';
import { spawn } from 'child_process';
import { URL } from 'url';

const BROWSERLESS_WS = process.env.BROWSERLESS_WS;
if (!BROWSERLESS_WS) {
  throw new Error('Missing BROWSERLESS_WS env var (e.g., wss://...browserless.io?token=YOUR_TOKEN)');
}

const PRIORITY_KEYWORDS = [
  "course","courses","curriculum","degree","syllabus","module","program",
  "programme","bsc","msc","master","bachelor","studyguide","study","education",
  "undergraduate","postgraduate","kurser","kurs","utbildning","studieplan","kandidat",
  "magister","kursplan","mooc","studiengang","studiengange","modul","modulhandbuch",
  "lehrveranstaltung","licence","etudes","formation","maquette","enseignement",
  "corso","corsi","programma","laurea","triennale","magistrale","insegnamento",
  "piano di studi","grado","licenciatura","maestria","master","estudios","asignatura",
  "plano de estudos","estudo","opleiding","cursus","cursussen","studieprogramma",
  "studiegids","kurssi","kurssit","opinto","opinto-opas","kandidaatti","maisteri",
  "opinto-ohjelma","kursus","kurser","uddannelse","studieordning","studieprogram",
  "utdanning","emne","studieplan","catalog","course-syllabus","studier"
];

const EXCLUDE_KEYWORDS = [
  "scholarship","scholarships","events","funding","tuition","fees","alumni","donate",
  "conference","news","about","staff","stories","how","why","apply","search","your",
  "registration","login","register","menu","fellowship","person","professor","contact","cookie"
];

const seen = new Set();
const MAX_PARALLEL_PYTHON = 30;
let runningPython = 0;
const pythonQueue = new Set();
const urlQueue = [];

function normalizeUrlForJson(u) {
  const url = new URL(u);
  if (url.hash && url.hash.length > 1) url.hash = '';
  return url.href;
}

function matchesPriorityKeyword(url) {
  const s = url.toLowerCase();
  return PRIORITY_KEYWORDS.some(k => s.includes(k));
}
function matchesExcludeKeyword(url) {
  const s = url.toLowerCase();
  return EXCLUDE_KEYWORDS.some(k => s.includes(k));
}

function runPipeline(url) {
  const norm = normalizeUrlForJson(url);
  if (pythonQueue.has(norm)) {
    log.info(`?? Skipping duplicate (same base): ${url}`);
    return;
  }
  log.info(`? Queued for pipeline: ${url} (normalized: ${norm})`);
  pythonQueue.add(norm);
  urlQueue.push(url);
  checkQueue();
}
function checkQueue() {
  while (runningPython < MAX_PARALLEL_PYTHON && urlQueue.length > 0) {
    const nextUrl = urlQueue.shift();
    processQueue(nextUrl);
  }
}
function processQueue(url) {
  runningPython++;
  log.info(`? Running pipeline on: ${url}`);
  const child = spawn('python', ['test_pipeline.py', url]);

  child.stdout.on('data', (d) => log.info(`PY> ${d.toString().trim()}`));
  child.stderr.on('data', (d) => log.warning(`PYERR> ${d.toString().trim()}`));
  child.on('close', (code) => {
    log.info(`¦ Pipeline finished for ${url} (exit ${code})`);
    runningPython--;
    checkQueue();
  });
}

log.info('Connecting to Browserless...');
const browser = await chromium.connectOverCDP(BROWSERLESS_WS);

let defaultContext = browser.contexts()[0];
if (!defaultContext) {
  defaultContext = await browser.newContext?.();
  if (!defaultContext) {
    throw new Error('No browser context available after CDP connect');
  }
}

const crawler = new BasicCrawler({
  maxConcurrency: 8,
  requestHandlerTimeoutSecs: 120,

  async requestHandler({ request, addRequests }) {
    const url = request.url;
    log.info(`?? Visiting: ${url}`);

    const page = await defaultContext.newPage();
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45_000 }).catch(() => {});

      try {
        const cookieBtn = page.locator(
          '#onetrust-accept-btn-handler, button:has-text("Accept"), button:has-text("Agree"), button:has-text("Allow")'
        ).first();
        if (await cookieBtn.count()) await cookieBtn.click({ timeout: 2_000 }).catch(() => {});
      } catch {}

      await page.waitForSelector('main, #content, [role="main"]', { state: 'attached', timeout: 20_000 }).catch(() => {});

      if (matchesPriorityKeyword(url) && !matchesExcludeKeyword(url)) {
        runPipeline(url);
      }

      const allLinks = await page.evaluate(() => {
        const scopes = document.querySelectorAll('main, #content, [role="main"]');
        const scopeEls = scopes.length ? scopes : [document.body];
        const anchors = [];
        scopeEls.forEach(sc => anchors.push(...sc.querySelectorAll('a[href]')));

        const isVisible = (el) => {
          const cs = getComputedStyle(el);
          if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
          const r = el.getBoundingClientRect();
          if (r.width === 0 && r.height === 0) return false;
          if (el.closest('[aria-hidden="true"], [hidden], .mm-menu, .mm-listitem, nav[aria-label*="Mobile" i]')) return false;
          return true;
        };

        const out = new Set();
        for (const a of anchors) {
          const href = a.getAttribute('href') || '';
          if (!href || href === '#' || href.startsWith('javascript:') || href.startsWith('mailto:') || href.startsWith('tel:')) continue;
          if (!isVisible(a)) continue;
          try { out.add(new URL(href, location.href).href); } catch {}
        }
        return Array.from(out);
      });

      const filteredLinks = allLinks.filter(link =>
        link &&
        !seen.has(link) &&
        !matchesExcludeKeyword(link) &&
        !link.toLowerCase().includes('login') &&
        !link.toLowerCase().includes('mailto:')
      );

      filteredLinks.forEach(link => seen.add(link));
      log.info(`?? Extracted ${filteredLinks.length} filtered links`);

      const hereHost = new URL(url).hostname;
      await addRequests(
        filteredLinks
          .filter(u => {
            try { return new URL(u).hostname === hereHost; } catch { return false; }
          })
          .map(u => ({ url: u }))
      );

      for (const link of filteredLinks) {
        if (matchesPriorityKeyword(link) && !matchesExcludeKeyword(link)) {
          log.info(`? Priority match: ${link}`);
          runPipeline(link);
        }
      }
    } finally {
      await page.close().catch(() => {});
    }
  },
});

const startUrls = process.argv.slice(2);
if (startUrls.length === 0) {
  log.error('No start URLs provided. Exiting.');
  log.error('Usage: node apify.js <url1> [url2] ...');
  process.exit(1);
}

log.info(`?? Starting crawler with URLs: ${startUrls.join(', ')}`);
await crawler.run(startUrls);
log.info('? Crawler finished its run.');

// await browser.close();
