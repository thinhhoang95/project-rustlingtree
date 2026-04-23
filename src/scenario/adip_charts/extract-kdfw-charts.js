#!/usr/bin/env node
// Usage: node src/scenario/adip_resources/extract-kdfw-charts.js data/adip/kdfw_adip_resources.mhtml --out data/adip/kdfw_adip_resources.json

const fs = require('fs');
const path = require('path');

function decodeQuotedPrintable(input) {
  return input
    .replace(/=\r?\n/g, '')
    .replace(/=([0-9A-Fa-f]{2})/g, (_, hex) =>
      String.fromCharCode(parseInt(hex, 16))
    );
}

function decodeHtmlEntities(input) {
  const named = {
    amp: '&',
    lt: '<',
    gt: '>',
    quot: '"',
    apos: "'",
    nbsp: ' ',
  };

  return input.replace(/&(#x[0-9A-Fa-f]+|#\d+|[A-Za-z]+);/g, (_, entity) => {
    if (entity[0] === '#') {
      const codePoint =
        entity[1].toLowerCase() === 'x'
          ? parseInt(entity.slice(2), 16)
          : parseInt(entity.slice(1), 10);
      if (Number.isFinite(codePoint)) {
        return String.fromCodePoint(codePoint);
      }
      return _;
    }

    return Object.prototype.hasOwnProperty.call(named, entity) ? named[entity] : _;
  });
}

function normalizeChartName(label) {
  return decodeHtmlEntities(label.trim())
    .replace(/,/g, '')
    .replace(/\s+/g, '_')
    .replace(/_+/g, '_');
}

function extractHtmlPart(mhtml) {
  const boundaryMatch = mhtml.match(/boundary="([^"]+)"/i);
  if (!boundaryMatch) {
    throw new Error('Could not find multipart boundary in MHTML file.');
  }

  const boundary = `--${boundaryMatch[1]}`;
  const parts = mhtml.split(boundary);
  const htmlPart = parts.find((part) => /Content-Type:\s*text\/html/i.test(part));

  if (!htmlPart) {
    throw new Error('Could not find the HTML part in the MHTML file.');
  }

  const bodyStart = htmlPart.indexOf('\r\n\r\n');
  const bodyStartFallback = htmlPart.indexOf('\n\n');
  const startIndex =
    bodyStart !== -1 ? bodyStart + 4 : bodyStartFallback !== -1 ? bodyStartFallback + 2 : -1;

  if (startIndex === -1) {
    throw new Error('Could not locate the start of the HTML body.');
  }

  return decodeQuotedPrintable(htmlPart.slice(startIndex));
}

function extractSection(html, heading, endMarker) {
  const start = html.indexOf(heading);
  if (start === -1) {
    throw new Error(`Could not find section heading: ${heading}`);
  }

  const end = html.indexOf(endMarker, start);
  if (end === -1) {
    throw new Error(`Could not find section end marker: ${endMarker}`);
  }

  return html.slice(start, end);
}

function extractChartsFromSection(section) {
  const charts = [];
  const seen = new Set();
  const anchorRe = /<a\s+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;

  for (const match of section.matchAll(anchorRe)) {
    const url = decodeHtmlEntities(match[1].trim());
    const label = decodeHtmlEntities(match[2].replace(/<[^>]+>/g, '').trim());

    if (!url || !label) {
      continue;
    }

    const key = normalizeChartName(label);
    const dedupeKey = `${key}::${url}`;
    if (seen.has(dedupeKey)) {
      continue;
    }

    seen.add(dedupeKey);
    charts.push({ [key]: url });
  }

  return charts;
}

function parseCharts(mhtml) {
  const html = extractHtmlPart(mhtml);

  const starSection = extractSection(
    html,
    'Standard Terminal Arrival (STAR) Charts',
    '<!-- end ngIf: starCharts.length -->'
  );
  const iapSection = extractSection(
    html,
    'Instrument Approach Procedure (IAP) Charts',
    '<!-- end ngIf: iapCharts.length -->'
  );

  return {
    star: extractChartsFromSection(starSection),
    iap: extractChartsFromSection(iapSection),
  };
}

function main() {
  const args = process.argv.slice(2);
  const inputPath = args[0] && !args[0].startsWith('-')
    ? args[0]
    : path.join(process.cwd(), 'data/adip/kdfw_adip_resources.mhtml');

  let outputPath = null;
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === '--out' || args[i] === '-o') {
      outputPath = args[i + 1];
    }
  }

  const raw = fs.readFileSync(inputPath, 'utf8');
  const result = parseCharts(raw);
  const json = `${JSON.stringify(result, null, 2)}\n`;

  if (outputPath) {
    fs.writeFileSync(outputPath, json);
  } else {
    process.stdout.write(json);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  decodeQuotedPrintable,
  decodeHtmlEntities,
  extractHtmlPart,
  extractSection,
  extractChartsFromSection,
  parseCharts,
};
