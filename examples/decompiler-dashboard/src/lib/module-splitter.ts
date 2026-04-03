/**
 * Browser port of scripts/lib/module-splitter.mjs
 * Splits a bundled JS file into logical modules by keyword matching.
 */

import type { VersionMetrics, ModuleMetrics } from '../types';

const MODULE_KEYWORDS: Record<string, string[]> = {
  'tool-dispatch': [
    'BashTool', 'FileReadTool', 'FileEditTool', 'FileWriteTool',
    'AgentOutputTool', 'WebFetch', 'WebSearch', 'TodoWrite',
    'NotebookEdit', 'GlobTool', 'GrepTool',
  ],
  'permission-system': [
    'canUseTool', 'alwaysAllowRules', 'denyWrite',
    'Permission', 'permission',
  ],
  'mcp-client': [
    'mcp__', 'McpClient', 'McpServer', 'McpError',
    'callTool', 'listTools',
  ],
  'streaming-handler': [
    'content_block_delta', 'message_start', 'message_stop',
    'message_delta', 'content_block_start', 'content_block_stop',
    'stream_event', 'text_delta', 'input_json_delta',
  ],
  'context-manager': [
    'tengu_compact', 'microcompact', 'auto_compact',
    'compact_boundary', 'preCompactTokenCount',
    'postCompactTokenCount', 'compaction',
  ],
  'agent-loop': [
    'agentLoop', 'mainLoop', 'querySource',
    'toolUseContext', 'systemPrompt',
  ],
};

const SIMPLE_PATTERNS: Record<string, RegExp> = {
  telemetry: /"tengu_[^"]*"/g,
  commands: /name:"[a-z][-a-z]*",description:"[^"]*"/g,
  'class-hierarchy': /class \w+( extends \w+)?/g,
};

function splitStatements(source: string): string[] {
  const MAX_CHUNK = 2048;
  const raw = source.split(';');
  const chunks: string[] = [];
  let buffer = '';

  for (const part of raw) {
    if (buffer.length + part.length > MAX_CHUNK && buffer.length > 0) {
      chunks.push(buffer);
      buffer = part;
    } else {
      buffer += (buffer ? ';' : '') + part;
    }
  }
  if (buffer.length > 0) chunks.push(buffer);
  return chunks;
}

function classifyStatements(statements: string[]): Record<string, string[]> {
  const modules: Record<string, string[]> = {};

  for (const stmt of statements) {
    if (stmt.length < 10) continue;
    for (const [modName, keywords] of Object.entries(MODULE_KEYWORDS)) {
      const matched = keywords.some((kw) => stmt.includes(kw));
      if (matched) {
        if (!modules[modName]) modules[modName] = [];
        modules[modName].push(stmt.trim());
        break;
      }
    }
  }
  return modules;
}

function extractSimplePatterns(source: string): Record<string, string[]> {
  const results: Record<string, string[]> = {};

  for (const [modName, pattern] of Object.entries(SIMPLE_PATTERNS)) {
    pattern.lastIndex = 0;
    const matches = new Set<string>();
    let m;
    while ((m = pattern.exec(source)) !== null) {
      const frag = m[0].trim();
      if (frag.length > 3) matches.add(frag);
    }
    if (matches.size > 0) {
      results[modName] = [...matches];
    }
  }
  return results;
}

export function computeMetrics(source: string, fileName: string): Omit<VersionMetrics, 'modules'> {
  const sizeBytes = new Blob([source]).size;
  const versionMatch = source.match(/VERSION[=:]"?(\d+\.\d+\.\d+)/);
  const version = versionMatch ? versionMatch[1] : 'unknown';

  return {
    version,
    sizeBytes,
    lines: source.split('\n').length,
    functions: (source.match(/function\s*\w*\s*\(/g) || []).length,
    asyncFunctions: (source.match(/async\s+function/g) || []).length,
    arrowFunctions: (source.match(/=>/g) || []).length,
    classes: (source.match(/class \w+/g) || []).length,
    extends: (source.match(/extends \w+/g) || []).length,
    sourceFile: fileName,
    extractedAt: new Date().toISOString(),
  };
}

export interface SplitResult {
  metrics: VersionMetrics;
  modules: Record<string, string>;
}

export function splitBundle(source: string, fileName: string): SplitResult {
  const baseMetrics = computeMetrics(source, fileName);

  const statements = splitStatements(source);
  const classified = classifyStatements(statements);
  const moduleResults: Record<string, ModuleMetrics> = {};
  const modules: Record<string, string> = {};

  for (const [modName, fragments] of Object.entries(classified)) {
    const content = fragments.join('\n\n');
    modules[modName] = content;
    moduleResults[modName] = {
      fragments: fragments.length,
      sizeBytes: new Blob([content]).size,
    };
  }

  const simple = extractSimplePatterns(source);
  for (const [modName, fragments] of Object.entries(simple)) {
    const content = fragments.join('\n');
    modules[modName] = content;
    moduleResults[modName] = {
      fragments: fragments.length,
      sizeBytes: new Blob([content]).size,
    };
  }

  return {
    metrics: { ...baseMetrics, modules: moduleResults },
    modules,
  };
}
