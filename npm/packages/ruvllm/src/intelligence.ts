/**
 * External Intelligence Providers for SONA Learning (ADR-043)
 *
 * TypeScript bindings for the IntelligenceProvider trait, enabling
 * external systems to feed quality signals into RuvLLM's learning loops.
 *
 * @example
 * ```typescript
 * import { IntelligenceLoader, FileSignalProvider, QualitySignal } from '@ruvector/ruvllm';
 *
 * const loader = new IntelligenceLoader();
 * loader.registerProvider(new FileSignalProvider('./signals.json'));
 *
 * const { signals, errors } = loader.loadAllSignals();
 * console.log(`Loaded ${signals.length} signals`);
 * ```
 */

/**
 * A quality signal from an external system.
 *
 * Represents one completed task with quality assessment data
 * that can feed into SONA trajectories, the embedding classifier,
 * and model router calibration.
 */
export interface QualitySignal {
  /** Unique identifier for this signal */
  id: string;
  /** Human-readable task description (used for embedding generation) */
  taskDescription: string;
  /** Execution outcome */
  outcome: 'success' | 'partial_success' | 'failure';
  /** Composite quality score (0.0 - 1.0) */
  qualityScore: number;
  /** Optional human verdict */
  humanVerdict?: 'approved' | 'rejected';
  /** Optional structured quality factors for detailed analysis */
  qualityFactors?: QualityFactors;
  /** ISO 8601 timestamp of task completion */
  completedAt: string;
}

/**
 * Granular quality factor breakdown.
 *
 * Not all providers will have all factors. Undefined fields mean
 * "not assessed" (distinct from 0.0, which means "assessed as zero").
 */
export interface QualityFactors {
  acceptanceCriteriaMet?: number;
  testsPassing?: number;
  noRegressions?: number;
  lintClean?: number;
  typeCheckClean?: number;
  followsPatterns?: number;
  contextRelevance?: number;
  reasoningCoherence?: number;
  executionEfficiency?: number;
}

/**
 * Quality weight overrides from a provider.
 *
 * Weights should sum to approximately 1.0.
 */
export interface ProviderQualityWeights {
  taskCompletion: number;
  codeQuality: number;
  process: number;
}

/**
 * Error from a single provider during batch loading.
 */
export interface ProviderError {
  providerName: string;
  message: string;
}

/**
 * Result from a single provider during grouped loading.
 */
export interface ProviderResult {
  providerName: string;
  signals: QualitySignal[];
  weights?: ProviderQualityWeights;
}

/**
 * Interface for external systems that supply quality signals to RuvLLM.
 *
 * Implement this interface and register with IntelligenceLoader.
 */
export interface IntelligenceProvider {
  /** Human-readable name for this provider */
  name(): string;
  /** Load quality signals from this provider's data source */
  loadSignals(): QualitySignal[];
  /** Optional quality weight overrides */
  qualityWeights?(): ProviderQualityWeights | undefined;
}

/**
 * Built-in file-based intelligence provider.
 *
 * Reads quality signals from a JSON file. This is the default provider
 * for non-Rust integrations that write signal files.
 */
export class FileSignalProvider implements IntelligenceProvider {
  private readonly filePath: string;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  name(): string {
    return 'file-signals';
  }

  loadSignals(): QualitySignal[] {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      const fs = require('fs');
      if (!fs.existsSync(this.filePath)) {
        return [];
      }
      const raw = fs.readFileSync(this.filePath, 'utf-8');
      const data = JSON.parse(raw);
      if (!Array.isArray(data)) {
        return [];
      }
      return data.map((item: Record<string, unknown>) => ({
        id: String(item.id ?? ''),
        taskDescription: String(item.task_description ?? item.taskDescription ?? ''),
        outcome: String(item.outcome ?? 'failure') as QualitySignal['outcome'],
        qualityScore: Number(item.quality_score ?? item.qualityScore ?? 0),
        humanVerdict: item.human_verdict ?? item.humanVerdict
          ? String(item.human_verdict ?? item.humanVerdict) as QualitySignal['humanVerdict']
          : undefined,
        qualityFactors: (item.quality_factors || item.qualityFactors)
          ? mapQualityFactors((item.quality_factors ?? item.qualityFactors) as Record<string, unknown>)
          : undefined,
        completedAt: String(item.completed_at ?? item.completedAt ?? new Date().toISOString()),
      }));
    } catch {
      return [];
    }
  }

  qualityWeights(): ProviderQualityWeights | undefined {
    try {
      const fs = require('fs');
      const path = require('path');
      const weightsPath = path.join(path.dirname(this.filePath), 'quality-weights.json');
      if (!fs.existsSync(weightsPath)) return undefined;
      const raw = fs.readFileSync(weightsPath, 'utf-8');
      const data = JSON.parse(raw);
      return {
        taskCompletion: Number(data.task_completion ?? data.taskCompletion ?? 0.5),
        codeQuality: Number(data.code_quality ?? data.codeQuality ?? 0.3),
        process: Number(data.process ?? 0.2),
      };
    } catch {
      return undefined;
    }
  }
}

function mapQualityFactors(raw: Record<string, unknown>): QualityFactors {
  return {
    acceptanceCriteriaMet: raw.acceptance_criteria_met as number | undefined,
    testsPassing: raw.tests_passing as number | undefined,
    noRegressions: raw.no_regressions as number | undefined,
    lintClean: raw.lint_clean as number | undefined,
    typeCheckClean: raw.type_check_clean as number | undefined,
    followsPatterns: raw.follows_patterns as number | undefined,
    contextRelevance: raw.context_relevance as number | undefined,
    reasoningCoherence: raw.reasoning_coherence as number | undefined,
    executionEfficiency: raw.execution_efficiency as number | undefined,
  };
}

/**
 * Aggregates quality signals from multiple registered providers.
 *
 * If no providers are registered, loadAllSignals returns empty arrays
 * with zero overhead.
 */
export class IntelligenceLoader {
  private providers: IntelligenceProvider[] = [];

  /** Register an external intelligence provider */
  registerProvider(provider: IntelligenceProvider): void {
    this.providers.push(provider);
  }

  /** Returns the number of registered providers */
  get providerCount(): number {
    return this.providers.length;
  }

  /** Returns the names of all registered providers */
  get providerNames(): string[] {
    return this.providers.map(p => p.name());
  }

  /**
   * Load signals from all registered providers.
   *
   * Non-fatal: if a provider fails, its error is captured but
   * other providers continue loading.
   */
  loadAllSignals(): { signals: QualitySignal[]; errors: ProviderError[] } {
    const signals: QualitySignal[] = [];
    const errors: ProviderError[] = [];

    for (const provider of this.providers) {
      try {
        const providerSignals = provider.loadSignals();
        signals.push(...providerSignals);
      } catch (e) {
        errors.push({
          providerName: provider.name(),
          message: e instanceof Error ? e.message : String(e),
        });
      }
    }

    return { signals, errors };
  }

  /** Load signals grouped by provider with weight overrides */
  loadGrouped(): ProviderResult[] {
    return this.providers.map(provider => {
      let providerSignals: QualitySignal[] = [];
      try {
        providerSignals = provider.loadSignals();
      } catch {
        // Non-fatal
      }
      return {
        providerName: provider.name(),
        signals: providerSignals,
        weights: provider.qualityWeights?.(),
      };
    });
  }
}
