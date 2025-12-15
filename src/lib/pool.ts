/**
 * AI Pool - Global concurrency management for AI API calls
 *
 * Features:
 * - Configurable max concurrency (semaphore-based)
 * - Adaptive backoff on rate limiting (429 errors)
 * - Automatic retries with exponential backoff
 * - Model abstraction
 */

import { generateObject, GenerateObjectResult } from 'ai';
import { google } from '@ai-sdk/google';
import { z } from 'zod';

// ============================================
// Types
// ============================================

export type PoolConfig = {
  /** Max concurrent API calls */
  maxConcurrency: number;
  /** Model name (e.g., 'gemini-3-pro-preview') */
  model: string;
  /** Base delay for retries in ms (default: 1000) */
  baseDelayMs?: number;
  /** Max retry attempts (default: 4) */
  maxRetries?: number;
  /** Enable debug logging */
  debug?: boolean;
};

export type GenerateOptions<T extends z.ZodType> = {
  schema: T;
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string | Array<{ type: 'text'; text: string } | { type: 'image'; image: Buffer }>;
  }>;
  temperature?: number;
};

type QueuedTask<T> = {
  execute: () => Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
};

// ============================================
// Pool Implementation
// ============================================

export class AIPool {
  private config: Required<PoolConfig>;
  private inFlight = 0;
  private queue: QueuedTask<unknown>[] = [];
  private globalBackoffUntil = 0;
  private consecutiveRateLimits = 0;
  private stats = {
    totalCalls: 0,
    totalPromptTokens: 0,
    totalCompletionTokens: 0,
    estimatedCost: 0,
  };

  constructor(config: PoolConfig) {
    this.config = {
      maxConcurrency: config.maxConcurrency,
      model: config.model,
      baseDelayMs: config.baseDelayMs ?? 2000,
      maxRetries: config.maxRetries ?? 6,
      debug: config.debug ?? false,
    };
  }

  /**
   * Execute a generateObject call through the pool
   */
  async generateObject<T extends z.ZodType>(
    options: GenerateOptions<T>
  ): Promise<z.infer<T>> {
    return this.enqueue(() => this.executeWithRetry(options));
  }

  /**
   * Get current pool stats
   */
  getStats() {
    return {
      inFlight: this.inFlight,
      queued: this.queue.length,
      maxConcurrency: this.config.maxConcurrency,
      backoffActive: Date.now() < this.globalBackoffUntil,
    };
  }

  /**
   * Get usage statistics (calls, tokens, cost)
   */
  getUsageStats() {
    return { ...this.stats };
  }

  // ============================================
  // Private Methods
  // ============================================

  private async enqueue<T>(execute: () => Promise<T>): Promise<T> {
    // Wait for global backoff if active
    await this.waitForBackoff();

    // If we have capacity, execute immediately
    if (this.inFlight < this.config.maxConcurrency) {
      return this.runTask(execute);
    }

    // Otherwise, queue and wait
    return new Promise<T>((resolve, reject) => {
      this.queue.push({
        execute: execute as () => Promise<unknown>,
        resolve: resolve as (value: unknown) => void,
        reject,
      });
      this.log(`Queued task (queue size: ${this.queue.length})`);
    });
  }

  private async runTask<T>(execute: () => Promise<T>): Promise<T> {
    this.inFlight++;
    this.log(`Starting task (in-flight: ${this.inFlight})`);

    try {
      const result = await execute();
      this.consecutiveRateLimits = 0; // Reset on success
      return result;
    } finally {
      this.inFlight--;
      this.log(`Completed task (in-flight: ${this.inFlight})`);
      this.processQueue();
    }
  }

  private processQueue() {
    if (this.queue.length === 0) return;
    if (this.inFlight >= this.config.maxConcurrency) return;
    if (Date.now() < this.globalBackoffUntil) return;

    const task = this.queue.shift()!;
    this.runTask(task.execute)
      .then(task.resolve)
      .catch(task.reject);
  }

  private async waitForBackoff() {
    const now = Date.now();
    if (now < this.globalBackoffUntil) {
      const waitMs = this.globalBackoffUntil - now;
      this.log(`Waiting for global backoff: ${waitMs}ms`);
      await sleep(waitMs);
    }
  }

  private applyRateLimitBackoff() {
    this.consecutiveRateLimits++;
    // Exponential backoff: 2s, 4s, 8s, 16s, 32s, capped at 60s
    const backoffMs = Math.min(
      this.config.baseDelayMs * Math.pow(2, this.consecutiveRateLimits),
      60000
    );
    this.globalBackoffUntil = Date.now() + backoffMs;
    this.log(`Rate limited! Global backoff for ${backoffMs}ms (consecutive: ${this.consecutiveRateLimits})`);
  }

  private async executeWithRetry<T extends z.ZodType>(
    options: GenerateOptions<T>
  ): Promise<z.infer<T>> {
    let lastError: unknown;

    for (let attempt = 0; attempt < this.config.maxRetries; attempt++) {
      try {
        const result = await (generateObject as any)({
          model: google(this.config.model),
          schema: options.schema,
          messages: options.messages,
          temperature: options.temperature ?? 0.2,
        });

        // Track usage
        this.stats.totalCalls++;
        // Try to extract usage from various possible locations
        const usage = result.usage || (result as any).response?.usage || (result as any).rawResponse?.usage;
        if (usage) {
          const promptTokens = usage.promptTokens || usage.prompt_tokens || usage.inputTokens || 0;
          const completionTokens = usage.completionTokens || usage.completion_tokens || usage.outputTokens || 0;
          this.stats.totalPromptTokens += promptTokens;
          this.stats.totalCompletionTokens += completionTokens;
          this.stats.estimatedCost += this.estimateCost(this.config.model, { promptTokens, completionTokens });
        }

        return result.object as z.infer<T>;
      } catch (error) {
        lastError = error;

        if (this.isRateLimitError(error)) {
          this.applyRateLimitBackoff();
          await this.waitForBackoff();
          // Don't count rate limits against retry attempts as harshly
          if (attempt > 0) attempt--;
        } else if (this.isRetryableError(error)) {
          const delay = this.config.baseDelayMs * Math.pow(2, attempt);
          this.log(`Retryable error, waiting ${delay}ms (attempt ${attempt + 1}/${this.config.maxRetries})`);
          await sleep(delay);
        } else {
          // Non-retryable error, throw immediately
          throw error;
        }
      }
    }

    throw lastError;
  }

  private estimateCost(model: string, usage: { promptTokens: number; completionTokens: number }): number {
    // Pricing per 1M tokens (approximate as of late 2024)
    const pricing: Record<string, { input: number; output: number }> = {
      'gemini-1.5-pro': { input: 3.50, output: 10.50 },
      'gemini-1.5-flash': { input: 0.075, output: 0.30 },
      // Assume Pro pricing for unknown models as a safe upper bound, or 0 if preview
      'gemini-3-pro-preview': { input: 0, output: 0 }, 
    };

    // Normalize model name to find match
    const modelKey = Object.keys(pricing).find(k => model.includes(k));
    const price = modelKey ? pricing[modelKey] : { input: 0, output: 0 };

    const inputCost = (usage.promptTokens / 1_000_000) * price.input;
    const outputCost = (usage.completionTokens / 1_000_000) * price.output;
    
    return inputCost + outputCost;
  }

  private isRateLimitError(error: unknown): boolean {
    if (error instanceof Error) {
      const message = error.message.toLowerCase();
      const anyError = error as any;
      return (
        message.includes('rate limit') ||
        message.includes('429') ||
        message.includes('too many requests') ||
        message.includes('quota') ||
        anyError.status === 429 ||
        anyError.statusCode === 429
      );
    }
    return false;
  }

  private isRetryableError(error: unknown): boolean {
    if (error instanceof Error) {
      const anyError = error as any;
      const status = anyError.status ?? anyError.statusCode;
      // Retry on 5xx errors, timeouts, network errors
      return (
        (status >= 500 && status < 600) ||
        error.message.includes('timeout') ||
        error.message.includes('ECONNRESET') ||
        error.message.includes('ETIMEDOUT') ||
        error.message.includes('other side closed') ||
        error.message.includes('fetch failed') ||
        error.message.includes('socket')
      );
    }
    return false;
  }

  private log(message: string) {
    if (this.config.debug) {
      console.log(`[AIPool] ${message}`);
    }
  }
}

// ============================================
// Helper
// ============================================

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

// ============================================
// Factory Function
// ============================================

export function createAIPool(config: PoolConfig): AIPool {
  return new AIPool(config);
}
