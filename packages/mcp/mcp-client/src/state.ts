/**
 * Connection supervisor state: typed phases, per-server snapshots, and the
 * Cordis event/service contract for observing MCP client lifecycle.
 *
 * @module
 */

import type { Context } from '@deepseek-ai/cordis'

/**
 * Connection lifecycle phase for one MCP server.
 *
 * - `connecting` — initial connect attempt in progress.
 * - `connected` — live client with synced tools.
 * - `reconnecting` — a reconnect attempt is in progress.
 * - `backoff` — waiting for the next reconnect attempt.
 * - `exhausted` — gave up after `maxAttempts` consecutive failures; tools unregistered.
 * - `disposed` — plugin instance disposed; connection torn down.
 */
export type McpClientPhase =
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'backoff'
  | 'awaiting-auth'
  | 'exhausted'
  | 'disposed'

/** Observable state snapshot for one MCP server connection. */
export interface McpClientState {
  /** Server namespace from plugin config. */
  serverName: string
  /** Transport type. */
  transport: 'stdio' | 'streamable-http'
  /** Current lifecycle phase. */
  phase: McpClientPhase
  /** Consecutive failed connection attempts within the current outage. */
  failedAttempts: number
  /** Max attempts before giving up. */
  maxAttempts: number
  /** Timestamp (ms since epoch) when the current generation connected; undefined while down. */
  connectedAt?: number
  /** Timestamp (ms since epoch) of the next scheduled reconnect; undefined when not in backoff. */
  nextReconnectAt?: number
  /** Error message from the latest failed attempt; undefined when no failure recorded. */
  lastError?: string
  /** For stdio transport: the child process PID, when accessible. */
  pid?: number
  /** For stdio transport: trailing stderr output (last ~8 KB), when available. */
  stderrTail?: string
  /** OAuth authorization URL the user must visit; set during the `awaiting-auth` phase. */
  authorizationUrl?: string
}

/**
 * Mutable state tracker for a single server connection. The connection
 * supervisor updates this object at each lifecycle transition and emits
 * the frozen snapshot through the Cordis event.
 */
export class McpClientStateTracker {
  private static readonly STDERR_MAX_BYTES = 8192
  private _state: McpClientState
  private _ctx: Context

  /**
   * @param ctx - Cordis context for emitting state events.
   * @param serverName - Server namespace.
   * @param transport - Transport type.
   * @param maxAttempts - Reconnect budget from the resolved policy.
   */
  constructor(ctx: Context, serverName: string, transport: 'stdio' | 'streamable-http', maxAttempts: number) {
    this._ctx = ctx
    this._state = {
      serverName,
      transport,
      phase: 'connecting',
      failedAttempts: 0,
      maxAttempts,
    }
  }

  /** Current frozen state snapshot. */
  get state(): Readonly<McpClientState> {
    return { ...this._state }
  }

  /** Transition to `connecting` (initial or reconnect attempt starting). */
  connecting(): void {
    this._state.phase = this._state.failedAttempts > 0 ? 'reconnecting' : 'connecting'
    delete this._state.nextReconnectAt
    delete this._state.stderrTail
    delete this._state.authorizationUrl
    this._emit()
  }

  /** Transition to `connected`. */
  connected(connectedAt: number, pid: number | undefined): void {
    this._state.phase = 'connected'
    this._state.connectedAt = connectedAt
    delete this._state.nextReconnectAt
    delete this._state.authorizationUrl
    if (pid !== undefined) this._state.pid = pid; else delete this._state.pid
    this._emit()
  }

  /**
   * Record a failed attempt error (counter tracked by supervisor via
   * setFailedAttempts). Emits so consumers see error + stderr together.
   */
  failed(error: string): void {
    this._state.lastError = error
    delete this._state.connectedAt
    delete this._state.pid
    this._emit()
  }

  /** Reset the attempt budget (after a stable connection outlives the stability window). */
  resetBudget(): void {
    this._state.failedAttempts = 0
  }

  /** Set the failedAttempts counter directly (for outage-tracking parity). */
  setFailedAttempts(n: number): void {
    this._state.failedAttempts = n
  }

  /**
   * Append a stderr chunk, keeping only the last {@link STDERR_MAX_BYTES} bytes.
   * Does not emit on its own — the next phase transition picks it up.
   * @param chunk - decoded stderr text from the child process.
   */
  pushStderr(chunk: string): void {
    const current = this._state.stderrTail ?? ''
    const combined = current + chunk
    if (combined.length > McpClientStateTracker.STDERR_MAX_BYTES) {
      this._state.stderrTail = combined.slice(-McpClientStateTracker.STDERR_MAX_BYTES)
    } else {
      this._state.stderrTail = combined
    }
  }

  /** Transition to `awaiting-auth`: the server requires OAuth and the user must authorize in their browser. */
  awaitingAuth(authorizationUrl: string): void {
    this._state.phase = 'awaiting-auth'
    this._state.authorizationUrl = authorizationUrl
    delete this._state.nextReconnectAt
    this._emit()
  }

  /** Transition to `backoff`. */
  backoff(nextReconnectAt: number): void {
    this._state.phase = 'backoff'
    this._state.nextReconnectAt = nextReconnectAt
    delete this._state.connectedAt
    delete this._state.pid
    this._emit()
  }

  /** Transition to `exhausted`. */
  exhausted(): void {
    this._state.phase = 'exhausted'
    delete this._state.connectedAt
    delete this._state.nextReconnectAt
    delete this._state.pid
    this._emit()
  }

  /** Transition to `disposed`. */
  disposed(): void {
    this._state.phase = 'disposed'
    delete this._state.connectedAt
    delete this._state.nextReconnectAt
    delete this._state.pid
    this._emit()
  }

  /** Emit the current state snapshot as a Cordis event. */
  private _emit(): void {
    this._ctx.emit('mcp-client/state', this._state.serverName, this.state)
  }
}

// ---- Cordis event declaration ----

declare module '@deepseek-ai/cordis' {
  interface Events {
    /**
     * Emitted when an MCP client connection transitions between lifecycle phases.
     * Listeners receive the server name and a frozen state snapshot.
     * @param serverName - the server namespace from config.
     * @param state - frozen connection state snapshot.
     * @mode emit
     */
    'mcp-client/state'(serverName: string, state: Readonly<McpClientState>): void
  }
}
