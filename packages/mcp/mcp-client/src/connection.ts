/**
 * Connection supervisor: owns the MCP client/transport generations for one
 * plugin instance, keeps the harness tool registry in sync with the live
 * generation, and — when the connection drops — restarts the configured
 * server with bounded exponential backoff.
 *
 * One outage shares one attempt budget (`maxAttempts` consecutive failed
 * attempts, delays doubling from `initialDelayMs` up to `maxDelayMs`). A
 * connection that stays up past the stability window closes the outage, so
 * the next disconnect starts a fresh budget while a crash-looping server —
 * even one whose connects briefly succeed — still exhausts the cap instead of
 * restarting forever. Exhaustion unregisters the server's tools and stops;
 * disposal (including HMR) is the only way back from that state.
 *
 * The supervisor emits `mcp-client/state` events at each lifecycle transition
 * and exposes PID for stdio transports when accessible.
 *
 * @module
 */

import { Client, UnauthorizedError } from '@modelcontextprotocol/client'
import type { StdioClientTransport } from '@modelcontextprotocol/client/stdio'
import type { Context } from '@deepseek-ai/cordis'
import { assertNever, type JsonValue } from '@deepseek-ai/dsh-util-values'
import type { ServerContext } from './server-context.ts'
import { MAX_TIMER_DELAY_MS } from '@deepseek-ai/dsh-timeout'
import { createTransport } from './transport.ts'
import type { TransportResult } from './transport.ts'
import { syncTools } from './tools.ts'
import type { ToolBridgeOptions, ToolDisposers } from './tools.ts'
import type { Config } from './index.ts'
import type { AuthProviderHandle } from './auth.ts'
import { McpClientStateTracker } from './state.ts'
import type { McpClientState } from './state.ts'
// Side-effect type import: declaration-merges `mcp-client/state` onto Events.
import type {} from './state.ts'

/** Automatic reconnect policy for one MCP server connection. */
export interface ReconnectConfig {
  /** Reconnect automatically after a lost connection (default true). */
  enabled?: boolean
  /** First reconnect delay in milliseconds; doubles per consecutive failed attempt (default 500). */
  initialDelayMs?: number
  /** Backoff ceiling in milliseconds; also the uptime after which the attempt budget resets (default 30000). */
  maxDelayMs?: number
  /** Consecutive failed attempts per outage before giving up for good (default 10). */
  maxAttempts?: number
}

/** Defaults shared by the Config schema and {@link resolveReconnectPolicy}. */
export const RECONNECT_DEFAULTS: Required<ReconnectConfig> = Object.freeze({
  enabled: true,
  initialDelayMs: 500,
  maxDelayMs: 30_000,
  maxAttempts: 10,
})

/** Default UTF-8 byte limit for attributed server instructions. */
export const DEFAULT_MAX_INSTRUCTION_BYTES = 32_768

// The SDK's stdio transport owns two two-second termination grace periods.
// Keep one additional second for the process-close event that proves the old
// generation is gone; timing out fails closed instead of overlapping children.
const GENERATION_CLOSE_TIMEOUT_MS = 5_000

/** Fully resolved reconnect policy captured at plugin load. */
export type ResolvedReconnectPolicy = Readonly<Required<ReconnectConfig>>

/**
 * The one explicit resolve step from raw reconnect config to the policy the
 * supervisor runs. Programmatic construction may bypass Schemastery
 * normalization, so every default and bound is re-judged here — misconfiguration
 * fails the plugin instance at load.
 *
 * @param config - Raw `reconnect` config; omission uses the defaults.
 * @param path - Diagnostic prefix naming the config location in thrown messages.
 * @returns The frozen resolved policy.
 */
export function resolveReconnectPolicy(config: ReconnectConfig | undefined, path: string): ResolvedReconnectPolicy {
  if (config !== undefined) {
    for (const key of Object.keys(config)) {
      if (!Object.hasOwn(RECONNECT_DEFAULTS, key)) throw new Error(`${path}.${key} is not a reconnect option`)
    }
  }
  const enabled = config?.enabled ?? RECONNECT_DEFAULTS.enabled
  const initialDelayMs = config?.initialDelayMs ?? RECONNECT_DEFAULTS.initialDelayMs
  const maxDelayMs = config?.maxDelayMs ?? RECONNECT_DEFAULTS.maxDelayMs
  const maxAttempts = config?.maxAttempts ?? RECONNECT_DEFAULTS.maxAttempts
  /* jscpd:ignore-start — domain-specific delay validation parallels llm retry-policy; not extractable */
  if (!Number.isFinite(initialDelayMs) || initialDelayMs <= 0 || initialDelayMs > MAX_TIMER_DELAY_MS) {
    throw new Error(`${path}.initialDelayMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`)
  }
  if (!Number.isFinite(maxDelayMs) || maxDelayMs <= 0 || maxDelayMs > MAX_TIMER_DELAY_MS) {
    throw new Error(`${path}.maxDelayMs must be a positive finite number no greater than ${MAX_TIMER_DELAY_MS}`)
  }
  if (initialDelayMs > maxDelayMs) {
    throw new Error(`${path}.initialDelayMs must be less than or equal to maxDelayMs`)
  }
  if (!Number.isInteger(maxAttempts) || maxAttempts < 1) {
    throw new Error(`${path}.maxAttempts must be a positive integer`)
  }
  /* jscpd:ignore-end */
  return Object.freeze({ enabled, initialDelayMs, maxDelayMs, maxAttempts })
}

/** Result from the initial connection attempt, for startup-await semantics. */
export interface ConnectionOutcome {
  /** If the initial connection or tool sync failed, the error; otherwise absent. */
  error?: unknown
}

/** Handle for one plugin instance's supervised connection. */
export interface ConnectionHandle extends ServerContext {
  /**
   * Settles when the first connection attempt completes (success or failure).
   * The supervisor enters its reconnect loop regardless; the caller decides
   * whether a failed startup is fatal via `failOnStartupError`.
   */
  ready: Promise<ConnectionOutcome>
  /**
   * Stop reconnection, close the negotiating transport or live client, wait
   * for the in-flight attempt and queued tool syncs to quiesce, then
   * unregister every tool this server still owns.
   */
  dispose(): Promise<void>
  /** Current connection state snapshot. */
  state(): Readonly<McpClientState>
}

/**
 * Start the supervised connection for one MCP server and keep it alive per
 * the reconnect policy.
 *
 * @param ctx - Cordis context providing the `tools` registry and logger.
 * @param config - Resolved plugin config selecting the transport and server identity.
 * @param policy - Resolved reconnect policy from {@link resolveReconnectPolicy}.
 * @param authHandle - OAuth provider handle for Streamable HTTP transports; undefined for stdio.
 * @returns Handle with a `ready` promise for startup-await and a `dispose` for teardown.
 */
export function startConnection(
  ctx: Context, config: Config, policy: ResolvedReconnectPolicy, authHandle?: AuthProviderHandle,
): ConnectionHandle {
  const label = `mcp-client(${config.serverName})`
  const incompleteDisposalMessage = `${label}: transport closure could not be confirmed during disposal — server shutdown may be incomplete`
  const opts: ToolBridgeOptions = {
    registrationFailure: 'contain',
    serverName: config.serverName,
    toolCallTimeoutMs: config.toolCallTimeoutMs,
  }
  // The initial sync uses 'throw' when failOnStartupError is configured, so
  // a registration conflict propagates to the startup-await path. Re-syncs
  // and reconnect syncs always contain conflicts.
  const startupOpts: ToolBridgeOptions = config.failOnStartupError
    ? { ...opts, registrationFailure: 'throw' }
    : opts

  const stateTracker = new McpClientStateTracker(ctx, config.serverName, config.transport, policy.maxAttempts)

  let disposed = false
  const maxInstructionBytes = config.maxInstructionBytes ?? DEFAULT_MAX_INSTRUCTION_BYTES
  let serverInstructions = ''
  /** Current generation: the connecting or connected client; undefined during backoff waits and after final failure. */
  let client: Client | undefined
  /** Transport-aware close operation paired with {@link client}. */
  let closeClient: (() => Promise<boolean>) | undefined
  /** Current stdio transport, for PID access after connect. */
  let currentStdioTransport: StdioClientTransport | undefined
  /** Live tool registrations owned by this server; only {@link enqueueSync} and dispose swap it. */
  let disposers: ToolDisposers = new Map()
  let reconnectTimer: NodeJS.Timeout | undefined
  /** Consecutive failed connection attempts within the current outage. */
  let failedAttempts = 0
  /** When the current generation finished connect + initial sync; undefined while down. */
  let connectedAt: number | undefined
  /** The real error from the first connection attempt, for startup-await diagnostics. */
  let firstAttemptError: unknown

  /** A generation may act only while it is the current one on a live plugin. */
  const isCurrent = (generation: Client): boolean => !disposed && client === generation

  /**
   * Serializes every syncTools call — initial syncs and notification re-syncs
   * across all generations — so two syncs can never interleave their
   * dispose-previous/register-next swap (which would double-dispose one
   * generation and leak another).
   */
  let syncChain: Promise<void> = Promise.resolve()
  function enqueueSync(generation: Client, syncOpts: ToolBridgeOptions = opts): Promise<void> {
    const run = syncChain.then(async () => {
      if (!isCurrent(generation)) return
      disposers = await syncTools(generation, ctx, syncOpts, disposers)
    })
    // The chain tail must survive a failed sync; the enqueuing caller owns reporting.
    syncChain = run.catch(() => {})
    return run
  }

  /** One disconnect decision per generation: the isCurrent guard makes racing close/error signals idempotent. */
  function generationDown(generation: Client): void {
    if (!isCurrent(generation)) return
    client = undefined
    closeClient = undefined
    currentStdioTransport = undefined
    scheduleReconnect()
  }

  /** Decide retry ownership after a failed connection's close barrier settles. */
  function settleFailedGeneration(generation: Client, quiesced: boolean): void {
    if (!isCurrent(generation)) return
    if (!quiesced) {
      client = undefined
      closeClient = undefined
      currentStdioTransport = undefined
      ctx.logger.error(`${label}: failed generation could not confirm transport closure — reconnect stopped to avoid overlapping server processes; reload the plugin or restart the Host to retry`)
      return
    }
    generationDown(generation)
  }

  /** Wait for the transport-owned close signal without letting a broken transport wedge teardown forever. */
  function waitForClose(closed: Promise<void>): Promise<boolean> {
    return new Promise((resolve) => {
      const timeout = setTimeout(() => { resolve(false) }, GENERATION_CLOSE_TIMEOUT_MS)
      timeout.unref()
      void closed.then(() => {
        clearTimeout(timeout)
        resolve(true)
      })
    })
  }

  function scheduleReconnect(): void {
    const lostEstablishedConnection = connectedAt !== undefined
    if (!policy.enabled) {
      const message = lostEstablishedConnection
        ? 'connection lost and reconnect is disabled — registered tools will fail until an HMR reload or Host restart'
        : 'connection failed and reconnect is disabled — no tools were registered; reload the plugin or restart the Host to connect'
      ctx.logger.error(`${label}: ${message}`)
      return
    }
    // A connection that stayed up past the stability window (= maxDelayMs, the
    // longest backoff spacing) ended the previous outage: start a fresh budget.
    if (connectedAt !== undefined && Date.now() - connectedAt >= policy.maxDelayMs) {
      failedAttempts = 0
      stateTracker.resetBudget()
    }
    connectedAt = undefined
    failedAttempts += 1
    stateTracker.setFailedAttempts(failedAttempts)
    if (failedAttempts > policy.maxAttempts) {
      // Enqueue the give-up disposal so it cannot race an in-flight sync's
      // phase-2 swap (which checks isCurrent inside the queue).
      syncChain = syncChain.then(() => {
        for (const dispose of disposers.values()) dispose()
        disposers = new Map()
        serverInstructions = ''
      })
      stateTracker.exhausted()
      ctx.logger.error(`${label}: giving up after ${policy.maxAttempts} consecutive failed reconnect attempts — tools unregistered; reload the plugin or restart the Host to reconnect`)
      return
    }
    const delayMs = Math.min(policy.maxDelayMs, policy.initialDelayMs * 2 ** (failedAttempts - 1))
    const action = lostEstablishedConnection ? 'connection lost; reconnecting' : 'connection failed; retrying'
    ctx.logger.warn(`${label}: ${action} in ${delayMs}ms (attempt ${failedAttempts}/${policy.maxAttempts})`)
    stateTracker.backoff(Date.now() + delayMs)
    reconnectTimer = setTimeout(() => {
      reconnectTimer = undefined
      settling = connectGeneration(false)
    }, delayMs)
    // An armed reconnect timer must never hold the process open on its own.
    reconnectTimer.unref()
  }

  /**
   * One connection attempt: fresh transport + client (the MCP SDK binds a
   * Protocol to one transport for life), connect, then queue the initial tool
   * sync. The startup flag belongs to the attempt rather than the shared sync
   * queue, so an early notification cannot consume strict startup semantics.
   * Every failure funnels through {@link generationDown}; success arms the
   * onclose-driven disconnect path. Never rejects.
   *
   * @param startup - Whether this is the plugin's activation attempt.
   */
  async function connectGeneration(startup: boolean): Promise<void> {
    stateTracker.connecting()
    const generation = new Client(
      { name: 'dsh-mcp-client', version: '0.0.1' },
      {
        capabilities: {},
        versionNegotiation: { mode: 'auto' },
        listChanged: {
          tools: {
            autoRefresh: false,
            debounceMs: 0,
            onChanged: () => { void refreshTools() },
          },
        },
      },
    )
    const closed: PromiseWithResolvers<void> = Promise.withResolvers()
    let attemptSettled = false
    let closeObserved = false
    let transportResult: TransportResult | undefined
    const hasClosed = (): boolean => closeObserved
    client = generation
    closeClient = closeGeneration
    currentStdioTransport = undefined
    generation.onclose = () => {
      closeObserved = true
      closed.resolve()
      // A failed connect owns its close barrier in the catch path below. An
      // established generation can transition down directly from this signal.
      if (attemptSettled) generationDown(generation)
    }
    /** Unattached probes close through their transport; attached clients must also report transport closure. */
    async function closeGeneration(): Promise<boolean> {
      const attached = generation.transport !== undefined
      try {
        await (attached ? generation.close() : transportResult?.transport.close())
      } catch (_error) {
        // Unattached transports may already be dead: fall back to the close
        // barrier, which only a transport close observer can resolve.
        if (!attached) return hasClosed()
      }
      return !attached || hasClosed() || await waitForClose(closed.promise)
    }
    async function refreshTools(): Promise<void> {
      if (!isCurrent(generation)) return
      ctx.logger.info(`${label}: tool list changed, re-syncing`)
      try {
        await enqueueSync(generation)
      } catch (error) {
        // Fetch-phase failure: the previous generation is still registered
        // and `disposers` still owns it — keep serving the last good list.
        if (!disposed) ctx.logger.error(`${label}: tool re-sync failed: ${String(error)}`)
      }
    }
    let instructions: string
    try {
      transportResult = createTransport(config, authHandle?.provider)
      // Capture stderr from stdio servers into the state tracker so crash
      // traces and config errors are visible in the UI via stderrTail.
      const stdioTransport = transportResult.stdioTransport
      currentStdioTransport = stdioTransport
      if (stdioTransport?.stderr) {
        stdioTransport.stderr.on('data', (chunk: Buffer) => {
          stateTracker.pushStderr(chunk.toString('utf8'))
        })
      }
      await generation.connect(transportResult.transport)
      if (hasClosed()) {
        attemptSettled = true
        generationDown(generation)
        return
      }
      if (!isCurrent(generation)) {
        if (!await closeGeneration()) ctx.logger.error(incompleteDisposalMessage)
        return
      }
      const serverText = generation.getInstructions()?.trimEnd() ?? ''
      instructions = serverText ? `### MCP server: ${config.serverName}\n\n${serverText}` : ''
      if (Buffer.byteLength(instructions) > maxInstructionBytes) {
        throw new Error(`${label}: server instructions exceed maxInstructionBytes (${maxInstructionBytes})`)
      }
      await enqueueSync(generation, startup ? startupOpts : opts)
    } catch (error) {
      // OAuth redirect: the SDK called redirectToAuthorization and threw
      // UnauthorizedError. Wait for the user to authorize in the browser,
      // then call finishAuth on the SAME transport and retry connect.
      if (error instanceof UnauthorizedError && authHandle !== undefined && !disposed) {
        const authUrl = authHandle.authorizationUrl
        if (authUrl !== undefined) {
          ctx.logger.info(`${label}: OAuth authorization required — waiting for browser callback`)
          stateTracker.awaitingAuth(authUrl.toString())
          try {
            const code = await authHandle.callbackServer.code
            if (disposed) return
            // finishAuth exchanges the code for tokens on the same transport
            // that started the auth flow. The transport holds the PKCE and
            // resource metadata from the initial discovery; the exchanged
            // tokens persist through the auth provider. transportResult is
            // always set: createTransport runs before generation.connect, and
            // this path runs after connect threw.
            const httpTransport = transportResult?.httpTransport
            if (httpTransport === undefined) {
              throw new Error(
                `${label}: OAuth authorization flow reached a transport without finishAuth`,
              )
            }
            await httpTransport.finishAuth(code)
            if (disposed) return
            // Close the auth-holding generation, then retry from the top with
            // a fresh client + transport (the SDK binds a Protocol to one
            // transport for life) that connects using the persisted tokens.
            const quiesced = await closeGeneration()
            attemptSettled = true
            if (!quiesced) {
              settleFailedGeneration(generation, false)
              return
            }
            settling = connectGeneration(startup)
            return
          } catch (authError) {
            // Auth callback failed (user denied, timeout, etc.) — fall
            // through to normal failure handling.
            if (firstAttemptError === undefined) firstAttemptError = authError
            stateTracker.failed(String(authError))
            if (isCurrent(generation)) ctx.logger.warn(`${label}: OAuth authorization failed: ${String(authError)}`)
            const quiesced = await closeGeneration()
            attemptSettled = true
            settleFailedGeneration(generation, quiesced)
            return
          }
        }
      }
      if (firstAttemptError === undefined) firstAttemptError = error
      stateTracker.failed(String(error))
      // Disposal clears current ownership before it closes the generation, so
      // only a live supervisor reports an attempt failure.
      if (isCurrent(generation)) ctx.logger.warn(`${label}: connection attempt failed: ${String(error)}`)
      const quiesced = await closeGeneration()
      attemptSettled = true
      settleFailedGeneration(generation, quiesced)
      return
    }
    attemptSettled = true
    if (hasClosed()) {
      generationDown(generation)
      return
    }
    if (!isCurrent(generation)) return
    serverInstructions = instructions
    connectedAt = Date.now()
    const pid = currentStdioTransport?.pid ?? undefined
    stateTracker.connected(connectedAt, pid)
    if (failedAttempts > 0) ctx.logger.info(`${label}: reconnected and re-synced tools (attempt ${failedAttempts}/${policy.maxAttempts})`)
  }

  /** The in-flight (or last settled) connection attempt; dispose awaits it for quiescence. */
  let settling = connectGeneration(true)

  // The ready promise settles when the first attempt finishes (regardless of
  // success). If the first attempt fails and reconnect is enabled, the
  // supervisor is already scheduling a retry — ready just reports the outcome.
  const ready: Promise<ConnectionOutcome> = settling.then(() => {
    // After settling: if client is set the initial connect+sync succeeded.
    // If not, the supervisor either scheduled a retry (error logged) or gave
    // up (error logged). Either way the outcome is reported with the real error.
    // Note: settling.then() is a microtask; stdio onclose is a macrotask — so
    // a server that crashes AFTER a successful initial sync cannot flip client
    // to undefined before this continuation runs.
    if (client !== undefined) return {}
    /* v8 ignore next -- defensive: firstAttemptError is always set when connect/sync fails */
    return { error: firstAttemptError ?? new Error(`${label}: initial connection failed`) }
  })

  return {
    ready,
    instructions: () => serverInstructions,
    state(): Readonly<McpClientState> {
      return stateTracker.state
    },
    resources: {
      async request(request, exec): Promise<JsonValue> {
        const generation = client
        if (!generation || connectedAt === undefined) throw new Error(`${label}: server is disconnected`)
        const options = { signal: exec.signal, timeout: config.toolCallTimeoutMs }
        // The SDK answers an unadvertised capability with the same empty page, but
        // writes a console.debug line to stdout, which the ACP profile reserves for JSON-RPC.
        const listable = generation.getServerCapabilities()?.resources !== undefined
        switch (request.method) {
          case 'resources/list':
            return listable
              ? await generation.listResources(
                request.cursor === undefined ? undefined : { cursor: request.cursor }, options,
              ) as JsonValue
              : { resources: [] }
          case 'resources/templates/list':
            return listable
              ? await generation.listResourceTemplates(
                request.cursor === undefined ? undefined : { cursor: request.cursor }, options,
              ) as JsonValue
              : { resourceTemplates: [] }
          case 'resources/read':
            return await generation.readResource({ uri: request.uri }, options) as JsonValue
          /* v8 ignore next 2 -- resource requests are the closed, typed tool operation union */
          default:
            return assertNever(request)
        }
      },
    },
    async dispose(): Promise<void> {
      disposed = true
      serverInstructions = ''
      if (reconnectTimer !== undefined) {
        clearTimeout(reconnectTimer)
        reconnectTimer = undefined
      }
      const close = closeClient
      client = undefined
      closeClient = undefined
      currentStdioTransport = undefined
      if (close !== undefined && !await close()) {
        ctx.logger.error(incompleteDisposalMessage)
      }
      // Quiesce, don't just request it: the in-flight attempt enqueues its
      // sync before settling, so awaiting both leaves `disposers` final.
      await settling
      await syncChain
      for (const dispose of disposers.values()) dispose()
      disposers = new Map()
      stateTracker.disposed()
    },
  }
}
