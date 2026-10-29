/**
 * Tests for the mcp-client connection state tracker and emitted events.
 * Isolated file so vi.mock of the MCP SDK doesn't pollute other test suites.
 */
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import SystemPrompt from '@deepseek-ai/dsh-system-prompt'
import ToolRuntime from '@deepseek-ai/dsh-tools'
import type { Config } from '@deepseek-ai/dsh-mcp-client'
import type { McpClientState } from '@deepseek-ai/dsh-mcp-client/src/state.ts'

// ---- Mock MCP SDK ----

const { mockConnect, mockClose, mockListTools, MockClient, instances } = vi.hoisted(() => {
  const mockConnect = vi.fn<() => Promise<void>>()
  const mockClose = vi.fn<() => Promise<void>>()
  const mockListTools = vi.fn<(_params?: Record<string, unknown>) => Promise<unknown>>()
  class MockClient {
    onclose: (() => void) | undefined
    /** Truthy transport makes the supervisor treat the generation as attached, exercising `Client.close()`. */
    transport: Record<string, never> = {}
    connect = mockConnect
    close = mockClose
    listTools = mockListTools
    getServerCapabilities = () => ({ tools: {} })
    getInstructions(): string | undefined { return undefined }
    constructor() { instances.push(this) }
  }
  const instances: MockClient[] = []
  return { mockConnect, mockClose, mockListTools, MockClient, instances }
})

vi.mock('@modelcontextprotocol/client', async importOriginal => ({
  ...await importOriginal<typeof import('@modelcontextprotocol/client')>(),
  Client: MockClient,
}))

vi.mock('@modelcontextprotocol/client/stdio', () => ({
  StdioClientTransport: vi.fn(),
}))

import { apply } from '@deepseek-ai/dsh-mcp-client/src/index.ts'
import { startConnection, resolveReconnectPolicy } from '@deepseek-ai/dsh-mcp-client/src/connection.ts'

// ---- Helpers ----

async function mountRegistry(): Promise<Context> {
  const ctx = new Context()
  await ctx.plugin(SystemPrompt)
  await ctx.plugin(ToolRuntime)
  return ctx
}

function stdioConfig(reconnect?: Config['reconnect']): Config {
  return {
    transport: 'stdio',
    serverName: 'srv',
    command: 'echo',
    args: [],
    env: {},
    cwd: '',
    toolCallTimeoutMs: 60_000,
    failOnStartupError: false,
    ...reconnect === undefined ? {} : { reconnect },
  }
}

function listing(...names: string[]): { tools: { name: string; inputSchema: { type: string } }[]; nextCursor: undefined } {
  return {
    tools: names.map(name => ({ name, inputSchema: { type: 'object' } })),
    nextCursor: undefined,
  }
}

// ---- Tests ----

describe('connection state events', () => {
  let ctx: Context

  beforeEach(async () => {
    vi.clearAllMocks()
    instances.length = 0
    mockConnect.mockResolvedValue(undefined)
    mockClose.mockImplementation(function (this: { onclose?: () => void }) {
      this.onclose?.()
      return Promise.resolve()
    })
    mockListTools.mockResolvedValue(listing('remote'))
    ctx = await mountRegistry()
  })

  it('emits connecting then connected on successful startup', async () => {
    const states: McpClientState[] = []
    ctx.on('mcp-client/state', (_serverName, state) => {
      states.push({ ...state })
    })

    await apply(ctx, stdioConfig())

    const phases = states.map(s => s.phase)
    expect(phases).toContain('connecting')
    expect(phases).toContain('connected')
    const connected = states.find(s => s.phase === 'connected')
    expect(connected).toBeDefined()
    expect(connected!.serverName).toBe('srv')
    expect(connected!.transport).toBe('stdio')
    expect(connected!.connectedAt).toBeGreaterThan(0)
    expect(connected!.failedAttempts).toBe(0)
  })

  it('emits backoff then reconnecting then connected on reconnect', async () => {
    const states: McpClientState[] = []
    ctx.on('mcp-client/state', (_serverName, state) => {
      states.push({ ...state })
    })

    await apply(ctx, stdioConfig({ initialDelayMs: 5, maxDelayMs: 40, maxAttempts: 5 }))
    states.length = 0 // Clear startup events

    mockListTools.mockResolvedValue(listing('revived'))
    instances[0]!.onclose?.()

    await vi.waitFor(() => { expect(ctx.tools.get('mcp__srv__revived')).toBeDefined() })

    const phases = states.map(s => s.phase)
    expect(phases).toContain('backoff')
    expect(phases).toContain('reconnecting')
    expect(phases).toContain('connected')
    const backoff = states.find(s => s.phase === 'backoff')
    expect(backoff!.nextReconnectAt).toBeGreaterThan(0)
    expect(backoff!.failedAttempts).toBe(1)
  })

  it('emits exhausted when reconnect budget runs out', async () => {
    const states: McpClientState[] = []
    ctx.on('mcp-client/state', (_serverName, state) => {
      states.push({ ...state })
    })

    await apply(ctx, stdioConfig({ initialDelayMs: 2, maxDelayMs: 8, maxAttempts: 1 }))
    states.length = 0

    mockConnect.mockRejectedValue(new Error('server gone'))
    instances[0]!.onclose?.()

    await vi.waitFor(() => {
      expect(states.some(s => s.phase === 'exhausted')).toBe(true)
    })

    const exhausted = states.find(s => s.phase === 'exhausted')
    expect(exhausted!.serverName).toBe('srv')
    expect(exhausted!.connectedAt).toBeUndefined()
    expect(exhausted!.pid).toBeUndefined()
  })

  it('emits disposed on plugin dispose', async () => {
    const states: McpClientState[] = []
    ctx.on('mcp-client/state', (_serverName, state) => {
      states.push({ ...state })
    })

    const fiber = ctx.plugin({ name: 'mcp-client', inject: ['tools'], apply }, stdioConfig())
    await fiber

    states.length = 0
    await fiber.dispose()

    expect(states.some(s => s.phase === 'disposed')).toBe(true)
  })

  it('exposes queryable state via connection handle', async () => {
    const policy = resolveReconnectPolicy(undefined, 'test')
    const handle = startConnection(ctx, stdioConfig(), policy)
    await handle.ready

    const state = handle.state()
    expect(state.serverName).toBe('srv')
    expect(state.transport).toBe('stdio')
    expect(state.phase).toBe('connected')
    expect(state.maxAttempts).toBe(10)
    expect(state.failedAttempts).toBe(0)

    await handle.dispose()

    const disposed = handle.state()
    expect(disposed.phase).toBe('disposed')
  })

  it('tracks failedAttempts through backoff cycles', async () => {
    const states: McpClientState[] = []
    ctx.on('mcp-client/state', (_serverName, state) => {
      states.push({ ...state })
    })

    await apply(ctx, stdioConfig({ initialDelayMs: 2, maxDelayMs: 8, maxAttempts: 3 }))
    states.length = 0

    mockConnect.mockRejectedValue(new Error('server gone'))
    instances[0]!.onclose?.()

    await vi.waitFor(() => {
      expect(states.some(s => s.phase === 'exhausted')).toBe(true)
    })

    const backoffs = states.filter(s => s.phase === 'backoff')
    expect(backoffs.length).toBeGreaterThanOrEqual(2)
    expect(backoffs[0]!.failedAttempts).toBe(1)
    expect(backoffs[1]!.failedAttempts).toBe(2)
  })

  it('records lastError from failed attempts in state events', async () => {
    const states: McpClientState[] = []
    ctx.on('mcp-client/state', (_serverName, state) => {
      states.push({ ...state })
    })

    await apply(ctx, stdioConfig({ initialDelayMs: 2, maxDelayMs: 8, maxAttempts: 2 }))
    states.length = 0

    mockConnect.mockRejectedValue(new Error('server gone'))
    instances[0]!.onclose?.()

    await vi.waitFor(() => {
      expect(states.some(s => s.phase === 'exhausted')).toBe(true)
    })

    // After the first reconnect attempt fails, the second backoff event
    // carries the lastError from that failed attempt.
    const backoffsWithError = states.filter(s => s.phase === 'backoff' && s.lastError !== undefined)
    expect(backoffsWithError.length).toBeGreaterThanOrEqual(1)
    expect(backoffsWithError[0]!.lastError).toContain('server gone')
  })
})
