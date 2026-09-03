/**
 * Tests for the OAuth auth provider and callback server.
 */
import { describe, expect, it, afterEach } from 'vitest'
import { readFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { randomUUID } from 'node:crypto'
import type { OAuthTokens, OAuthClientInformationMixed } from '@modelcontextprotocol/sdk/shared/auth.js'
import { buildAuthProvider, startCallbackServer } from '@deepseek-ai/dsh-mcp-client/src/auth.ts'

let portSeq = 19200

/** Allocate a unique port for each test to avoid collisions. */
function nextPort(): number {
  return portSeq++
}

describe('buildAuthProvider', () => {
  let storageDir: string
  let closeServer: (() => Promise<void>) | undefined

  afterEach(async () => {
    if (closeServer) { await closeServer(); closeServer = undefined }
    if (storageDir) await rm(storageDir, { recursive: true, force: true })
  })

  function build(name: string, overrides?: Parameters<typeof buildAuthProvider>[1]) {
    storageDir = join(tmpdir(), `mcp-auth-test-${randomUUID()}`)
    const port = nextPort()
    const handle = buildAuthProvider(name, { storageDir, callbackPort: port, ...overrides })
    closeServer = () => handle.callbackServer.close()
    return handle
  }

  it('builds a provider with default clientMetadata and redirect URL without config', () => {
    const handle = build('test-server')
    expect(handle.provider.redirectUrl).toBeInstanceOf(URL)
    expect(handle.provider.clientMetadata.client_name).toBe('dsh-mcp-client')
    expect(handle.provider.clientMetadata.redirect_uris).toHaveLength(1)
    expect(handle.authorizationUrl).toBeUndefined()
  })

  it('merges user-provided clientMetadata overrides with defaults', () => {
    const handle = build('test-server', { clientMetadata: { client_name: 'my-app' } })
    expect(handle.provider.clientMetadata.client_name).toBe('my-app')
    expect(handle.provider.clientMetadata.redirect_uris).toHaveLength(1)
  })

  it('persists and reads tokens', async () => {
    const { provider } = build('test-server')

    expect(await provider.tokens()).toBeUndefined()

    const tokens: OAuthTokens = {
      access_token: 'access-123',
      token_type: 'bearer',
      refresh_token: 'refresh-456',
    }
    await provider.saveTokens(tokens)
    expect(await provider.tokens()).toEqual(tokens)

    const raw = await readFile(join(storageDir, 'test-server.json'), 'utf8')
    const state = JSON.parse(raw)
    expect(state.tokens.access_token).toBe('access-123')
  })

  it('persists and reads client information', async () => {
    const { provider } = build('test-server')

    expect(await provider.clientInformation()).toBeUndefined()

    const info: OAuthClientInformationMixed = {
      client_id: 'client-789',
      client_secret: 'secret-abc',
    }
    await provider.saveClientInformation!(info)
    expect(await provider.clientInformation()).toEqual(info)
  })

  it('persists and reads code verifier', async () => {
    const { provider } = build('test-server')

    await expect(provider.codeVerifier()).rejects.toThrow('code verifier not found')

    await provider.saveCodeVerifier('verifier-xyz')
    expect(await provider.codeVerifier()).toBe('verifier-xyz')
  })

  it('persists and reads discovery state', async () => {
    const { provider } = build('test-server')

    expect(await provider.discoveryState!()).toBeUndefined()

    const discoveryState = {
      authorizationServerUrl: 'https://auth.example.com',
    }
    await provider.saveDiscoveryState!(discoveryState)
    expect(await provider.discoveryState!()).toEqual(discoveryState)
  })

  it('creates storage directory when absent', async () => {
    storageDir = join(tmpdir(), `mcp-auth-test-${randomUUID()}`, 'nested', 'dir')
    const port = nextPort()
    const handle = buildAuthProvider('test-server', { storageDir, callbackPort: port })
    closeServer = () => handle.callbackServer.close()

    await handle.provider.saveTokens({ access_token: 'test', token_type: 'bearer' })
    const raw = await readFile(join(storageDir, 'test-server.json'), 'utf8')
    expect(JSON.parse(raw).tokens.access_token).toBe('test')
  })

  it('preserves existing state when saving new fields', async () => {
    const { provider } = build('test-server')

    await provider.saveTokens({ access_token: 'a', token_type: 'bearer' })
    await provider.saveCodeVerifier('v')
    await provider.saveClientInformation!({ client_id: 'c' })

    const raw = JSON.parse(await readFile(join(storageDir, 'test-server.json'), 'utf8'))
    expect(raw.tokens.access_token).toBe('a')
    expect(raw.codeVerifier).toBe('v')
    expect(raw.clientInformation.client_id).toBe('c')
  })

  it('invalidateCredentials selectively clears persisted state', async () => {
    const { provider } = build('test-server')

    await provider.saveTokens({ access_token: 'a', token_type: 'bearer' })
    await provider.saveCodeVerifier('v')
    await provider.saveClientInformation!({ client_id: 'c' })
    await provider.saveDiscoveryState!({ authorizationServerUrl: 'https://auth.test' })

    await provider.invalidateCredentials!('tokens')
    expect(await provider.tokens()).toBeUndefined()
    expect(await provider.clientInformation()).toEqual({ client_id: 'c' })

    await provider.invalidateCredentials!('all')
    expect(await provider.clientInformation()).toBeUndefined()
    expect(await provider.codeVerifier().catch(() => 'missing')).toBe('missing')
    expect(await provider.discoveryState!()).toBeUndefined()
  })

  it('captures authorizationUrl when redirectToAuthorization is called', async () => {
    const handle = build('test-server')
    expect(handle.authorizationUrl).toBeUndefined()
    // redirectToAuthorization tries to exec 'open' which may fail in CI,
    // but it still captures the URL before that.
    try {
      await handle.provider.redirectToAuthorization(new URL('https://auth.example.com/authorize?code=123'))
    } catch { /* exec may fail in headless CI */ }
    expect(handle.authorizationUrl).toBeInstanceOf(URL)
    expect(handle.authorizationUrl!.toString()).toBe('https://auth.example.com/authorize?code=123')
  })
})

describe('startCallbackServer', () => {
  it('receives an authorization code via GET /oauth/callback', async () => {
    const port = nextPort()
    const handle = startCallbackServer(port)
    try {
      const codePromise = handle.code
      const response = await fetch(`http://127.0.0.1:${port}/oauth/callback?code=test-code-123`)
      expect(response.status).toBe(200)
      const text = await response.text()
      expect(text).toContain('Authorization Successful')

      const code = await codePromise
      expect(code).toBe('test-code-123')
    } finally {
      await handle.close()
    }
  })

  it('rejects with an error response when OAuth returns an error', async () => {
    const port = nextPort()
    const handle = startCallbackServer(port)
    try {
      const codePromise = handle.code.catch((error: unknown) => error as Error)
      const response = await fetch(`http://127.0.0.1:${port}/oauth/callback?error=access_denied&error_description=User+denied`)
      expect(response.status).toBe(200)
      const text = await response.text()
      expect(text).toContain('Authorization Failed')

      const result = await codePromise
      expect(result).toBeInstanceOf(Error)
      expect((result as Error).message).toContain('OAuth authorization error: User denied')
    } finally {
      await handle.close()
    }
  })

  it('resets after receiving a code so subsequent flows work', async () => {
    const port = nextPort()
    const handle = startCallbackServer(port)
    try {
      const first = handle.code
      await fetch(`http://127.0.0.1:${port}/oauth/callback?code=first`)
      expect(await first).toBe('first')

      const second = handle.code
      await fetch(`http://127.0.0.1:${port}/oauth/callback?code=second`)
      expect(await second).toBe('second')
    } finally {
      await handle.close()
    }
  })

  it('returns 404 for non-callback paths', async () => {
    const port = nextPort()
    const handle = startCallbackServer(port)
    try {
      const response = await fetch(`http://127.0.0.1:${port}/other`)
      expect(response.status).toBe(404)
    } finally {
      await handle.close()
    }
  })

  it('returns 400 when no code is present in callback', async () => {
    const port = nextPort()
    const handle = startCallbackServer(port)
    try {
      const response = await fetch(`http://127.0.0.1:${port}/oauth/callback`)
      expect(response.status).toBe(400)
      const text = await response.text()
      expect(text).toContain('Missing Code')
    } finally {
      await handle.close()
    }
  })
})
