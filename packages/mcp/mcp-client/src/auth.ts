/**
 * OAuth 2.1 client provider for MCP Streamable HTTP servers.
 *
 * Every streamable-http transport gets an `OAuthClientProvider` automatically.
 * No user configuration is required: when the server returns a 401 the SDK
 * discovers the authorization server (RFC 9728), dynamically registers the
 * client (RFC 7591), generates PKCE, and calls `redirectToAuthorization` with
 * the authorization URL. This module provides the storage and interaction
 * adapter the SDK needs.
 *
 * The provider persists tokens, client registration, PKCE verifiers, and
 * discovery state to a per-server JSON file under `~/.dsh/mcp-auth/`.
 * Authorization redirect opens the user's browser; a local HTTP callback
 * server receives the authorization code and resolves it to the caller
 * so the connection supervisor can call `transport.finishAuth(code)`.
 *
 * @module
 */

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { createServer, type Server } from 'node:http'
import { homedir } from 'node:os'
import { join } from 'node:path'
import type {
  OAuthClientInformationMixed,
  OAuthClientMetadata,
  OAuthClientProvider,
  OAuthDiscoveryState,
  OAuthTokens,
} from '@modelcontextprotocol/client'

/**
 * Optional overrides for OAuth behavior on a Streamable HTTP transport.
 * Every field is optional — omitting `auth` entirely still enables OAuth
 * when the server requires it.
 */
export interface OAuthAuthConfig {
  /**
   * Override OAuth client metadata used for dynamic registration.
   * Default provides `redirect_uris` pointing at the local callback server
   * and `client_name: 'dsh-mcp-client'`.
   */
  clientMetadata?: Partial<OAuthClientMetadata>
  /**
   * Local callback port for the OAuth redirect. The provider starts a
   * temporary HTTP server on this port to receive the authorization code.
   * Default 8912.
   */
  callbackPort?: number
  /**
   * Directory for persisting tokens, client registration, and PKCE state.
   * Default `~/.dsh/mcp-auth`.
   */
  storageDir?: string
}

/** Default local callback port for the OAuth redirect server. */
const DEFAULT_CALLBACK_PORT = 8912

/** Default directory for persisting OAuth state. */
function defaultStorageDir(): string {
  return join(homedir(), '.dsh', 'mcp-auth')
}

/** Persisted state for one OAuth server relationship. */
interface PersistedOAuthState {
  tokens?: OAuthTokens
  clientInformation?: OAuthClientInformationMixed
  codeVerifier?: string
  discoveryState?: OAuthDiscoveryState
}

/**
 * Read persisted OAuth state for one server.
 * @param path - Absolute path to the JSON file.
 * @returns The parsed state, or an empty object when the file is absent.
 */
async function readState(path: string): Promise<PersistedOAuthState> {
  try {
    const raw = await readFile(path, 'utf8')
    return JSON.parse(raw) as PersistedOAuthState
  } catch {
    return {}
  }
}

/**
 * Write persisted OAuth state for one server, creating the parent directory
 * when absent.
 * @param path - Absolute path to the JSON file.
 * @param state - The state to persist.
 */
async function writeState(path: string, state: PersistedOAuthState): Promise<void> {
  const dir = join(path, '..')
  await mkdir(dir, { recursive: true })
  await writeFile(path, JSON.stringify(state, null, 2), 'utf8')
}

/** Result of building an auth provider: the provider and its callback server handle. */
export interface AuthProviderHandle {
  /** The `OAuthClientProvider` to pass to `StreamableHTTPClientTransport`. */
  provider: OAuthClientProvider
  /** The callback server that receives the OAuth redirect. */
  callbackServer: CallbackServerHandle
  /**
   * The authorization URL the SDK redirected the user to, set by
   * `redirectToAuthorization`. Undefined until the first redirect.
   */
  authorizationUrl: URL | undefined
}

/**
 * Build a concrete `OAuthClientProvider` for one MCP server and start
 * the local callback server.
 *
 * No user configuration is required. The provider persists all OAuth
 * artifacts to `<storageDir>/<serverName>.json`. When the SDK discovers
 * that a server requires OAuth and calls `redirectToAuthorization`, the
 * provider opens the user's browser and captures the URL for state events.
 *
 * @param serverName - Stable server namespace used for the storage filename.
 * @param overrides - Optional auth config overrides.
 * @returns Handle with the provider, callback server, and authorization URL reference.
 */
export function buildAuthProvider(serverName: string, overrides?: OAuthAuthConfig): AuthProviderHandle {
  const storageDir = overrides?.storageDir ?? defaultStorageDir()
  const statePath = join(storageDir, `${serverName}.json`)
  const callbackPort = overrides?.callbackPort ?? DEFAULT_CALLBACK_PORT
  const redirectUrl = new URL(`http://127.0.0.1:${callbackPort}/oauth/callback`)

  const defaultMetadata: OAuthClientMetadata = {
    redirect_uris: [redirectUrl.toString()],
    client_name: 'dsh-mcp-client',
  }

  const clientMetadata: OAuthClientMetadata = overrides?.clientMetadata !== undefined
    ? { ...defaultMetadata, ...overrides.clientMetadata }
    : defaultMetadata

  const callbackServer = startCallbackServer(callbackPort)

  // The handle's provider field is set after the provider object literal is
  // built; the mutable reference lets redirectToAuthorization write back the
  // authorization URL. Declared as let + reassigned below.
  let authorizationUrl: URL | undefined

  const provider: OAuthClientProvider = {
    get redirectUrl(): URL {
      return redirectUrl
    },

    get clientMetadata(): OAuthClientMetadata {
      return clientMetadata
    },

    async clientInformation(): Promise<OAuthClientInformationMixed | undefined> {
      const state = await readState(statePath)
      return state.clientInformation
    },

    async saveClientInformation(info: OAuthClientInformationMixed): Promise<void> {
      const state = await readState(statePath)
      state.clientInformation = info
      await writeState(statePath, state)
    },

    async tokens(): Promise<OAuthTokens | undefined> {
      const state = await readState(statePath)
      return state.tokens
    },

    async saveTokens(tokens: OAuthTokens): Promise<void> {
      const state = await readState(statePath)
      state.tokens = tokens
      await writeState(statePath, state)
    },

    async redirectToAuthorization(targetUrl: URL): Promise<void> {
      // Capture the URL so the state event can surface it.
      authorizationUrl = targetUrl
      // Open the authorization URL in the user's default browser using
      // platform-native commands. No external dependency required.
      const { exec } = await import('node:child_process')
      const href = targetUrl.toString()
      const command = process.platform === 'darwin'
        ? `open ${JSON.stringify(href)}`
        : process.platform === 'win32'
          ? `start "" ${JSON.stringify(href)}`
          : `xdg-open ${JSON.stringify(href)}`
      exec(command)
    },

    async saveCodeVerifier(verifier: string): Promise<void> {
      const state = await readState(statePath)
      state.codeVerifier = verifier
      await writeState(statePath, state)
    },

    async codeVerifier(): Promise<string> {
      const state = await readState(statePath)
      if (state.codeVerifier === undefined) throw new Error('mcp-client: OAuth code verifier not found in persisted state')
      return state.codeVerifier
    },

    async saveDiscoveryState(discoveryState: OAuthDiscoveryState): Promise<void> {
      const state = await readState(statePath)
      state.discoveryState = discoveryState
      await writeState(statePath, state)
    },

    async discoveryState(): Promise<OAuthDiscoveryState | undefined> {
      const state = await readState(statePath)
      return state.discoveryState
    },

    async invalidateCredentials(scope: 'all' | 'client' | 'tokens' | 'verifier' | 'discovery'): Promise<void> {
      const state = await readState(statePath)
      if (scope === 'all' || scope === 'tokens') delete state.tokens
      if (scope === 'all' || scope === 'client') delete state.clientInformation
      if (scope === 'all' || scope === 'verifier') delete state.codeVerifier
      if (scope === 'all' || scope === 'discovery') delete state.discoveryState
      await writeState(statePath, state)
    },
  }

  return {
    provider,
    callbackServer,
    get authorizationUrl() { return authorizationUrl },
  }
}

/** Handle returned by {@link startCallbackServer}. */
export interface CallbackServerHandle {
  /**
   * Promise that resolves with the authorization code when the OAuth
   * redirect callback is received. Resets after each resolution so
   * subsequent auth flows get a fresh promise.
   */
  code: Promise<string>
  /** Stop the callback server. */
  close(): Promise<void>
}

/**
 * Start a local HTTP server to receive the OAuth redirect callback.
 *
 * The server listens on `127.0.0.1:<port>` and handles
 * `GET /oauth/callback?code=<code>` requests. After receiving the code
 * it responds with a user-facing success page and resolves the returned
 * promise.
 *
 * @param port - Port to listen on.
 * @returns Handle with the `code` promise and a `close` function.
 */
export function startCallbackServer(port: number): CallbackServerHandle {
  let gate: PromiseWithResolvers<string> = Promise.withResolvers()
  let server: Server | undefined

  const httpServer = createServer((req, res) => {
    if (req.url === undefined) {
      res.writeHead(400).end('Bad request')
      return
    }
    const url = new URL(req.url, `http://127.0.0.1:${port}`)
    if (url.pathname !== '/oauth/callback') {
      res.writeHead(404).end('Not found')
      return
    }
    const code = url.searchParams.get('code')
    const error = url.searchParams.get('error')
    if (error !== null) {
      const description = url.searchParams.get('error_description') ?? error
      res.writeHead(200, { 'Content-Type': 'text/html' })
        .end(`<html><body><h1>Authorization Failed</h1><p>${description}</p><p>You can close this tab.</p></body></html>`)
      gate.reject(new Error(`mcp-client: OAuth authorization error: ${description}`))
      // Reset the gate so subsequent auth flows get a fresh promise.
      gate = Promise.withResolvers()
      return
    }
    if (code === null) {
      res.writeHead(400, { 'Content-Type': 'text/html' })
        .end('<html><body><h1>Missing Code</h1><p>No authorization code in callback.</p></body></html>')
      return
    }
    res.writeHead(200, { 'Content-Type': 'text/html' })
      .end('<html><body><h1>Authorization Successful</h1><p>You can close this tab and return to the terminal.</p></body></html>')
    gate.resolve(code)
    // Reset the gate so subsequent auth flows (e.g. token expiry) get a fresh promise.
    gate = Promise.withResolvers()
  })

  server = httpServer

  const listening: PromiseWithResolvers<void> = Promise.withResolvers()
  httpServer.listen(port, '127.0.0.1', listening.resolve)

  // The code getter always returns the current gate's promise, chained
  // behind the listening promise so callers don't need to separately await it.
  const listeningPromise = listening.promise

  return {
    get code(): Promise<string> {
      return listeningPromise.then(() => gate.promise)
    },
    async close(): Promise<void> {
      // Wait for the server to finish listening before closing; a close
      // before listen settles throws "Server is not running."
      await listeningPromise.catch(() => {})
      if (server === undefined) return
      const s = server
      server = undefined
      await new Promise<void>((resolve, reject) => {
        s.close((error) => { if (error === undefined) resolve(); else reject(error) })
      })
    },
  }
}
