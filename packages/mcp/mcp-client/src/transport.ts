/**
 * Transport factory: creates the appropriate MCP transport based on the
 * plugin's resolved config. Stdio spawns a child process (with credential
 * scrubbing); Streamable HTTP connects to a URL with optional OAuth auth.
 *
 * @module
 */

import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js'
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js'
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js'
import type { OAuthClientProvider } from '@modelcontextprotocol/sdk/client/auth.js'
import { scrubbedParentEnv } from '@deepseek-ai/dsh-subprocess'
import type { Config } from './index.ts'

/**
 * The subprocess seam's scrubbed parent env (credential-shaped and stale
 * `DSH_*` names dropped), plus the spec's explicit env. The MCP SDK owns the
 * actual spawn, so this transport shares the scrub definition rather than the
 * spawn path.
 */
function buildChildEnv(extra: Record<string, string>): Record<string, string> {
  return { ...scrubbedParentEnv(), ...extra }
}

/** Result of creating an MCP transport, including transport-specific metadata. */
export interface TransportResult {
  /** The MCP transport ready for `client.connect()`. */
  transport: Transport
  /** For stdio: the `StdioClientTransport` instance, exposing `.pid` after connect. */
  stdioTransport?: StdioClientTransport
}

/**
 * Create an MCP transport from the resolved plugin config.
 *
 * @param config - Resolved plugin config discriminated on `transport`.
 * @param authProvider - OAuth provider for Streamable HTTP transports; ignored for stdio.
 * @returns Transport and transport-specific metadata.
 */
export function createTransport(config: Config, authProvider?: OAuthClientProvider): TransportResult {
  switch (config.transport) {
    case 'stdio': {
      const stdioTransport = new StdioClientTransport({
        command: config.command,
        args: config.args,
        env: buildChildEnv(config.env),
        cwd: config.cwd,
        stderr: 'pipe',
      })
      return { transport: stdioTransport, stdioTransport }
    }
    case 'streamable-http':
      // The MCP SDK's StreamableHTTPClientTransport has optional callback
      // properties typed without `| undefined` (exactOptionalPropertyTypes
      // mismatch with the Transport interface); the SDK constructed the
      // object, so the cast records only that widening.
      return {
        transport: new StreamableHTTPClientTransport(
          new URL(config.url),
          {
            ...authProvider !== undefined ? { authProvider } : {},
            requestInit: { headers: config.headers },
          },
        ) as Transport,
      }
  }
}
